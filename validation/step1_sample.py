"""
step1_sample.py
===============
Stratified sampling of 200 Belief blocks for LLM-as-judge validation of CRA and GF.

Strata:
  50  high-CRA  (CRA ≥ 0.8)  — tests whether high token-overlap = real comprehension
  50  low-CRA   (CRA ≤ 0.2)  — tests whether low token-overlap = real confabulation
  50  GF-pos T5 (keyword hit, meme_tier=5)  — keyword scan caught real acknowledgment?
  50  GF-neg T5 (no keyword hit, meme_tier=5) — keyword scan correctly missed?

Spread across 5 models: gpt5_1, gemini3_flash, mistral_large, gemma3_12B, qwen235b
Output: validation/samples.json — one entry per sampled block
"""

import json, glob, random, re
from pathlib import Path
from collections import defaultdict

random.seed(42)

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_FILE    = Path(__file__).parent / "samples.json"

MODELS = ["gpt5_1", "gemini3_flash", "mistral_large", "gemma3_12B", "qwen235b"]

RISK_WORDS = {
    "edgy","offensive","inappropriate","insensitive","dark","problematic",
    "risky","risk","backlash","harmful","violent","violence","ban","banned",
    "removal","remove","removed","flagged","controversial","sensitive",
    "too far","ethically","unethical","unsafe","exploitative","gross",
    "danger","dangerous","disturbing","explicit","crude","vulgar",
}

def tokenize(text):
    return set(re.findall(r"[a-z]+", text.lower()))

def compute_cra(belief, instance_name):
    ref_tokens = tokenize(instance_name.replace("_"," "))
    if not ref_tokens:
        return None
    summary   = belief.get("event_summary", "")
    audience  = belief.get("audience_reaction", "")
    meme_asmt = belief.get("meme_assessment", "")
    full_text = f"{summary} {audience} {meme_asmt}"

    summary_tokens = tokenize(summary)
    entity_recall  = len(ref_tokens & summary_tokens) / len(ref_tokens)

    full_tokens = tokenize(full_text)
    i_event = 1.0 if (ref_tokens & full_tokens) else 0.0

    stop = {"a","an","the","is","are","was","were","it","this","that","and","or",
            "of","in","to","for","with","on","at","by","from","as","be","has","have"}
    content_words = {w for w in tokenize(meme_asmt) if w not in stop and len(w) > 3}
    event_words   = {w for w in ref_tokens if w not in stop and len(w) > 3}
    i_quotes = 1.0 if (content_words & event_words) else 0.0

    return round((entity_recall + i_event + i_quotes) / 3.0, 4)


def gf_keyword_hit(belief):
    full = " ".join([
        belief.get("event_summary",""),
        belief.get("audience_reaction",""),
        belief.get("meme_assessment",""),
    ]).lower()
    return any(w in full for w in RISK_WORDS)


def load_all_blocks():
    """
    Scan all portfolio_run*.json files for the 5 target models.
    Returns list of block dicts with CRA, GF, belief text, metadata.
    """
    blocks = []
    for model in MODELS:
        model_dir = RESULTS_DIR / model
        if not model_dir.exists():
            continue
        for pf in sorted(model_dir.glob("portfolio_run*.json")):
            try:
                data = json.load(open(pf))
            except Exception:
                continue
            rh  = data.get("round_history", [])
            bl  = data.get("bdi_log", [])
            cond = data.get("condition","?")
            bdi_map = {b.get("round", b.get("round_number")): b
                       for b in bl if isinstance(b, dict)}

            for r in rh:
                rn   = r.get("round_number")
                tier = r.get("meme_tier", 0)
                inst = r.get("instance_name","")
                bdi  = bdi_map.get(rn, {})
                belief = bdi.get("belief", {})

                if not isinstance(belief, dict) or not belief.get("event_summary"):
                    continue

                cra = compute_cra(belief, inst)
                if cra is None:
                    continue

                gf_hit = gf_keyword_hit(belief) if tier == 5 else None

                blocks.append({
                    "model":      model,
                    "condition":  cond,
                    "round":      rn,
                    "instance":   inst,
                    "meme_tier":  tier,
                    "cra_score":  cra,
                    "gf_hit":     gf_hit,      # None if not T5
                    "belief":     belief,
                    "source_file": pf.name,
                })
    return blocks


def stratified_sample(blocks, n_per_stratum=50):
    # Separate T5 and non-T5
    t5_blocks     = [b for b in blocks if b["meme_tier"] == 5]
    all_blocks    = blocks  # CRA strata use all rounds

    # Strata
    high_cra = [b for b in all_blocks if b["cra_score"] >= 0.8]
    low_cra  = [b for b in all_blocks if b["cra_score"] <= 0.2]
    gf_pos   = [b for b in t5_blocks  if b["gf_hit"] is True]
    gf_neg   = [b for b in t5_blocks  if b["gf_hit"] is False]

    print(f"Pool sizes: high_CRA={len(high_cra)}  low_CRA={len(low_cra)}  "
          f"GF-pos={len(gf_pos)}  GF-neg={len(gf_neg)}")

    def sample_balanced(pool, n):
        """Sample n items balanced across the 5 models."""
        by_model = defaultdict(list)
        for b in pool:
            by_model[b["model"]].append(b)
        per_model = max(1, n // len(MODELS))
        sampled = []
        for model, items in by_model.items():
            random.shuffle(items)
            sampled.extend(items[:per_model])
        # Top-up if needed
        remaining = [b for b in pool if b not in sampled]
        random.shuffle(remaining)
        sampled.extend(remaining[:max(0, n - len(sampled))])
        return sampled[:n]

    sampled = {
        "high_cra": sample_balanced(high_cra, n_per_stratum),
        "low_cra":  sample_balanced(low_cra,  n_per_stratum),
        "gf_pos":   sample_balanced(gf_pos,   n_per_stratum),
        "gf_neg":   sample_balanced(gf_neg,   n_per_stratum),
    }

    # Assign unique IDs and flatten
    output = []
    seen_ids = set()
    uid = 0
    for stratum, items in sampled.items():
        for b in items:
            uid += 1
            b = dict(b)
            b["uid"]     = uid
            b["stratum"] = stratum
            # Remove large nested belief for dedup check
            key = f"{b['model']}-{b['instance']}-{b['round']}-{b['source_file']}"
            if key in seen_ids:
                continue
            seen_ids.add(key)
            output.append(b)

    return output


if __name__ == "__main__":
    Path(__file__).parent.mkdir(exist_ok=True)
    print("Loading all belief blocks from portfolio files...")
    blocks = load_all_blocks()
    print(f"Total blocks loaded: {len(blocks)}")

    samples = stratified_sample(blocks)
    print(f"\nFinal sample size: {len(samples)}")

    by_stratum = defaultdict(int)
    by_model   = defaultdict(int)
    for s in samples:
        by_stratum[s["stratum"]] += 1
        by_model[s["model"]]     += 1

    print("\nBy stratum:", dict(by_stratum))
    print("By model:  ", dict(by_model))

    with open(OUT_FILE, "w") as f:
        json.dump(samples, f, indent=2, default=str)
    print(f"\nSaved → {OUT_FILE}")
