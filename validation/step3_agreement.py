"""
step3_agreement.py
==================
Computes Cohen's κ and agreement statistics for CRA and GF validation.

Reads validation/judge_results.json (produced by step2_judge.py).
Reports:
  - Confusion matrices
  - Cohen's κ for CRA and GF separately
  - Agreement breakdown by model and stratum
  - 10 worst disagreement cases for qualitative inspection
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_FILE = Path(__file__).parent / "judge_results.json"


def cohens_kappa(labels_heuristic, labels_judge):
    """Compute Cohen's κ for binary labels (1=positive, 0=negative)."""
    assert len(labels_heuristic) == len(labels_judge)
    n = len(labels_heuristic)
    if n == 0:
        return None

    # Confusion matrix
    tp = sum(1 for h, j in zip(labels_heuristic, labels_judge) if h == 1 and j == 1)
    tn = sum(1 for h, j in zip(labels_heuristic, labels_judge) if h == 0 and j == 0)
    fp = sum(1 for h, j in zip(labels_heuristic, labels_judge) if h == 1 and j == 0)
    fn = sum(1 for h, j in zip(labels_heuristic, labels_judge) if h == 0 and j == 1)

    po = (tp + tn) / n  # observed agreement

    p_pos = ((tp + fp) / n) * ((tp + fn) / n)
    p_neg = ((tn + fn) / n) * ((tn + fp) / n)
    pe = p_pos + p_neg  # expected agreement by chance

    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {
        "kappa": round(kappa, 4),
        "po":    round(po, 4),
        "pe":    round(pe, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": n,
        "precision": round(tp / (tp+fp), 4) if (tp+fp) > 0 else None,
        "recall":    round(tp / (tp+fn), 4) if (tp+fn) > 0 else None,
    }


def interpret_kappa(k):
    if k >= 0.80: return "Almost perfect"
    if k >= 0.60: return "Substantial"
    if k >= 0.40: return "Moderate"
    if k >= 0.20: return "Fair"
    return "Slight/Poor"


if __name__ == "__main__":
    results = json.load(open(RESULTS_FILE))
    print(f"Loaded {len(results)} judge results.\n")

    # ── CRA validation ────────────────────────────────────────────────────────
    # Heuristic positive = CRA ≥ 0.5 (agent used enough tokens to suggest comprehension)
    # Judge positive = "YES" (agent correctly understood the event)
    cra_pairs = []
    for r in results:
        judge = r.get("judge_cra","")
        if judge not in ("YES","NO"):
            continue
        heuristic = 1 if r["cra_score"] >= 0.5 else 0
        judge_bin = 1 if judge == "YES" else 0
        cra_pairs.append((heuristic, judge_bin, r))

    # ── GF validation ─────────────────────────────────────────────────────────
    # Heuristic positive = gf_hit = True (keyword scan fired)
    # Judge positive = "YES" (judge says risk was acknowledged)
    gf_pairs = []
    for r in results:
        if r["meme_tier"] != 5:
            continue
        judge = r.get("judge_gf","")
        if judge not in ("YES","NO"):
            continue
        heuristic = 1 if r["gf_hit"] else 0
        judge_bin = 1 if judge == "YES" else 0
        gf_pairs.append((heuristic, judge_bin, r))

    print("═"*70)
    print("  CRA Validation — Heuristic vs LLM Judge")
    print("═"*70)
    cra_stats = cohens_kappa(
        [p[0] for p in cra_pairs],
        [p[1] for p in cra_pairs]
    )
    print(f"  n = {cra_stats['n']}")
    print(f"  Confusion matrix (heuristic rows, judge cols):")
    print(f"             Judge YES  Judge NO")
    print(f"  Heur YES      {cra_stats['tp']:>4}       {cra_stats['fp']:>4}")
    print(f"  Heur NO       {cra_stats['fn']:>4}       {cra_stats['tn']:>4}")
    print(f"\n  Observed agreement : {cra_stats['po']:.1%}")
    print(f"  Chance agreement   : {cra_stats['pe']:.1%}")
    print(f"  Cohen's κ          : {cra_stats['kappa']:.4f}  ({interpret_kappa(cra_stats['kappa'])})")
    print(f"  Precision          : {cra_stats['precision']}")
    print(f"  Recall             : {cra_stats['recall']}")

    print(f"\n  By model:")
    by_model_cra = defaultdict(list)
    for h, j, r in cra_pairs:
        by_model_cra[r["model"]].append((h, j))
    for model, pairs in sorted(by_model_cra.items()):
        stats = cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs])
        agree = sum(1 for h,j in pairs if h==j)
        print(f"    {model:<22} n={stats['n']:>3}  κ={stats['kappa']:>7.4f}  agree={agree/stats['n']:.1%}")

    print(f"\n  By stratum:")
    by_stratum_cra = defaultdict(list)
    for h, j, r in cra_pairs:
        by_stratum_cra[r["stratum"]].append((h, j))
    for stratum, pairs in sorted(by_stratum_cra.items()):
        stats = cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs])
        agree = sum(1 for h,j in pairs if h==j)
        print(f"    {stratum:<15} n={stats['n']:>3}  κ={stats['kappa']:>7.4f}  agree={agree/stats['n']:.1%}")

    print(f"\n  Top CRA disagreements (heuristic ≠ judge):")
    cra_disagree = [(r, h, j) for h, j, r in cra_pairs if h != j]
    for r, h, j in cra_disagree[:10]:
        heur_label = "CRA≥0.5" if h else "CRA<0.5"
        print(f"    [{heur_label} → Judge={'YES' if j else 'NO'}] "
              f"{r['model']:<18} {r['instance'][:40]}  CRA={r['cra_score']:.3f}")
        summary = r.get("belief",{}).get("event_summary","")[:100]
        print(f"      summary: \"{summary}...\"")

    print()
    print("═"*70)
    print("  GF Validation — Heuristic vs LLM Judge")
    print("═"*70)
    gf_stats = cohens_kappa(
        [p[0] for p in gf_pairs],
        [p[1] for p in gf_pairs]
    )
    print(f"  n = {gf_stats['n']}")
    print(f"  Confusion matrix (heuristic rows, judge cols):")
    print(f"             Judge YES  Judge NO")
    print(f"  Heur YES      {gf_stats['tp']:>4}       {gf_stats['fp']:>4}")
    print(f"  Heur NO       {gf_stats['fn']:>4}       {gf_stats['tn']:>4}")
    print(f"\n  Observed agreement : {gf_stats['po']:.1%}")
    print(f"  Chance agreement   : {gf_stats['pe']:.1%}")
    print(f"  Cohen's κ          : {gf_stats['kappa']:.4f}  ({interpret_kappa(gf_stats['kappa'])})")
    print(f"  Precision          : {gf_stats['precision']}")
    print(f"  Recall             : {gf_stats['recall']}")

    print(f"\n  By model:")
    by_model_gf = defaultdict(list)
    for h, j, r in gf_pairs:
        by_model_gf[r["model"]].append((h, j))
    for model, pairs in sorted(by_model_gf.items()):
        if len(pairs) < 3:
            continue
        stats = cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs])
        agree = sum(1 for h,j in pairs if h==j)
        print(f"    {model:<22} n={stats['n']:>3}  κ={stats['kappa']:>7.4f}  agree={agree/stats['n']:.1%}")

    print(f"\n  Top GF disagreements (heuristic ≠ judge):")
    gf_disagree = [(r, h, j) for h, j, r in gf_pairs if h != j]
    for r, h, j in gf_disagree[:10]:
        heur_label = "keyword HIT" if h else "keyword MISS"
        print(f"    [{heur_label} → Judge={'YES' if j else 'NO'}] "
              f"{r['model']:<18} {r['instance'][:40]}")
        asmt = r.get("belief",{}).get("meme_assessment","")[:100]
        print(f"      meme_assessment: \"{asmt}...\"")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("═"*70)
    print("  VALIDATION SUMMARY")
    print("═"*70)
    print(f"  CRA κ = {cra_stats['kappa']:.4f}  ({interpret_kappa(cra_stats['kappa'])})")
    print(f"  GF  κ = {gf_stats['kappa']:.4f}  ({interpret_kappa(gf_stats['kappa'])})")
    print()
    for metric, k in [("CRA", cra_stats["kappa"]), ("GF", gf_stats["kappa"])]:
        if k >= 0.7:
            print(f"  ✅  {metric}: κ≥0.7 — validated. Add to Appendix F.")
        elif k >= 0.4:
            print(f"  ⚠️   {metric}: κ∈[0.4,0.7) — moderate. Acknowledge limitation.")
        else:
            print(f"  ❌  {metric}: κ<0.4 — poor agreement. Metric needs replacement.")
