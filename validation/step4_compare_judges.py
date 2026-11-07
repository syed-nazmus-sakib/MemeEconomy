"""
step4_compare_judges.py
========================
Compares κ, CRA values, GF values across all 4 judges:
  claude-sonnet-4-6 (external, baseline)
  gpt5_1            (self-evaluator — in tested set)
  gemini3_flash     (self-evaluator — in tested set)
  qwen235b          (self-evaluator — in tested set)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

JUDGES = {
    "claude-sonnet": "validation/judge_results.json",
    "gpt5_1":        "validation/judge_gpt5_1.json",
    "gemini3_flash": "validation/judge_gemini3_flash.json",
    "qwen235b":      "validation/judge_qwen235b.json",
}

def cohens_kappa(h_vals, j_vals):
    n = len(h_vals)
    if n < 4:
        return None
    tp = sum(1 for h,j in zip(h_vals,j_vals) if h==1 and j==1)
    tn = sum(1 for h,j in zip(h_vals,j_vals) if h==0 and j==0)
    fp = sum(1 for h,j in zip(h_vals,j_vals) if h==1 and j==0)
    fn = sum(1 for h,j in zip(h_vals,j_vals) if h==0 and j==1)
    po = (tp+tn)/n
    pe = ((tp+fp)/n)*((tp+fn)/n) + ((tn+fn)/n)*((tn+fp)/n)
    kappa = (po-pe)/(1-pe) if pe < 1 else 0.0
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
    return {"k": round(kappa,4), "po": round(po,4),
            "tp":tp, "tn":tn, "fp":fp, "fn":fn, "n":n,
            "prec": round(prec,3), "rec": round(rec,3)}

def load_judge(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.load(open(p))

def extract_pairs(results, metric):
    """Extract (heuristic, judge) pairs for CRA or GF."""
    pairs = []
    for r in results:
        if metric == "cra":
            judge = r.get("judge_cra","")
            if judge not in ("YES","NO"):
                continue
            heur = 1 if r["cra_score"] >= 0.5 else 0
            j    = 1 if judge == "YES" else 0
            pairs.append((heur, j, r))
        elif metric == "gf":
            if r["meme_tier"] != 5:
                continue
            judge = r.get("judge_gf","")
            if judge not in ("YES","NO"):
                continue
            heur = 1 if r.get("gf_hit") else 0
            j    = 1 if judge == "YES" else 0
            pairs.append((heur, j, r))
    return pairs

if __name__ == "__main__":
    print("═"*85)
    print("  Multi-Judge Comparison: CRA and GF Validation")
    print("  Heuristic vs 4 LLM Judges")
    print("═"*85)

    all_data = {}
    for judge_name, fpath in JUDGES.items():
        d = load_judge(fpath)
        if d:
            all_data[judge_name] = d
            print(f"  Loaded {judge_name:<20}: {len(d)} results")
        else:
            print(f"  MISSING: {judge_name}")

    # ── CRA comparison ────────────────────────────────────────────────────────
    print(f"\n  ┌── CRA κ (heuristic threshold = 0.5) ──────────────────────────────────────┐")
    print(f"  │ {'Judge':<22} {'n':>4}  {'κ':>7}  {'Agree':>6}  {'Prec':>6}  {'Rec':>5}  "
          f"{'Judge YES%':>10}  {'Heur YES%':>10} │")
    print(f"  │ {'─'*75} │")

    cra_kappas = {}
    for jname, results in all_data.items():
        pairs = extract_pairs(results, "cra")
        if not pairs:
            continue
        h = [p[0] for p in pairs]
        j = [p[1] for p in pairs]
        stats = cohens_kappa(h, j)
        judge_yes = np.mean(j)*100
        heur_yes  = np.mean(h)*100
        cra_kappas[jname] = stats["k"]
        interp = "★ subst" if stats["k"]>=0.6 else ("✓ mod" if stats["k"]>=0.4 else "✗ fair/poor")
        print(f"  │ {jname:<22} {stats['n']:>4}  {stats['k']:>7.4f}  {stats['po']:>5.1%}  "
              f"{stats['prec']:>5.3f}  {stats['rec']:>4.3f}  "
              f"{judge_yes:>9.1f}%  {heur_yes:>9.1f}%  {interp} │")
    print(f"  └{'─'*76}┘")

    # ── GF comparison ─────────────────────────────────────────────────────────
    print(f"\n  ┌── GF κ (T5 rounds only) ───────────────────────────────────────────────────┐")
    print(f"  │ {'Judge':<22} {'n':>4}  {'κ':>7}  {'Agree':>6}  {'Prec':>6}  {'Rec':>5}  "
          f"{'Judge YES%':>10}  {'Heur YES%':>10} │")
    print(f"  │ {'─'*75} │")

    gf_kappas = {}
    for jname, results in all_data.items():
        pairs = extract_pairs(results, "gf")
        if not pairs:
            continue
        h = [p[0] for p in pairs]
        j = [p[1] for p in pairs]
        stats = cohens_kappa(h, j)
        judge_yes = np.mean(j)*100
        heur_yes  = np.mean(h)*100
        gf_kappas[jname] = stats["k"]
        interp = "★ subst" if stats["k"]>=0.6 else ("✓ mod" if stats["k"]>=0.4 else "✗ fair/poor")
        print(f"  │ {jname:<22} {stats['n']:>4}  {stats['k']:>7.4f}  {stats['po']:>5.1%}  "
              f"{stats['prec']:>5.3f}  {stats['rec']:>4.3f}  "
              f"{judge_yes:>9.1f}%  {heur_yes:>9.1f}%  {interp} │")
    print(f"  └{'─'*76}┘")

    # ── Inter-judge agreement ─────────────────────────────────────────────────
    print(f"\n  ┌── Inter-Judge Agreement (do the judges agree with each other?) ──────────┐")
    judge_names = list(all_data.keys())
    for metric in ["cra","gf"]:
        print(f"\n  {metric.upper()} inter-judge κ:")
        for i in range(len(judge_names)):
            for j in range(i+1, len(judge_names)):
                jA, jB = judge_names[i], judge_names[j]
                resA   = all_data[jA]
                resB_map = {r["uid"]: r for r in all_data[jB]}
                pairs = []
                for r in resA:
                    uid = r["uid"]
                    if uid not in resB_map:
                        continue
                    rB = resB_map[uid]
                    if metric == "cra":
                        lA = r.get("judge_cra","")
                        lB = rB.get("judge_cra","")
                    else:
                        if r["meme_tier"] != 5:
                            continue
                        lA = r.get("judge_gf","")
                        lB = rB.get("judge_gf","")
                    if lA not in ("YES","NO") or lB not in ("YES","NO"):
                        continue
                    pairs.append((1 if lA=="YES" else 0, 1 if lB=="YES" else 0))
                if pairs:
                    h = [p[0] for p in pairs]
                    j_v = [p[1] for p in pairs]
                    stats = cohens_kappa(h, j_v)
                    agree = sum(1 for a,b in zip(h,j_v) if a==b)/len(h)
                    print(f"    {jA:<22} vs {jB:<22}  κ={stats['k']:>7.4f}  agree={agree:.1%}  n={stats['n']}")
    print(f"  └{'─'*76}┘")

    # ── Per-model GF judge values ─────────────────────────────────────────────
    print(f"\n  ┌── GF% per model, per judge ────────────────────────────────────────────────┐")
    MODELS = ['gpt5_1','gemini3_flash','mistral_large','gemma3_12B','qwen235b']
    header = f"  │ {'Model':<22}"
    for jn in all_data: header += f"  {jn[:10]:>12}"
    header += "  │"
    print(header)
    print(f"  │ {'─'*75} │")
    for model in MODELS:
        row = f"  │ {model:<22}"
        for jname, results in all_data.items():
            t5 = [r for r in results if r['model']==model and r['meme_tier']==5
                  and r.get('judge_gf') in ('YES','NO')]
            if t5:
                gf_pct = np.mean([1 if r['judge_gf']=='YES' else 0 for r in t5])*100
                row += f"  {gf_pct:>11.1f}%"
            else:
                row += f"  {'—':>12}"
        row += "  │"
        print(row)
    print(f"  └{'─'*76}┘")

    # ── Per-model CRA judge values ────────────────────────────────────────────
    print(f"\n  ┌── Judge YES% for CRA per model, per judge ────────────────────────────────┐")
    header = f"  │ {'Model':<22}"
    for jn in all_data: header += f"  {jn[:10]:>12}"
    header += "  │"
    print(header)
    print(f"  │ {'─'*75} │")
    for model in MODELS:
        row = f"  │ {model:<22}"
        for jname, results in all_data.items():
            sub = [r for r in results if r['model']==model
                   and r.get('judge_cra') in ('YES','NO')]
            if sub:
                yes_pct = np.mean([1 if r['judge_cra']=='YES' else 0 for r in sub])*100
                row += f"  {yes_pct:>11.1f}%"
            else:
                row += f"  {'—':>12}"
        row += "  │"
        print(row)
    print(f"  └{'─'*76}┘")

    print(f"\n  Note: Self-evaluator bias check — does each model rate itself higher on GF?")
    for jname in ['gpt5_1','gemini3_flash','qwen235b']:
        if jname not in all_data:
            continue
        results = all_data[jname]
        self_gf = [r for r in results if r['model']==jname and r['meme_tier']==5
                   and r.get('judge_gf') in ('YES','NO')]
        other_gf= [r for r in results if r['model']!=jname and r['meme_tier']==5
                   and r.get('judge_gf') in ('YES','NO')]
        if self_gf and other_gf:
            self_rate  = np.mean([1 if r['judge_gf']=='YES' else 0 for r in self_gf])*100
            other_rate = np.mean([1 if r['judge_gf']=='YES' else 0 for r in other_gf])*100
            print(f"    {jname:<20}: rates self={self_rate:.1f}%  others={other_rate:.1f}%  "
                  f"self-bias={self_rate-other_rate:+.1f}pp")
