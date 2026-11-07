#!/usr/bin/env python3
"""Preregistered analysis for the matched pressure-factorial experiment."""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "results" / "pressure_factorial_2026"
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

# Descriptive arms outside the confirmatory 2x2x2 factorial. They carry no
# factor coding, so they are loaded for contrasts/summaries but are excluded
# from factorial_effect_bootstrap and gee_factorial (both filter on
# MANIFEST["core_conditions"]).
SUPPLEMENTARY_CONDITIONS = ["reward_only", "threat_only"]

CONTRASTS = [
    # Preregistered within-factorial contrasts. NOTE: "reward_only_minus_normal"
    # here is the *within-factorial* aggressive-language cell (a1 vs a0 at low
    # target, no elimination) — NOT the supplementary `reward_only` arm below.
    ("reward_only_minus_normal", "pf_t25_e0_a1", "pf_t25_e0_a0"),
    ("target_only_minus_normal", "pf_t50_e0_a0", "pf_t25_e0_a0"),
    ("structural_survival_minus_target_only", "pf_t50_e1_a0", "pf_t50_e0_a0"),
    ("full_minus_structural_survival", "pf_t50_e1_a1", "pf_t50_e1_a0"),
    ("full_minus_normal", "pf_t50_e1_a1", "pf_t25_e0_a0"),
    # Supplementary identification contrasts (rebuttal). Both are compared to
    # `neutral`, the no-target reference arm, because neither supplementary arm
    # states a karma target.
    #   threat_only : menace with NO economic structure -> falsification cell
    #                 for the "this is just adversarial prompting" reading.
    #   reward_only : goal intensity with NO target and NO threat vocabulary.
    ("supp_threat_only_minus_neutral", "threat_only", "neutral"),
    ("supp_reward_only_minus_neutral", "reward_only", "neutral"),
]


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return None, None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def latest_complete_log(model_dir):
    candidates = []
    for path in model_dir.glob("run*.jsonl"):
        try:
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        except Exception:
            continue
        if rows and any(r.get("type") == "run_end" for r in rows):
            candidates.append((path.stat().st_mtime, path, rows))
    return max(candidates, default=(None, None, None), key=lambda x: x[0])


def load_records():
    records = []
    runs = []
    conditions = (MANIFEST["core_conditions"]
                  + MANIFEST["reference_conditions"]
                  + SUPPLEMENTARY_CONDITIONS)
    for condition in conditions:
        for rep in range(1, len(MANIFEST["replicate_seeds"]) + 1):
            for model in MANIFEST["models"]:
                model_dir = RESULTS / condition / f"rep_{rep:02d}" / model
                _, path, rows = latest_complete_log(model_dir)
                if path is None:
                    continue
                start = next(r for r in rows if r.get("type") == "run_start")
                end = next(r for r in reversed(rows) if r.get("type") == "run_end")
                factors = start.get("condition_factors", {})
                round_rows = [r for r in rows if r.get("type") == "round_result"]
                valid = [r for r in round_rows if not r.get("incomplete") and r.get("meme_tier") is not None]
                runs.append({
                    "condition": condition,
                    "model": model,
                    "replicate": rep,
                    "planned": start.get("num_rounds", len(start.get("instances", []))),
                    "observed": len(round_rows),
                    "valid": len(valid),
                    "terminal_reason": end.get("terminal_reason"),
                    "final_karma": end.get("final_karma"),
                    "log": str(path),
                })
                for row in valid:
                    records.append({
                        "condition": condition,
                        "model": model,
                        "replicate": rep,
                        "instance": row["instance"],
                        "episode_cluster": f"{model}|{condition}|{rep}",
                        "high_target": factors.get("high_target"),
                        "elimination": factors.get("elimination"),
                        "aggressive": factors.get("aggressive_language"),
                        "t5": int(row["meme_tier"] == 5),
                        "risk_ack": int(bool(row.get("risk_acknowledged"))),
                        "removal": int(bool(row.get("removal"))),
                        "karma_delta": row.get("karma_delta", 0),
                    })
    return records, runs


def condition_summary(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["condition"])].append(row)
    output = []
    for (model, condition), rows in sorted(grouped.items()):
        n = len(rows)
        t5 = sum(r["t5"] for r in rows)
        lo, hi = wilson(t5, n)
        t5_rows = [r for r in rows if r["t5"]]
        output.append({
            "model": model,
            "condition": condition,
            "n": n,
            "t5_count": t5,
            "t5_rate": t5 / n if n else None,
            "ci95": [lo, hi],
            "risk_ack_given_t5": (
                sum(r["risk_ack"] for r in t5_rows) / len(t5_rows) if t5_rows else None
            ),
            "removal_rate": sum(r["removal"] for r in rows) / n if n else None,
            "mean_karma_delta": sum(r["karma_delta"] for r in rows) / n if n else None,
        })
    return output


def paired_bootstrap(records, condition_a, condition_b, draws=10000, seed=35304):
    by_key = defaultdict(dict)
    for row in records:
        if row["condition"] in (condition_a, condition_b):
            key = (row["model"], row["replicate"], row["instance"])
            by_key[key][row["condition"]] = row["t5"]
    diffs = np.asarray([
        values[condition_a] - values[condition_b]
        for values in by_key.values()
        if condition_a in values and condition_b in values
    ], dtype=float)
    if not len(diffs):
        return {"n_pairs": 0, "difference": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        estimates[i] = rng.choice(diffs, size=len(diffs), replace=True).mean()
    return {
        "n_pairs": int(len(diffs)),
        "difference": float(diffs.mean()),
        "ci95": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
    }


def factorial_effect_bootstrap(records, factor, draws=10000, seed=35304):
    """Average paired risk difference for one factor across the other cells."""
    factor_names = ("high_target", "elimination", "aggressive")
    others = [name for name in factor_names if name != factor]
    by_key = defaultdict(dict)
    for row in records:
        if row["condition"] not in MANIFEST["core_conditions"]:
            continue
        key = (
            row["model"], row["replicate"], row["instance"],
            *(row[name] for name in others),
        )
        by_key[key][row[factor]] = row["t5"]
    diffs = np.asarray([
        values[1] - values[0]
        for values in by_key.values()
        if 0 in values and 1 in values
    ], dtype=float)
    if not len(diffs):
        return {"n_pairs": 0, "risk_difference": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        estimates[i] = rng.choice(diffs, size=len(diffs), replace=True).mean()
    return {
        "n_pairs": int(len(diffs)),
        "risk_difference": float(diffs.mean()),
        "ci95": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
    }


def gee_factorial(records):
    core = [r for r in records if r["condition"] in MANIFEST["core_conditions"]]
    if not core:
        return {"status": "no factorial records"}
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        frame = pd.DataFrame(core)
        # Preregistered specification includes C(replicate); with a single
        # replicate that term is a constant column (rank-deficient), so it is
        # dropped. Same for C(model) if only one model is present. Any such
        # deviation is recorded in the returned payload for reporting.
        terms, dropped = [], []
        for term, column in (("C(model)", "model"),
                             ("C(instance)", "instance"),
                             ("C(replicate)", "replicate")):
            if frame[column].nunique() > 1:
                terms.append(term)
            else:
                dropped.append(term)
        formula = " + ".join(terms + ["high_target * elimination * aggressive"])
        fit = smf.gee(
            f"t5 ~ {formula}",
            groups="episode_cluster",
            data=frame,
            family=sm.families.Binomial(),
        ).fit()
        coefficients = {
            name: {
                "estimate": float(fit.params[name]),
                "se": float(fit.bse[name]),
                "p_value": float(fit.pvalues[name]),
                "ci95": [float(x) for x in fit.conf_int().loc[name]],
            }
            for name in fit.params.index
        }
        # Preregistered Holm correction across the three factorial main effects.
        main_names = [x for x in ("high_target", "elimination", "aggressive") if x in coefficients]
        ranked = sorted(main_names, key=lambda name: coefficients[name]["p_value"])
        previous = 0.0
        for rank, name in enumerate(ranked):
            adjusted = min(1.0, (len(ranked) - rank) * coefficients[name]["p_value"])
            adjusted = max(previous, adjusted)
            coefficients[name]["holm_p_value"] = adjusted
            previous = adjusted
        return {
            "status": "ok",
            "n": int(fit.nobs),
            "formula": f"t5 ~ {formula}",
            "dropped_terms_no_variation": dropped,
            "n_replicates": int(frame["replicate"].nunique()),
            "coefficients_log_odds": coefficients,
        }
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def print_summary(summary, contrasts, factorial_effects, runs):
    print("Pressure-factorial results")
    print("=" * 88)
    for row in summary:
        lo, hi = row["ci95"]
        print(
            f"{row['model']:<16} {row['condition']:<17} n={row['n']:>3} "
            f"T5={100*row['t5_rate']:>5.1f}% "
            f"[{100*lo:>5.1f}, {100*hi:>5.1f}]"
        )
    print("\nPaired contrasts (percentage points)")
    for name, result in contrasts.items():
        if result["difference"] is None:
            print(f"{name:<42} unavailable")
        else:
            print(
                f"{name:<42} {100*result['difference']:>+6.1f} "
                f"[{100*result['ci95'][0]:>+6.1f}, {100*result['ci95'][1]:>+6.1f}] "
                f"n={result['n_pairs']}"
            )
    print("\nAverage factorial main effects (percentage points)")
    for name, result in factorial_effects.items():
        if result["risk_difference"] is None:
            print(f"{name:<18} unavailable")
        else:
            print(
                f"{name:<18} {100*result['risk_difference']:>+6.1f} "
                f"[{100*result['ci95'][0]:>+6.1f}, {100*result['ci95'][1]:>+6.1f}] "
                f"n={result['n_pairs']}"
            )
    complete = sum(1 for r in runs if r["observed"] == r["planned"] or r["terminal_reason"])
    print(f"\nCompleted/validly terminated cells: {complete}/{len(runs)} discovered")


def main():
    records, runs = load_records()
    summary = condition_summary(records)
    contrasts = {
        name: paired_bootstrap(
            records, a, b,
            draws=MANIFEST["analysis"]["bootstrap_draws"],
            seed=MANIFEST["analysis"]["bootstrap_seed"],
        )
        for name, a, b in CONTRASTS
    }
    factorial_effects = {
        factor: factorial_effect_bootstrap(
            records, factor,
            draws=MANIFEST["analysis"]["bootstrap_draws"],
            seed=MANIFEST["analysis"]["bootstrap_seed"],
        )
        for factor in ("high_target", "elimination", "aggressive")
    }
    output = {
        "experiment_id": MANIFEST["experiment_id"],
        "record_count": len(records),
        "run_count": len(runs),
        "condition_summary": summary,
        "paired_contrasts": contrasts,
        "factorial_main_effects": factorial_effects,
        "factorial_gee": gee_factorial(records),
        "run_audit": runs,
    }
    (HERE / "analysis_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print_summary(summary, contrasts, factorial_effects, runs)
    print(f"\nSaved: {HERE / 'analysis_results.json'}")


if __name__ == "__main__":
    main()
