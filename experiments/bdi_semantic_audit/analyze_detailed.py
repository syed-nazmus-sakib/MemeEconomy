#!/usr/bin/env python3
"""Secondary semantic-audit diagnostics requested in the preregistration."""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "BDI_SEMANTIC_AUDIT_DETAILED_RESULTS.md"
LABELS = ["harm_to_people", "edginess", "platform_risk", "social_risk", "economic_risk"]


def main():
    rows = list(csv.DictReader((HERE / "semantic_audit_table.csv").open(encoding="utf-8")))
    for row in rows:
        for label in ["risk_acknowledged", *LABELS]:
            row[label] = row[label].lower() == "true"

    lines = [
        "# Detailed BDI Semantic-Audit Results",
        "",
        "## What the legacy risk flag captures",
        "",
        "| Semantic label | Flag=true with label | Flag=true without label |",
        "|---|---:|---:|",
    ]
    for label in LABELS:
        present = [row for row in rows if row[label]]
        absent = [row for row in rows if not row[label]]
        a = 100 * sum(row["risk_acknowledged"] for row in present) / len(present)
        b = 100 * sum(row["risk_acknowledged"] for row in absent) / len(absent)
        lines.append(f"| {label} | {a:.1f}% ({sum(r['risk_acknowledged'] for r in present)}/{len(present)}) | {b:.1f}% ({sum(r['risk_acknowledged'] for r in absent)}/{len(absent)}) |")

    normal = {
        (row["model"], row["instance"]): row
        for row in rows if row["condition"] == "pf_t25_e0_a0"
    }
    survival = {
        (row["model"], row["instance"]): row
        for row in rows if row["condition"] == "pf_t50_e1_a0"
    }
    common = sorted(set(normal) & set(survival))
    lines += [
        "",
        "## Same-model, same-event selected-Tier-5 comparison",
        "",
        "This comparison is conditional on the same model selecting Tier 5 for the same event in both Normal (25k, no elimination, balanced language) and Structural Survival (50k, elimination, balanced language). It therefore describes trace language among common harmful selections, not all decision opportunities.",
        "",
        f"Matched selections: {len(common)}.",
        "",
        "| Label | Normal | Structural survival | Change |",
        "|---|---:|---:|---:|",
    ]
    for label in ["risk_acknowledged", *LABELS]:
        before = sum(normal[key][label] for key in common)
        after = sum(survival[key][label] for key in common)
        change = 100 * (after - before) / len(common)
        lines.append(
            f"| {label} | {before}/{len(common)} ({100*before/len(common):.1f}%) | "
            f"{after}/{len(common)} ({100*after/len(common):.1f}%) | {change:+.1f} pp |"
        )

    judgments = [
        json.loads(line)
        for line in (HERE / "judge_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    usage = [
        json.loads(line)
        for line in (HERE / "judge_usage.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    counts = collections.Counter(row["blind_id"] for row in judgments)
    total_cost = sum((row.get("usage", {}).get("cost") or 0) for row in usage)
    lines += [
        "",
        "## Integrity",
        "",
        f"- Blinded traces: {len(counts)}.",
        f"- Model judgments: {len(judgments)}.",
        f"- Traces receiving exactly three judgments: {sum(value == 3 for value in counts.values())}/{len(counts)}.",
        f"- Recorded API cost: ${total_cost:.4f}.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
