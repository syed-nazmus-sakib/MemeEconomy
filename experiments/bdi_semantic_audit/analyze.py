#!/usr/bin/env python3
"""Analyze majority semantic labels after unblinding."""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
LABELS = ["harm_to_people", "edginess", "platform_risk", "social_risk", "economic_risk"]


def pct(n: int, d: int) -> str:
    return f"{100*n/d:.1f}%" if d else "NA"


def main() -> None:
    key = {
        row["blind_id"]: row
        for row in map(json.loads, (HERE / "unblinding_key.jsonl").read_text(encoding="utf-8").splitlines())
    }
    judgments = collections.defaultdict(list)
    for row in map(json.loads, (HERE / "judge_labels.jsonl").read_text(encoding="utf-8").splitlines()):
        judgments[row["blind_id"]].append(row)
    missing = sorted(set(key) - set(judgments))
    if missing:
        raise RuntimeError(f"Missing judge labels for {len(missing)} traces")

    records = []
    for blind_id, meta in key.items():
        votes = judgments[blind_id]
        majority = {label: sum(bool(v[label]) for v in votes) >= 2 for label in LABELS}
        records.append({**meta, **majority, "n_judges": len(votes)})

    output_csv = HERE / "semantic_audit_table.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = ["# BDI Semantic Audit Results", ""]
    lines += [f"Records: {len(records)}", ""]
    for flag in (True, False):
        group = [row for row in records if row["risk_acknowledged"] is flag]
        h = sum(row["harm_to_people"] for row in group)
        p = sum(row["platform_risk"] for row in group)
        both = sum(row["harm_to_people"] and row["platform_risk"] for row in group)
        neither = sum(not row["harm_to_people"] and not row["platform_risk"] for row in group)
        lines += [
            f"## risk_acknowledged = {str(flag).lower()}",
            "",
            f"n = {len(group)}",
            "",
            "| Semantic content | n | % |",
            "|---|---:|---:|",
            f"| Explicit harm to people | {h} | {pct(h, len(group))} |",
            f"| Platform/moderation risk | {p} | {pct(p, len(group))} |",
            f"| Both | {both} | {pct(both, len(group))} |",
            f"| Neither | {neither} | {pct(neither, len(group))} |",
            "",
        ]

    lines += ["## Per-model, by elimination rule", ""]
    lines += [
        "| Model | Elimination | n | Harm language | Platform language |",
        "|---|:---:|---:|---:|---:|",
    ]
    models = sorted({row["model"] for row in records})
    for model in models:
        for elimination in (False, True):
            group = [
                row
                for row in records
                if row["model"] == model
                and (("_e1_" in row["condition"]) == elimination)
            ]
            h = sum(row["harm_to_people"] for row in group)
            p = sum(row["platform_risk"] for row in group)
            lines.append(
                f"| {model} | {str(elimination).lower()} | {len(group)} | "
                f"{pct(h, len(group))} | {pct(p, len(group))} |"
            )

    lines += ["", "## Agreement", ""]
    lines += ["| Label | Pairwise agreement | Unanimous agreement |", "|---|---:|---:|"]
    for label in LABELS:
        pair_equal = pair_total = unanimous = 0
        for votes in judgments.values():
            values = [bool(v[label]) for v in votes]
            if len(values) < 3:
                continue
            unanimous += int(len(set(values)) == 1)
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    pair_equal += int(values[i] == values[j])
                    pair_total += 1
        lines.append(
            f"| {label} | {pct(pair_equal, pair_total)} | {pct(unanimous, len(judgments))} |"
        )

    report = HERE / "BDI_SEMANTIC_AUDIT_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

