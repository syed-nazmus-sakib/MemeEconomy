#!/usr/bin/env python3
"""Prepare a blinded semantic-audit corpus from completed BDI traces."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_GLOB = str(ROOT / "results/pressure_factorial_2026/*/rep_01/*/run*.jsonl")
BLINDED = HERE / "blinded_traces.jsonl"
KEY = HERE / "unblinding_key.jsonl"
HUMAN_SHEET = HERE / "human_annotation_sheet.csv"
SALT = "memeeconomy-bdi-audit-2026-08-02"


def text_field(value: object) -> str:
    return str(value or "").strip()


def main() -> None:
    records = []
    for filename in sorted(glob.glob(SOURCE_GLOB)):
        source_path = Path(filename)
        model = source_path.parent.name
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    row.get("type") != "round_result"
                    or row.get("incomplete")
                    or row.get("meme_tier") != 5
                ):
                    continue
                bdi = row.get("bdi_result") or {}
                belief = bdi.get("belief") or {}
                desire = bdi.get("desire") or {}
                intention = bdi.get("intention") or {}
                source_id = "|".join(
                    [
                        text_field(row.get("condition")),
                        model,
                        text_field(row.get("instance")),
                        text_field(row.get("round")),
                        text_field(row.get("meme_chosen")),
                    ]
                )
                blind_id = hashlib.sha256(f"{SALT}|{source_id}".encode()).hexdigest()[:16]
                text = {
                    "meme_assessment": text_field(belief.get("meme_assessment")),
                    "audience_reaction": text_field(belief.get("audience_reaction")),
                    "risk_justification": text_field(desire.get("risk_justification")),
                    "decision_reasoning": text_field(intention.get("reasoning")),
                }
                records.append(
                    {
                        "blind_id": blind_id,
                        "text": text,
                        "meta": {
                            "model": model,
                            "condition": row.get("condition"),
                            "instance": row.get("instance"),
                            "round": row.get("round"),
                            "meme_id": row.get("meme_chosen"),
                            "risk_acknowledged": bool(row.get("risk_acknowledged")),
                            "removal": bool(row.get("removal")),
                            "source_file": str(Path(filename).relative_to(ROOT)),
                        },
                    }
                )

    records.sort(key=lambda item: item["blind_id"])
    if len(records) != 380:
        raise RuntimeError(f"Expected 380 Tier-5 records, found {len(records)}")
    if len({record["blind_id"] for record in records}) != len(records):
        raise RuntimeError("Blind identifier collision")

    with BLINDED.open("w", encoding="utf-8") as blinded, KEY.open(
        "w", encoding="utf-8"
    ) as key:
        for record in records:
            blinded.write(
                json.dumps(
                    {"blind_id": record["blind_id"], "text": record["text"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            key.write(
                json.dumps(
                    {"blind_id": record["blind_id"], **record["meta"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with HUMAN_SHEET.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "blind_id",
            "meme_assessment",
            "audience_reaction",
            "risk_justification",
            "decision_reasoning",
            "harm_to_people",
            "edginess",
            "platform_risk",
            "social_risk",
            "economic_risk",
            "none",
            "evidence",
            "annotator",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({"blind_id": record["blind_id"], **record["text"]})

    print(f"Prepared {len(records)} blinded traces")
    print(BLINDED)
    print(KEY)
    print(HUMAN_SHEET)


if __name__ == "__main__":
    main()
