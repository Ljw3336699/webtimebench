#!/usr/bin/env python3
"""Build the released regime-label spot-check sample (deterministic, seed 20261116).

The sample is an author-review sheet: it ships the records to check, not verdicts.
Auditors fill the `auditor_*` columns; an empty verdict means "not yet reviewed".
"""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

REGIMES = ("EXP", "IMP", "DIS")


def body_of(row):
    return row["prompt"][0]["content"].split("Content:", 1)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset variant directory, e.g. webtimebench/clean")
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-regime", type=int, default=34)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--xlsx-output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads((args.dataset / f"{args.split}.json").read_text(encoding="utf-8"))
    by_regime = {r: [] for r in REGIMES}
    for row in rows:
        regime = row.get("difficulty")
        if regime in by_regime:
            by_regime[regime].append(row)

    rng = random.Random(args.seed)
    sample = []
    for regime in REGIMES:
        pool = sorted(by_regime[regime], key=lambda r: r["id"])
        if len(pool) < args.per_regime:
            raise SystemExit(f"{regime}: only {len(pool)} records available")
        picked = rng.sample(pool, args.per_regime)
        for row in sorted(picked, key=lambda r: r["id"]):
            body = body_of(row)
            sample.append({
                "id": row["id"],
                "split": args.split,
                "regime": regime,
                "gold_date": row["golden_answers"][0],
                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "body_excerpt": body[:300],
                "auditor_regime": "",
                "auditor_gold_date": "",
                "auditor_notes": "",
            })

    report = {
        "schema": "timeagent.annotation_audit.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(args.dataset),
        "split": args.split,
        "seed": args.seed,
        "per_regime": args.per_regime,
        "sample_size": len(sample),
        "regime_counts": {r: sum(1 for s in sample if s["regime"] == r) for r in REGIMES},
        "audit_status": "pending",
        "reviewed": 0,
        "disagreements": [],
        "notes": "auditor_* fields are blank until the author spot-check is completed",
        "sample_ids": [s["id"] for s in sample],
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    from openpyxl import Workbook
    book = Workbook()
    sheet = book.active
    sheet.title = "annotation_audit"
    columns = ["id", "split", "regime", "gold_date", "body_sha256", "body_excerpt",
               "auditor_regime", "auditor_gold_date", "auditor_notes"]
    sheet.append(columns)
    for item in sample:
        sheet.append([item[c] for c in columns])
    book.save(args.xlsx_output)
    print(json.dumps({k: report[k] for k in
                      ("split", "seed", "sample_size", "regime_counts", "audit_status")}, indent=2))


if __name__ == "__main__":
    main()
