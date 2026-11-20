#!/usr/bin/env python3
"""Build a reviewer-friendly workbook for the 102-record annotation audit.

One sheet per regime. Most rows only need `confirm=Y`; rows flagged by the
rule-based precheck are highlighted so they can be inspected in detail.
"""
import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


def body_of(row):
    return row["prompt"][0]["content"].split("Content:", 1)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-report", type=Path, required=True)
    parser.add_argument("--precheck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.sample_report.read_text(encoding="utf-8"))
    precheck = {r["id"]: r for r in json.loads(args.precheck.read_text(encoding="utf-8"))["records"]}
    rows = {r["id"]: r for r in json.loads((args.dataset / f"{args.split}.json").read_text(encoding="utf-8"))
            if r["id"] in set(report["sample_ids"])}

    book = Workbook()
    readme = book.active
    readme.title = "README"
    readme["A1"] = "Annotation audit — 102 records (34 per regime, seed 20261116)"
    readme["A1"].font = Font(bold=True, size=13)
    instructions = [
        "Each regime has its own sheet.",
        "For a correct record: set confirm=Y and leave auditor_regime/auditor_gold_date empty.",
        "If the regime or gold date is wrong: set confirm=N, fill auditor_regime and/or auditor_gold_date, add a note.",
        "Rows with auto_needs_review=TRUE are precheck flags (see auto_flags); inspect those bodies first.",
        "Rules used: EXP -> gold year/month should appear in body; IMP -> no full date; DIS -> competing temporal cues present.",
        "Date parsing covers English, Spanish, Portuguese, French, German, Japanese/Chinese and Korean formats.",
        "Do not delete rows; the aggregation reads id, confirm, auditor_regime, auditor_gold_date, auditor_notes.",
    ]
    for i, text in enumerate(instructions, start=3):
        readme.cell(row=i, column=1, value=f"- {text}")
    readme.column_dimensions["A"].width = 120

    headers = ["id", "split", "regime", "gold_date", "auto_severity", "auto_rule_pass",
               "auto_needs_review", "auto_flags", "body_excerpt", "confirm",
               "auditor_regime", "auditor_gold_date", "auditor_notes"]
    widths = [13, 8, 8, 12, 10, 14, 14, 42, 100, 9, 15, 17, 32]
    highlight = PatternFill("solid", fgColor="FFF2CC")
    for regime in ("EXP", "IMP", "DIS"):
        sheet = book.create_sheet(regime)
        sheet.append(headers)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for col, width in enumerate(widths, start=1):
            sheet.column_dimensions[chr(64 + col) if col <= 26 else "A"].width = width
        ids = [i for i in report["sample_ids"] if rows[i]["difficulty"] == regime]
        for rid in ids:
            row = rows[rid]
            check = precheck[rid]
            sheet.append([
                rid, args.split, regime, row["golden_answers"][0],
                check["severity"], "Y" if check["rule_pass"] else "N",
                check["needs_review"], check["auto_notes"],
                body_of(row)[:1200], "", "", "", "",
            ])
        last = sheet.max_row
        for r in range(2, last + 1):
            if sheet.cell(row=r, column=7).value:
                for c in range(1, len(headers) + 1):
                    sheet.cell(row=r, column=c).fill = highlight
            sheet.cell(row=r, column=9).alignment = Alignment(wrap_text=True, vertical="top")
            sheet.cell(row=r, column=8).alignment = Alignment(wrap_text=True, vertical="top")
        dv_regime = DataValidation(type="list", formula1='"EXP,IMP,DIS"', allow_blank=True)
        dv_confirm = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
        sheet.add_data_validation(dv_regime)
        sheet.add_data_validation(dv_confirm)
        dv_regime.add(f"K2:K{last}")
        dv_confirm.add(f"J2:J{last}")
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = f"A1:M{last}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    book.save(args.output)
    print(json.dumps({"output": str(args.output), "records": sum(1 for _ in report["sample_ids"]),
                      "flagged": sum(1 for v in precheck.values() if v["needs_review"])}, indent=2))


if __name__ == "__main__":
    main()
