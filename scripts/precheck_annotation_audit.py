#!/usr/bin/env python3
"""Pre-screen the released annotation-audit sample with rule-based checks.

Flags records whose body evidence does not match the assigned regime, so the
author spot-check can focus on suspicious rows. It never writes a human verdict:
`auditor_*` columns stay empty.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

ISO_FULL = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
ISO_MONTH = re.compile(r"\b((?:19|20)\d{2})-(\d{2})\b")
MONTH_NAMES = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # Spanish
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
    # Portuguese
    "janeiro": 1, "fevereiro": 2, "marco": 3, "maio": 5, "junho": 6, "julho": 7,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
    # French
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "octobre": 10, "novembre": 11, "decembre": 12,
    # German
    "januar": 1, "februar": 2, "marz": 3, "juni": 6, "juli": 7, "oktober": 10,
    "dezember": 12,
}
MONTHS = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
MDY = re.compile(rf"\b({MONTHS})\.?\s+(\d{{1,2}}),?\s+((?:19|20)\d{{2}})\b", re.I)
DMY = re.compile(rf"\b(\d{{1,2}})\s*(?:de\s+|del\s+)?({MONTHS})\.?,?\s*(?:de\s+|del\s+)?((?:19|20)\d{{2}})\b", re.I)
MY = re.compile(rf"\b({MONTHS})\.?\s+(?:de\s+|del\s+)?((?:19|20)\d{{2}})\b", re.I)
CN_DATE = re.compile(r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?")
KO_DATE = re.compile(r"((?:19|20)\d{2})\s*년\s*(\d{1,2})\s*월(?:\s*(\d{1,2})\s*일)?")
NUM_DATE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)")
YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def body_of(row):
    return row["prompt"][0]["content"].split("Content:", 1)[1]


def dates_in(text):
    """Return (year, month|None, day|None) tuples for dates found in text."""
    found = []
    for y, m, d in ISO_FULL.findall(text):
        found.append((int(y), int(m), int(d)))
    for y, m in ISO_MONTH.findall(text):
        found.append((int(y), int(m), None))
    for mon, d, y in MDY.findall(text):
        found.append((int(y), month_number(mon), int(d)))
    for d, mon, y in DMY.findall(text):
        found.append((int(y), month_number(mon), int(d)))
    for mon, y in MY.findall(text):
        found.append((int(y), month_number(mon), None))
    for y, m, d in CN_DATE.findall(text):
        found.append((int(y), int(m), int(d) if d else None))
    for y, m, d in KO_DATE.findall(text):
        found.append((int(y), int(m), int(d) if d else None))
    for a, b, y in NUM_DATE.findall(text):
        # Numeric dates are ambiguous: record both day/month and month/day readings.
        found.append((int(y), int(b), int(a)))
        found.append((int(y), int(a), int(b)))
    return found


def month_number(name):
    key = unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode()
    return MONTH_NAMES[key]


def gold_parts(gold):
    y, m, d = (int(x) for x in gold.split("-"))
    return y, m, d


def screen(row):
    body = body_of(row)
    regime = row["difficulty"]
    gold = row["golden_answers"][0]
    gy, gm, gd = gold_parts(gold)
    found = dates_in(body)
    years = sorted({y for y, _, _ in found} | {int(y) for y in YEAR.findall(body)})
    exact = any((y, m, d) == (gy, gm, gd) for y, m, d in found if d is not None)
    same_month = any((y, m) == (gy, gm) for y, m, _ in found)
    gold_year = gy in years
    flags = []
    severity = "ok"
    if not found:
        flags.append("body_has_no_parseable_date")
    if regime == "EXP":
        if same_month:
            flags.append("EXP_gold_month_in_body_ok")
        elif gold_year:
            flags.append("EXP_gold_year_in_body_partial")
        else:
            flags.append("EXP_no_gold_year_signal")
            severity = "high"
    elif regime == "IMP":
        if found:
            flags.append("IMP_but_body_has_full_dates")
            severity = "medium"
        if same_month:
            flags.append("IMP_gold_month_in_body")
            severity = "high"
    elif regime == "DIS":
        if not found and not years:
            flags.append("DIS_no_temporal_signal")
            severity = "high"
        elif same_month:
            flags.append("DIS_gold_month_present_may_be_EXP")
            severity = "medium" if severity == "ok" else severity
        elif len(years) == 1 and gold_year:
            flags.append("DIS_only_gold_year_present")
            severity = "medium" if severity == "ok" else severity
    if re.search(r"\b(updated|reposted|republished|archived|as of)\b", body, re.I):
        flags.append("update_or_repost_cue")
    if re.search(r"\b(today|yesterday|tomorrow|last night|this week)\b", body, re.I):
        flags.append("relative_time_cue")
    return {
        "id": row["id"], "regime": regime, "gold_date": gold,
        "gold_in_body_exact": exact, "gold_month_in_body": same_month,
        "gold_year_in_body": gold_year,
        "body_years": years, "n_dates": len(found),
        "severity": severity, "needs_review": severity != "ok",
        "rule_pass": severity == "ok",
        "auto_flags": flags, "auto_notes": "; ".join(flags),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset variant directory, e.g. webtimebench/clean")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-report", type=Path, required=True,
                        help="annotation_audit_report.json produced by build_annotation_audit.py")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--xlsx", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.sample_report.read_text(encoding="utf-8"))
    selected = set(report["sample_ids"])
    rows = {r["id"]: r for r in json.loads((args.dataset / f"{args.split}.json").read_text(encoding="utf-8"))
            if r["id"] in selected}
    missing = selected - set(rows)
    if missing:
        raise SystemExit(f"sample ids missing from dataset: {sorted(missing)[:5]}")

    results = [screen(rows[i]) for i in report["sample_ids"]]
    summary = {"sample_size": len(results),
               "needs_review": sum(r["needs_review"] for r in results),
               "high": sum(r["severity"] == "high" for r in results),
               "medium": sum(r["severity"] == "medium" for r in results),
               "high_ids": [r["id"] for r in results if r["severity"] == "high"],
               "medium_ids": [r["id"] for r in results if r["severity"] == "medium"],
               "by_regime": {}}
    for regime in ("EXP", "IMP", "DIS"):
        subset = [r for r in results if r["regime"] == regime]
        summary["by_regime"][regime] = {
            "n": len(subset),
            "needs_review": sum(r["needs_review"] for r in subset),
            "gold_month_in_body": sum(r["gold_month_in_body"] for r in subset),
            "body_has_no_date": sum("body_has_no_parseable_date" in r["auto_flags"] for r in subset),
            "gold_year_in_body": sum(r["gold_year_in_body"] for r in subset),
            "high": sum(r["severity"] == "high" for r in subset),
            "medium": sum(r["severity"] == "medium" for r in subset),
        }
    payload = {"schema": "timeagent.annotation_audit.precheck.v1",
               "source_sample_seed": report["seed"], "summary": summary, "records": results}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    from openpyxl import load_workbook
    book = load_workbook(args.xlsx)
    sheet = book.active
    headers = [c.value for c in sheet[1]]
    extra = ["auto_gold_in_body", "auto_gold_month_in_body", "auto_gold_year_in_body",
             "auto_n_dates", "auto_flags", "auto_severity", "auto_rule_pass", "auto_needs_review"]
    for name in extra:
        if name not in headers:
            sheet.cell(row=1, column=len(headers) + 1, value=name)
            headers.append(name)
    index = {r["id"]: r for r in results}
    col = {name: headers.index(name) + 1 for name in extra}
    for row in sheet.iter_rows(min_row=2):
        rid = row[headers.index("id")].value
        r = index.get(rid)
        if not r:
            continue
        values = {
            "auto_gold_in_body": r["gold_in_body_exact"],
            "auto_gold_month_in_body": r["gold_month_in_body"],
            "auto_n_dates": r["n_dates"],
            "auto_flags": r["auto_notes"],
            "auto_severity": r["severity"],
            "auto_rule_pass": "Y" if r["rule_pass"] else "N",
            "auto_needs_review": r["needs_review"],
        }
        for name, value in values.items():
            sheet.cell(row=row[0].row, column=col[name], value=value)
    book.save(args.xlsx)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
