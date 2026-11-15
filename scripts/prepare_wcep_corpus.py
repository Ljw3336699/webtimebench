#!/usr/bin/env python3
"""Convert the WCEP pilot articles into a frozen local-corpus JSONL for indexing.

Input:  articles.jsonl (id, event_id, url, title, text, article_time_raw, ...)
Output: cubic JSONL (corpus_id, url, title, content, publication_date, source_date_evidence)
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import re


def parse_date(value):
    if not value:
        return None
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value))
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with args.articles.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as out:
        for line in source:
            row = json.loads(line)
            content = row.get("text") or ""
            if not content.strip():
                skipped += 1
                continue
            pub = parse_date(row.get("article_time_raw") or row.get("publication_date_candidate"))
            record = {
                "corpus_id": row["id"],
                "url": row.get("url"),
                "title": row.get("title"),
                "content": content,
                "publication_date": pub,
                "source_date_evidence": {"kind": "wcep", "field": "article_time_raw",
                                         "value": row.get("article_time_raw")},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"written": written, "skipped": skipped, "output": str(args.output)}))


if __name__ == "__main__":
    main()