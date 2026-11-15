#!/usr/bin/env python3
"""Prepare the WCEP full dataset (val+test splits) for event-level retrieval
aggregation. Gold = the WCEP editor-supplied event date (100% complete);
evidence = event articles (title/text) with parsed publish_data where available
(null kept so failure is measured, never skipped).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from datetime import date
from pathlib import Path

MONTHS = {m: i + 1 for i, m in enumerate(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}


def parse_date(value):
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    for pattern in (r"^(\d{4})-(\d{2})-(\d{2})", r"^(\d{4})/(\d{1,2})/(\d{1,2})"):
        m = re.match(pattern, s)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    m = re.match(r"^([A-Z][a-z]{2})\w*\.?\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and m.group(1)[:3] in MONTHS:
        try:
            return date(int(m.group(3)), MONTHS[m.group(1)[:3]], int(m.group(2)))
        except ValueError:
            return None
    m = re.match(r"^(\d{1,2})\s+([A-Z][a-z]{2})\w*\s+(\d{4})", s)
    if m and m.group(2)[:3] in MONTHS:
        try:
            return date(int(m.group(3)), MONTHS[m.group(2)[:3]], int(m.group(1)))
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, nargs="+", required=True,
                        help="WCEP jsonl.gz splits (val/test)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-articles", type=int, default=20, help="articles kept per event")
    args = parser.parse_args()
    out = args.output_dir
    if (out / "corpus.jsonl").exists():
        raise SystemExit(f"Output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    events, corpus = [], []
    for split in args.splits:
        with gzip.open(split, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                event_id = "wcep:" + str(row["id"])
                event_date = parse_date(row.get("date"))
                if event_date is None:
                    continue
                events.append({"event_id": event_id, "date": event_date.isoformat(),
                               "category": row.get("category")})
                articles = row.get("articles") or []
                for i, a in enumerate(articles[:args.max_articles]):
                    text = (a.get("text") or "").strip()
                    if len(text) < 200:
                        continue
                    pub = parse_date(a.get("time") or a.get("publish_data"))
                    article_id = str(a.get("id") or f"{event_id}_{i}")
                    corpus.append({"corpus_id": article_id, "event_id": event_id,
                                   "title": a.get("title") or "", "content": text,
                                   "url": a.get("url"),
                                   "publication_date": pub.isoformat() if pub else None,
                                   "source_date_evidence": {"kind": "wcep",
                                                            "field": "publish_data",
                                                            "value": a.get("publish_data")}})
    with (out / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    dated = sum(1 for c in corpus if c["publication_date"])
    print(json.dumps({"events": len(events), "articles": len(corpus),
                      "articles_with_date": dated,
                      "date_coverage": round(dated / max(len(corpus), 1), 4),
                      "output_dir": str(out)}))


if __name__ == "__main__":
    main()