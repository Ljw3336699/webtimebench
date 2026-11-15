#!/usr/bin/env python3
"""Article-level event aggregation on the full WCEP corpus (same protocol as the
WCEP pilot and WebNewsEvents): each article's gold is its own parsed publication
date; evidence = retrieved same-event articles' dates aggregated by median.
Only articles with a parseable publication date are evaluated (denominator
reported separately).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timeagent.search.local_backend import LocalCorpusBackend  # noqa: E402


def parse_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def acc(actual, pred, tol):
    return 0.0 if pred is None else float(abs(actual - pred) <= tol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    rows = []
    by_id = {}
    for line in Path(args.corpus).open(encoding="utf-8"):
        r = json.loads(line)
        d = parse_date(r.get("publication_date"))
        rows.append({"id": r["corpus_id"], "event": r["event_id"], "title": r.get("title") or "",
                     "day": d.toordinal() if d else None})
        by_id[r["corpus_id"]] = r
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event"]].append(r)

    backend = LocalCorpusBackend(args.index)
    stats = {0: [], 7: [], 30: []}
    total = recall_hit = recall_den = 0
    for r in rows:
        if r["day"] is None:
            continue  # gold unavailable; other articles' dates still usable as evidence
        try:
            results = backend.search(str(r["title"]), args.topk)
        except ValueError:
            results = []
        others = [x for x in results if x.get("corpus_id") != r["id"]
                  and parse_date(x.get("publication_date")) is not None][:args.topk]
        total += 1
        own = set(x["id"] for x in by_event[r["event"]])
        recall_den += 1
        if any(x.get("corpus_id") in own for x in results):
            recall_hit += 1
        pred = statistics.median([parse_date(x.get("publication_date")).toordinal() for x in others]) if others else None
        for tol in (0, 7, 30):
            stats[tol].append(acc(r["day"], pred, tol))
    result = {"articles_evaluated": total, "topk": args.topk,
              "backend": backend.backend_id,
              "same_event_recall_topk": round(recall_hit / max(recall_den, 1), 4),
              "aggregate": {f"acc_at_{k}": round(statistics.mean(v), 4) for k, v in stats.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()