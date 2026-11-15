#!/usr/bin/env python3
"""Event-level retrieval aggregation on the full WCEP dataset.

For each event, query the frozen BM25 index with the event title (first article
title + category), retrieve top-k articles, aggregate their parsed
publication_date by median, and compare against the WCEP editor event date.
Full denominator (no retrieval → failure). Reports Acc@0/7/30 and same-event
recall.
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


def load_events(path):
    events = []
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        d = parse_date(row["date"])
        if d:
            events.append({"event_id": row["event_id"], "day": d.toordinal(),
                           "date": d.isoformat()})
    return events


def load_corpus_index(path):
    """Map event_id -> corpus ids and corpus_id -> date, plus first title query."""
    by_event = defaultdict(list)
    by_id = {}
    query = {}
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        by_event[row["event_id"]].append(row["corpus_id"])
        pd = parse_date(row.get("publication_date"))
        by_id[row["corpus_id"]] = pd.toordinal() if pd else None
        if row["event_id"] not in query:
            query[row["event_id"]] = row["title"]
    return by_event, by_id, query


def acc(actual, pred, tol):
    return 0.0 if pred is None else float(abs(actual - pred) <= tol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    events = load_events(args.events)
    by_event, by_id, query = load_corpus_index(args.corpus)
    backend = LocalCorpusBackend(args.index)
    stats = {0: [], 7: [], 30: []}
    recall_hit = recall_total = 0
    per_case = []
    for ev in events:
        q = query.get(ev["event_id"]) or " ".join(ev["event_id"].split(":"))
        try:
            results = backend.search(str(q), args.topk)
        except ValueError:
            results = []
        own = set(by_event[ev["event_id"]])
        others = [r for r in results if r.get("corpus_id") not in own
                  and by_id.get(r.get("corpus_id")) is not None][:args.topk]
        recall_total += 1
        if any(r.get("corpus_id") in own for r in results):
            recall_hit += 1
        pred = statistics.median([by_id[r["corpus_id"]] for r in others]) if others else None
        for tol in (0, 7, 30):
            stats[tol].append(acc(ev["day"], pred, tol))
        per_case.append({"event_id": ev["event_id"], "gold": ev["date"],
                         "predicted_ordinal": int(pred) if pred is not None else None,
                         "retrieved": [r.get("corpus_id") for r in others],
                         "same_event_in_retrieved": any(r.get("corpus_id") in own for r in results)})
    result = {"events": len(events), "topk": args.topk,
              "index_backend": backend.backend_id,
              "oracle_available_events": recall_total,
              "same_event_recall": round(recall_hit / max(recall_total, 1), 4),
              "aggregate": {f"acc_at_{k}": round(statistics.mean(v), 4) for k, v in stats.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"result": result, "cases": per_case},
                                      ensure_ascii=False, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()