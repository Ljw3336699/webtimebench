#!/usr/bin/env python3
"""Restrict a processed WCEP corpus to articles with a high-confidence event assignment.

In the upstream WCEP release most articles come from Common Crawl and carry an automatic event
assignment (`events[*].probability`); only the editor-cited reference articles have origin 'WCEP'.
This keeps rows of the processed corpus (`prepare_wcep_full.py` output) whose upstream article is an
editor-cited article or has an assignment probability >= a threshold, so event-level controls can be
re-run on the most confidently attributed members only.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
from pathlib import Path


def best_probability(article):
    events = article.get("events")
    if isinstance(events, str):
        try:
            events = ast.literal_eval(events)
        except (ValueError, SyntaxError):
            events = None
    probabilities = [e["probability"] for e in events or [] if isinstance(e, dict) and "probability" in e]
    return max(probabilities) if probabilities else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, nargs="+", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    keep, origin_counts = {}, {}
    for path in args.upstream:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                for article in json.loads(line)["articles"]:
                    origin = article.get("origin")
                    probability = best_probability(article)
                    origin_counts[origin] = origin_counts.get(origin, 0) + 1
                    keep[article["id"]] = origin == "WCEP" or (probability is not None and probability >= args.threshold)
    total = kept = unmatched = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.corpus.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as out:
        for line in src:
            row = json.loads(line)
            total += 1
            flag = keep.get(row["corpus_id"])
            if flag is None:
                unmatched += 1
            elif flag:
                kept += 1
                out.write(line)
    print(json.dumps({"threshold": args.threshold, "upstream_origin_counts": origin_counts,
                      "corpus_rows": total, "kept": kept, "unmatched_ids": unmatched}))


if __name__ == "__main__":
    main()
