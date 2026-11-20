#!/usr/bin/env python3
"""Corpus-level near-duplicate audit for event-grouped collections.

Event-level retrieval accuracy is inflated when articles sharing an event also
share a title or a body, because retrieval then recovers a copy of the query
rather than independent evidence. This diagnostic measures that directly on the
collection, with no retrieval system and no index, so it can be run on any
event-grouped corpus before any retrieval number is reported.

Reported per corpus:
  dup_title_pair_rate  fraction of same-event article pairs whose normalized
                       titles are identical
  dup_body_pair_rate   same, for normalized bodies
  events_any_dup       fraction of events containing at least one such pair
  articles_in_dup      fraction of articles having a same-event title twin
  same_day_pair_rate   fraction of same-event pairs published on the same day,
                       the quantity event-level date aggregation exploits

Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalize_text(value):
    """NFKC, case-folded, alphanumeric-only; the normalization title clustering uses."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^0-9a-z\u3400-\u9fff]+", " ", text).split())


def digest(value):
    return hashlib.sha256(normalize_text(value).encode()).hexdigest()[:16]


def open_maybe_gzip(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_flat(path, event_key, title_key, body_key, date_key):
    """One article per line, with an event identifier on the article."""
    for line in open_maybe_gzip(path):
        if not line.strip():
            continue
        row = json.loads(line)
        yield (row.get(event_key), row.get(title_key), row.get(body_key),
               str(row.get(date_key) or "")[:10])


def read_nested(path, articles_key, title_key, body_key, date_key):
    """One event per line with an embedded article list, as upstream WCEP ships."""
    for line in open_maybe_gzip(path):
        if not line.strip():
            continue
        event = json.loads(line)
        event_id = event.get("id")
        for article in event.get(articles_key) or []:
            yield (event_id, article.get(title_key), article.get(body_key),
                   str(article.get(date_key) or event.get(date_key) or "")[:10])


def audit(records, max_pairs_per_event=2000):
    by_event = defaultdict(list)
    for event_id, title, body, day in records:
        if event_id is None:
            continue
        by_event[event_id].append((digest(title), digest(body), day))

    pairs = dup_title = dup_body = same_day = 0
    events_any = events_sized = 0
    articles_total = articles_dup = 0
    for members in by_event.values():
        articles_total += len(members)
        titles = defaultdict(int)
        for title_hash, _, _ in members:
            titles[title_hash] += 1
        articles_dup += sum(count for count in titles.values() if count > 1)
        if len(members) < 2:
            continue
        events_sized += 1
        local_dup = False
        # Cap the pair enumeration so one huge event cannot dominate the cost.
        limit = members[:max_pairs_per_event]
        for i in range(len(limit)):
            for j in range(i + 1, len(limit)):
                pairs += 1
                if limit[i][0] == limit[j][0]:
                    dup_title += 1
                    local_dup = True
                if limit[i][1] == limit[j][1]:
                    dup_body += 1
                if limit[i][2] and limit[i][2] == limit[j][2]:
                    same_day += 1
        events_any += int(local_dup)

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {
        "events": len(by_event),
        "events_with_two_or_more": events_sized,
        "articles": articles_total,
        "same_event_pairs_examined": pairs,
        "dup_title_pair_rate": rate(dup_title, pairs),
        "dup_body_pair_rate": rate(dup_body, pairs),
        "same_day_pair_rate": rate(same_day, pairs),
        "events_any_dup": rate(events_any, events_sized),
        "articles_in_dup": rate(articles_dup, articles_total),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layout", choices=("flat", "nested"), default="flat",
                        help="flat: one article per line; nested: one event per line")
    parser.add_argument("--event-key", default="event_id")
    parser.add_argument("--articles-key", default="articles")
    parser.add_argument("--title-key", default="title")
    parser.add_argument("--body-key", default="text")
    parser.add_argument("--date-key", default="date")
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.layout == "flat":
        records = read_flat(args.input, args.event_key, args.title_key,
                            args.body_key, args.date_key)
    else:
        records = read_nested(args.input, args.articles_key, args.title_key,
                              args.body_key, args.date_key)
    result = audit(records)
    result["corpus"] = args.label or args.input.name
    result["input"] = args.input.as_posix()
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
