#!/usr/bin/env python3
"""Near-duplicate audit within and across published dataset splits.

Companion to audit_event_duplication.py. Where that script measures duplication
inside event groups, this one measures (a) how often an item's normalized key
recurs within a split and (b) how often a held-out item's key already occurs in
the training split. Keys use the same NFKC / case-fold / alphanumeric
normalization, with masking tokens such as <YEAR> removed first so that the same
article masked differently still counts as the same article.

Readers: jsonl / jsonl.gz (standard library), and parquet (needs pyarrow).
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

MASK = re.compile(r"<\s*(year|month|day|date|time)\s*>", re.I)


def norm(value):
    text = MASK.sub(" ", str(value or ""))
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^0-9a-z\u3400-\u9fff]+", " ", text).split())


def rows_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def rows_parquet(path):
    import pyarrow.parquet as pq
    yield from pq.read_table(path).to_pylist()


def rows_json(path):
    yield from json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    name = str(path)
    if name.endswith(".parquet"):
        return rows_parquet(path)
    if name.endswith(".json"):
        return rows_json(path)
    return rows_jsonl(path)


def prompt_text(row):
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        try:
            prompt = ast.literal_eval(prompt)
        except (ValueError, SyntaxError):
            return prompt
    if isinstance(prompt, list):
        return " ".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return str(prompt or "")


def field_after(text, label):
    match = re.search(rf"{label}:\s*(.+)", text)
    return match.group(1).strip() if match else ""


def keys_for(row, schema):
    """Yield (headline_key, content_key) pairs for one row; several for nested rows."""
    if schema == "timebench":
        text = prompt_text(row)
        head, abstract = field_after(text, "Headline"), field_after(text, "Abstract")
        yield norm(head), norm(head + " " + abstract)
    elif schema == "wcep":
        for article in row.get("articles") or []:
            yield norm(article.get("title")), norm(article.get("text"))
    elif schema == "pair":
        yield norm(row.get("title")), norm(row.get("article"))
    elif schema == "webtimebench":
        body = prompt_text(row)
        yield "", norm(body)
    else:
        raise ValueError(f"unknown schema {schema}")


def collect(paths, schema):
    heads, contents = [], []
    for path in paths:
        for row in read_rows(path):
            for head, content in keys_for(row, schema):
                heads.append(head)
                contents.append(content)
    return heads, contents


def within(keys):
    keys = [k for k in keys if k]
    if not keys:
        return None
    counts = Counter(keys)
    return {"items": len(keys),
            "items_with_recurring_key": round(sum(c for c in counts.values() if c > 1) / len(keys), 4),
            "distinct_keys": len(counts)}


def across(train_keys, test_keys):
    train = {k for k in train_keys if k}
    test = [k for k in test_keys if k]
    if not test:
        return None
    hits = sum(k in train for k in test)
    return {"held_out_items": len(test), "found_in_train": hits,
            "rate": round(hits / len(test), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", choices=("timebench", "wcep", "pair", "webtimebench"), required=True)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="*", default=[])
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    train_heads, train_contents = collect(args.train, args.schema)
    result = {"resource": args.label, "schema": args.schema,
              "train_files": [p.as_posix() for p in args.train],
              "test_files": [p.as_posix() for p in args.test],
              "within_train": {"headline": within(train_heads), "content": within(train_contents)}}
    if args.test:
        test_heads, test_contents = collect(args.test, args.schema)
        result["within_test"] = {"headline": within(test_heads), "content": within(test_contents)}
        result["test_in_train"] = {"headline": across(train_heads, test_heads),
                                   "content": across(train_contents, test_contents)}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
