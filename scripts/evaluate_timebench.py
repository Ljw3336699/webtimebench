#!/usr/bin/env python3
"""Evaluate a model on Time-Bench test_comprehension (third-party baseline).

Simple protocol: run the prompt as given; extract the first YYYY-MM date from
the response; compare against the first non-null date in reward_model.
ground_truth (month accuracy). Reports per-task and overall accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def extract_month(text):
    match = re.search(r"(\d{4})-(\d{2})", text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def gold_month(row):
    gt = (row.get("reward_model") or {}).get("ground_truth") or {}
    targets = [gt.get("event_pub_date"), gt.get("event1_pub_date"),
               gt.get("event2_pub_date"), gt.get("event3_pub_date"),
               gt.get("target"), gt.get("masked_date"), gt.get("time")]
    for value in targets:
        if isinstance(value, str) and re.match(r"^\d{4}-\d{2}", value):
            return (int(value[:4]), int(value[5:7]))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    table = pq.read_table(args.parquet)
    rows = table.to_pylist()
    if args.limit:
        rows = rows[:args.limit]
    print(f"rows: {len(rows)}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    parsed = 0
    results = []
    per_task = defaultdict(lambda: [0, 0])
    for i, row in enumerate(rows):
        gold = gold_month(row)
        messages = []
        for part in row.get("prompt") or []:
            role = part.get("role")
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": part.get("content")})
            else:
                messages.append({"role": "user", "content": part.get("content")})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_month(reply)
        ok = gold is not None and pred == gold
        task = (row.get("extra_info") or {}).get("task", "unknown")
        per_task[task][0] += 1
        per_task[task][1] += int(ok)
        if gold is not None:
            parsed += 1
        results.append({"index": i, "task": task, "gold": f"{gold[0]:04d}-{gold[1]:02d}" if gold else None,
                        "prediction": f"{pred[0]:04d}-{pred[1]:02d}" if pred else None, "correct": ok})
        if (i + 1) % 100 == 0:
            print(f"  .. {i+1}/{len(rows)}", flush=True)
    summary = {"model": args.model, "rows": len(rows), "with_gold": parsed,
               "month_accuracy": round(sum(1 for r in results if r["correct"]) / parsed, 4) if parsed else None,
               "per_task": {k: {"n": v[0], "acc": round(v[1] / v[0], 4)} for k, v in per_task.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "cases": results},
                                      ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()