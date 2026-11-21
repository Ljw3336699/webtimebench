#!/usr/bin/env python3
"""Model-agnostic evaluation client for the WebTimeBench search-then-answer loop.

Re-implements the greedy rollout used by ``launch.py evaluate`` (same prompt
template, same <search>/<answer> parsing, same information formatting, at most
``--max-turns`` executed searches followed by one forced final turn, same
metrics code) on top of a plain vLLM engine, so that models whose architecture
the pinned verl/vLLM/transformers stack cannot load (e.g. Qwen3) can be scored
under identical evidence conditions.

  generate: python evaluate_standalone_agent.py generate --model DIR --out DIR \
      --search-url http://127.0.0.1:8001 [--null] [--shard I --num-shards N]
  merge:    python evaluate_standalone_agent.py merge --out DIR
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


temporal = _load("wtb_temporal", ROOT / "src/verl/utils/reward_score/temporal.py")
client_mod = _load("wtb_client", ROOT / "src/timeagent/search/client.py")

ACTION = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
INVALID = "\nInvalid action. Emit a nonempty <search>query</search> or <answer>YYYY-MM-DD</answer>.\n"
FAILED = "Search request failed; no external evidence was returned."


def truncate_response(text):
    """Same stop rule as the verl rollout: cut after the first </search>, else </answer>."""
    if "</search>" in text:
        return text.split("</search>")[0] + "</search>"
    if "</answer>" in text:
        return text.split("</answer>")[0] + "</answer>"
    return text


def format_docs(group):
    out = ""
    for idx, item in enumerate(group):
        content = item["document"]["contents"]
        lines = content.split("\n")
        out += f"Doc {idx + 1}(Publish time: {lines[0]}) " + "\n".join(lines[1:]) + "\n"
    return out


def search(client, queries, null, topk):
    """Returns (observation strings, failed flags). One HTTP batch; never retried."""
    if null:
        return [""] * len(queries), [False] * len(queries)
    try:
        rows = client.retrieve(queries, topk=topk)["result"]
        return [format_docs(g) for g in rows], [False] * len(queries)
    except client_mod.RetrievalClientError:
        return [FAILED] * len(queries), [True] * len(queries)


def build_prompts(tokenizer, frame, enable_thinking):
    prompts = []
    for chat in frame["prompt"]:
        chat = [dict(m) for m in chat]
        kwargs = {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
        prompts.append(tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, **kwargs))
    return prompts


def run_loop(llm, tokenizer, params, prompts, client, args):
    n = len(prompts)
    texts = list(prompts)
    model_text = [""] * n
    executed, failed, gen_tokens = [0] * n, [0] * n, [0] * n
    active = list(range(n))
    for turn in range(args.max_turns + 1):
        if not active:
            break
        outs = llm.generate([texts[i] for i in active], params, use_tqdm=False)
        final = turn == args.max_turns
        queries, owners, still = [], [], []
        for i, out in zip(active, outs):
            gen_tokens[i] += len(out.outputs[0].token_ids)
            resp = truncate_response(out.outputs[0].text)
            texts[i] += resp
            model_text[i] += resp
            m = ACTION.search(resp)
            action, content = (m.group(1), m.group(2).strip()) if m else (None, "")
            if action == "answer" or final:
                continue
            if action == "search" and content:
                queries.append(content)
                owners.append(i)
                executed[i] += 1
            else:
                texts[i] += INVALID
                model_text[i] += ""
                still.append(i)
        if queries:
            obs, flags = search(client, queries, args.null, args.topk)
            for i, o, flag in zip(owners, obs, flags):
                failed[i] += int(flag)
                ids = tokenizer(o.strip(), add_special_tokens=False)["input_ids"][: args.max_obs_length]
                texts[i] += "\n\n<information>" + tokenizer.decode(ids) + "</information>\n\n"
                still.append(i)
        active = still
    return model_text, executed, failed, gen_tokens


def generate(args):
    import pandas as pd
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    frame = pd.read_parquet(args.data)
    if args.limit:
        frame = frame.iloc[: args.limit]
    frame = frame.iloc[args.shard :: args.num_shards].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    thinking = None if args.thinking == "default" else args.thinking == "on"
    prompts = build_prompts(tokenizer, frame, thinking)
    llm = LLM(args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=0, enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_response_length, stop=["</search>"],
                            include_stop_str_in_output=True)
    client = None if args.null else client_mod.RetrievalClient(args.search_url, timeout=60, topk=args.topk)
    start = time.time()
    model_text, executed, failed, gen_tokens = run_loop(llm, tokenizer, params, prompts, client, args)
    seconds = (time.time() - start) / max(len(prompts), 1)
    rows = []
    for i in range(len(frame)):
        gold = temporal.reference_date(frame.iloc[i]["reward_model"]["ground_truth"])
        rows.append({"id": frame.iloc[i]["id"], "prediction": temporal.extract_answer(model_text[i]), "gold": gold,
                     "regime": temporal.normalize_regime(frame.iloc[i]["difficulty"]),
                     "prompt_tokens": len(tokenizer(prompts[i], add_special_tokens=False)["input_ids"]),
                     "generated_tokens": gen_tokens[i], "amortized_batch_seconds": seconds,
                     "executed_searches": executed[i], "failed_searches": failed[i]})
    out = Path(args.out) / "predictions" / "step_0"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"shard_{args.shard:02d}_of_{args.num_shards:02d}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    (out / f"shard_{args.shard:02d}_of_{args.num_shards:02d}.sample.txt").write_text(
        "\n=====\n".join(model_text[:5]), encoding="utf-8")


def merge(args):
    import pandas as pd

    directory = Path(args.out) / "predictions" / "step_0"
    rows = [json.loads(line) for f in sorted(directory.glob("shard_*.jsonl")) for line in f.read_text().splitlines()]
    expected = pd.read_parquet(args.data)["id"].tolist()
    if args.limit:
        expected = expected[: args.limit]
    temporal.validate_prediction_ids(expected, [r["id"] for r in rows])
    order = {i: k for k, i in enumerate(expected)}
    rows.sort(key=lambda r: order[r["id"]])
    metrics, _, _, _ = temporal.compute_metrics(
        [r["prediction"] for r in rows], [r["gold"] for r in rows], [r["executed_searches"] for r in rows],
        [r["prompt_tokens"] for r in rows], [r["generated_tokens"] for r in rows],
        [r["amortized_batch_seconds"] for r in rows], [r["regime"] for r in rows])
    path = directory / "evaluation_standalone.jsonl"
    temporal.write_prediction_manifest(path, expected, rows)
    report = {"metrics": metrics, "record_count": len(rows), "complete_manifest": True,
              "evaluator": "scripts/evaluate_standalone_agent.py (greedy, vLLM, same loop as verl rollout)",
              "evaluation_reward": "pure exp(-0.05 * absolute_day_error / 30)"}
    path.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("count", "prediction_coverage", "score_micro", "acc_at_30", "search_rate")
                      if k in metrics}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "merge"])
    parser.add_argument("--model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=str(ROOT / "data/temporal/test.parquet"))
    parser.add_argument("--search-url", default="http://127.0.0.1:8001")
    parser.add_argument("--null", action="store_true", help="return empty evidence (no-retrieval control)")
    parser.add_argument("--thinking", choices=["default", "on", "off"], default="off",
                        help="chat-template thinking switch (Qwen3); 'default' omits the argument")
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--max-obs-length", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "generate":
        if not args.model:
            parser.error("--model is required for generate")
        generate(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
