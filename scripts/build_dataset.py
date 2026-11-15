#!/usr/bin/env python3
"""Build reproducible, exact-body-disjoint date datasets without changing inputs.

Python standard library handles cleaning and JSON; pyarrow is required for Parquet.
This is a local repaired dataset version for retraining, not a provenance audit.
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

VERSION = "timeagent_revision_v2.1"
REGIMES = ("EXP", "IMP", "DIS")
ALIASES = {"easy": "EXP", "medium": "IMP", "hard": "DIS", **{x: x for x in REGIMES}}
NORMALIZATION = "first prompt content after first Content:; Unicode NFKC; lowercase; collapse whitespace; strip"


def valid_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return False
    try:
        dt.date.fromisoformat(value)
        return True
    except ValueError:
        return False


def normalized_body(row):
    try:
        text = row["prompt"][0]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(text, str) or "Content:" not in text:
        return None
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text.split("Content:", 1)[1]).lower()).strip()


def body_hash(row):
    body = normalized_body(row)
    if not body:
        raise ValueError("Missing or empty body")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assign(rows, split):
    result = []
    for index, row in enumerate(rows):
        row = copy.deepcopy(row)
        row["extra_info"].update(index=index, split=split)
        result.append(row)
    return result


def build_splits(train, test, seed=20261103, dev_fraction=0.05):
    """Return cleaned data and all decisions; input row objects are never mutated."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must lie strictly between 0 and 1")
    seen = set()
    candidates = collections.defaultdict(list)
    quarantine, excluded, mapping = [], [], []
    by_id = {}
    for source_split, rows in (("train", train), ("test", test)):
        for source_position, raw in enumerate(rows):
            identity = raw.get("id")
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError(f"Missing or duplicate source ID: {identity!r}")
            seen.add(identity)
            entry = {"id": identity, "original_split": source_split,
                     "source_position": source_position, "outcome": None,
                     "clean_split": None, "temporal_split": None, "reasons": []}
            mapping.append(entry)
            by_id[identity] = entry
            reasons = []
            golds = raw.get("golden_answers")
            if not isinstance(golds, list) or len(golds) != 1 or not valid_date(golds[0]):
                reasons.append("invalid_gold")
            body = normalized_body(raw)
            if body is None:
                reasons.append("missing_body_separator_or_prompt")
            elif not body:
                reasons.append("empty_body")
            difficulty = raw.get("difficulty")
            regime = ALIASES.get(difficulty) if isinstance(difficulty, str) else None
            if regime is None:
                reasons.append("invalid_regime")
            try:
                reward_target = raw["reward_model"]["ground_truth"]["target"]
                if reward_target != golds:
                    reasons.append("gold_reward_mismatch")
            except (KeyError, TypeError):
                reasons.append("missing_reward_target")
            if reasons:
                entry.update(outcome="quarantine", reasons=reasons)
                quarantine.append({"id": identity, "original_split": source_split,
                                   "reasons": reasons, "original_row": raw})
                continue
            row = copy.deepcopy(raw)
            row["difficulty"] = regime
            extra = row.setdefault("extra_info", {})
            if not isinstance(extra, dict):
                raise ValueError(f"extra_info must be a mapping: {identity}")
            extra.update(original_split=source_split,
                         original_index=extra.get("index", source_position),
                         source_position=source_position, body_sha256=body_hash(raw))
            # The match was checked above; normalization cannot alter the reference date.
            row["reward_model"]["ground_truth"]["target"] = list(golds)
            candidates[extra["body_sha256"]].append((source_split, source_position, row, raw))
            entry["body_sha256"] = extra["body_sha256"]

    retained = {"train": [], "test": []}
    conflict_groups = collections.Counter()
    for digest, group in sorted(candidates.items()):
        reasons = []
        if len({x[2]["golden_answers"][0] for x in group}) > 1:
            reasons.append("conflicting_gold_for_body")
        if len({x[2]["difficulty"] for x in group}) > 1:
            reasons.append("conflicting_regime_for_body")
        if reasons:
            conflict_groups.update(reasons)
            for source_split, _, row, raw in group:
                by_id[row["id"]].update(outcome="quarantine", reasons=reasons)
                quarantine.append({"id": row["id"], "original_split": source_split,
                                   "body_sha256": digest, "reasons": reasons, "original_row": raw})
            continue
        # Keeping test first prevents retained original test content entering training.
        ordered = sorted(group, key=lambda x: (0 if x[0] == "test" else 1, x[1]))
        kept_split, _, kept, _ = ordered[0]
        retained[kept_split].append(kept)
        for source_split, _, row, raw in ordered[1:]:
            reason = "train_body_in_original_test" if source_split == "train" and kept_split == "test" else "duplicate_body_same_labels"
            by_id[row["id"]].update(outcome="excluded", reasons=[reason], retained_id=kept["id"])
            excluded.append({"id": row["id"], "original_split": source_split,
                             "body_sha256": digest, "reasons": [reason],
                             "retained_id": kept["id"], "original_row": raw})

    dev_ids = set()
    for regime in REGIMES:
        pool = sorted((r for r in retained["train"] if r["difficulty"] == regime), key=lambda r: r["id"])
        random.Random(f"{seed}:{regime}").shuffle(pool)
        n = math.floor(len(pool) * dev_fraction)
        if len(pool) >= 2:
            n = min(len(pool) - 1, max(1, n))
        dev_ids.update(r["id"] for r in pool[:n])
    order = lambda rows: sorted(rows, key=lambda r: r["extra_info"]["source_position"])
    clean = {
        "train": _assign(order([r for r in retained["train"] if r["id"] not in dev_ids]), "train"),
        "dev": _assign(order([r for r in retained["train"] if r["id"] in dev_ids]), "dev"),
        "test": _assign(order(retained["test"]), "test"),
    }
    temporal = {
        split: _assign([r for r in rows if (r["golden_answers"][0][:4] <= "2023" if split != "test"
                                          else r["golden_answers"][0][:4] >= "2024")], split)
        for split, rows in clean.items()
    }
    for split, rows in clean.items():
        for row in rows:
            by_id[row["id"]].update(outcome="retained", clean_split=split)
    for split, rows in temporal.items():
        for row in rows:
            by_id[row["id"]]["temporal_split"] = split
    result = {"clean": clean, "temporal": temporal,
              "quarantine": sorted(quarantine, key=lambda x: (x["original_split"], x["id"])),
              "excluded": sorted(excluded, key=lambda x: (x["original_split"], x["id"])),
              "id_mapping": mapping, "conflict_group_counts": dict(conflict_groups)}
    validate_splits(clean, temporal, mapping)
    return result


def validate_splits(clean, temporal=None, mapping=None):
    seen_ids, seen_bodies = set(), set()
    for split in ("train", "dev", "test"):
        for index, row in enumerate(clean[split]):
            identity = row["id"]
            digest = body_hash(row)
            if identity in seen_ids or digest in seen_bodies:
                raise ValueError("Duplicate ID or body within/across clean splits")
            seen_ids.add(identity)
            seen_bodies.add(digest)
            if len(row["golden_answers"]) != 1 or not valid_date(row["golden_answers"][0]):
                raise ValueError(f"Invalid reference date: {identity}")
            if row["reward_model"]["ground_truth"]["target"] != row["golden_answers"]:
                raise ValueError(f"Reward target mismatch: {identity}")
            if row["difficulty"] not in REGIMES:
                raise ValueError(f"Invalid regime: {identity}")
            extra = row["extra_info"]
            if extra["index"] != index or extra["split"] != split or extra["body_sha256"] != digest:
                raise ValueError(f"Invalid metadata: {identity}")
            if (split == "test") != (extra["original_split"] == "test"):
                raise ValueError(f"Original test crossed split boundary: {identity}")
    if temporal is not None:
        for split, rows in temporal.items():
            parent = {r["id"]: r for r in clean[split]}
            expected = {identity for identity, row in parent.items()
                        if (row["golden_answers"][0][:4] <= "2023" if split != "test"
                            else row["golden_answers"][0][:4] >= "2024")}
            if len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != expected:
                raise ValueError(f"Wrong temporal subset: {split}")
            for index, row in enumerate(rows):
                expected_row = copy.deepcopy(parent[row["id"]])
                expected_row["extra_info"].update(index=index, split=split)
                if row != expected_row:
                    raise ValueError(f"Temporal row changed: {row['id']}")
    if mapping is not None:
        if len({r["id"] for r in mapping}) != len(mapping):
            raise ValueError("Duplicate mapping ID")
        kept = {r["id"] for r in mapping if r["outcome"] == "retained"}
        if kept != seen_ids or any(r["outcome"] not in {"retained", "quarantine", "excluded"} for r in mapping):
            raise ValueError("Incomplete disposition mapping")
        clean_lookup = {row["id"]: (split, row) for split, rows in clean.items() for row in rows}
        temporal_lookup = {row["id"]: split for split, rows in (temporal or {}).items() for row in rows}
        for entry in mapping:
            identity = entry["id"]
            if identity in clean_lookup:
                split, row = clean_lookup[identity]
                if entry["clean_split"] != split or entry["body_sha256"] != body_hash(row):
                    raise ValueError(f"Incorrect retained-ID mapping: {identity}")
                if entry["original_split"] != row["extra_info"]["original_split"]:
                    raise ValueError(f"Incorrect source-ID mapping: {identity}")
            elif entry["clean_split"] is not None or not entry["reasons"]:
                raise ValueError(f"Incorrect removed-ID mapping: {identity}")
            if temporal is not None and entry["temporal_split"] != temporal_lookup.get(identity):
                raise ValueError(f"Incorrect temporal-ID mapping: {identity}")
    return {split: len(rows) for split, rows in clean.items()}


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in data:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def describe(rows):
    years = collections.Counter(r["golden_answers"][0][:4] for r in rows)
    return {"records": len(rows), "regimes": dict(collections.Counter(r["difficulty"] for r in rows)),
            "years": dict(sorted(years.items())),
            "date_range": [min((r["golden_answers"][0] for r in rows), default=None),
                           max((r["golden_answers"][0] for r in rows), default=None)]}


def write_dataset(result, output, input_paths, seed, dev_fraction, skip_parquet=False):
    output = Path(output).resolve()
    input_paths = [Path(p).resolve() for p in input_paths]
    if any(output == p.parent or output in p.parents for p in input_paths):
        raise ValueError("Output directory must not contain original inputs")
    if skip_parquet and any((output / variant / f"{split}.parquet").exists()
                            for variant in ("clean", "temporal") for split in ("train", "dev", "test")):
        raise ValueError("--skip-parquet would leave stale Parquet files; choose a new output directory")
    parquet = None
    if not skip_parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Parquet output requires pyarrow; install it or explicitly use --skip-parquet for JSON-only work") from exc
    paths = []
    summaries = {}
    for variant in ("clean", "temporal"):
        summaries[variant] = {}
        for split, rows in result[variant].items():
            file = output / variant / f"{split}.json"
            _write_json(file, rows)
            paths.append(file)
            if parquet is not None:
                if not rows:
                    raise ValueError("Empty split needs an explicit Parquet schema; no empty split is accepted for training")
                file = file.with_suffix(".parquet")
                parquet.write_table(pa.Table.from_pylist(rows), file, compression="zstd")
                paths.append(file)
            summaries[variant][split] = describe(rows)
    for key in ("quarantine", "excluded"):
        file = output / key / "records.jsonl"
        _write_jsonl(file, result[key])
        paths.append(file)
    file = output / "id_mapping.json"
    _write_json(file, result["id_mapping"])
    paths.append(file)
    manifest = {
        "version": VERSION, "seed": seed, "dev_fraction": dev_fraction,
        "normalization": NORMALIZATION,
        "input_files": {str(p): {"sha256": sha256(p), "records": len(json.loads(p.read_text(encoding="utf-8")))} for p in input_paths},
        "output_files": {str(p.relative_to(output)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths},
        "parquet_written": not skip_parquet,
        "split_summary": summaries,
        "quarantined_records": len(result["quarantine"]),
        "excluded_records": len(result["excluded"]),
        "quarantine_reason_counts": dict(collections.Counter(reason for row in result["quarantine"] for reason in row["reasons"])),
        "exclusion_reason_counts": dict(collections.Counter(reason for row in result["excluded"] for reason in row["reasons"])),
        "conflict_group_counts": result["conflict_group_counts"],
        "guarantees": {"exact_normalized_body_disjoint": True, "original_test_never_in_train_or_dev": True,
                       "source_disjoint_verified": False, "event_disjoint_verified": False,
                       "near_duplicate_disjoint_verified": False, "reference_provenance_verified": False,
                       "requires_retraining": True},
        "temporal_protocol": "train/dev: corresponding clean split with gold <= 2023-12-31; test: clean test with gold >= 2024-01-01; no original test enters training",
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def validate_dataset(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for filename, details in manifest["output_files"].items():
        if sha256(output / filename) != details["sha256"]:
            raise ValueError(f"Output hash mismatch: {filename}")
    variants = {variant: {split: json.loads((output / variant / f"{split}.json").read_text(encoding="utf-8"))
                          for split in ("train", "dev", "test")}
                for variant in ("clean", "temporal")}
    mapping = json.loads((output / "id_mapping.json").read_text(encoding="utf-8"))
    validate_splits(variants["clean"], variants["temporal"], mapping)
    if sum(d["records"] for d in manifest["input_files"].values()) != len(mapping):
        raise ValueError("Input/output accounting mismatch")
    for kind in ("quarantine", "excluded"):
        records = [json.loads(line) for line in (output / kind / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        expected = {r["id"] for r in mapping if r["outcome"] == kind}
        if len(records) != len(expected) or {r["id"] for r in records} != expected:
            raise ValueError(f"Wrong {kind} record set")
        decisions = {r["id"]: r for r in mapping if r["outcome"] == kind}
        for record in records:
            decision = decisions[record["id"]]
            if (record["original_row"]["id"] != record["id"] or
                    record["reasons"] != decision["reasons"] or
                    record["original_split"] != decision["original_split"]):
                raise ValueError(f"Incorrect exclusion provenance: {record['id']}")
    if manifest["parquet_written"]:
        import pyarrow.parquet as pq
        for variant, splits in variants.items():
            for split, rows in splits.items():
                restored = pq.read_table(output / variant / f"{split}.parquet").to_pylist()
                if restored != rows:
                    raise ValueError(f"Parquet/JSON mismatch: {variant}/{split}")
    return {"status": "passed", "clean_counts": {k: len(v) for k, v in variants["clean"].items()},
            "temporal_counts": {k: len(v) for k, v in variants["temporal"].items()},
            "accounted_source_ids": len(mapping), "parquet_json_equivalent": manifest["parquet_written"]}


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=base.parent / "data/train_data7.json")
    parser.add_argument("--test", type=Path, default=base.parent / "data/test_data7.json")
    parser.add_argument("--output", type=Path, default=base / "data")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--skip-parquet", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_dataset(args.output), indent=2))
        return
    inputs = [args.train, args.test]
    before = {str(p): sha256(p) for p in inputs}
    result = build_splits(*(json.loads(p.read_text(encoding="utf-8")) for p in inputs),
                          seed=args.seed, dev_fraction=args.dev_fraction)
    manifest = write_dataset(result, args.output, inputs, args.seed, args.dev_fraction, args.skip_parquet)
    validation = validate_dataset(args.output)
    if before != {str(p): sha256(p) for p in inputs}:
        raise RuntimeError("An original input changed during construction")
    report = {"validation": validation, "source_inputs_unchanged": True, **manifest}
    reports = base / "reports"
    _write_json(reports / "dataset_build.json", report)
    counts = manifest["split_summary"]
    lines = ["# 数据构建报告", "", f"版本：{VERSION}。固定随机种子：{args.seed}。原始输入文件未修改。", "",
             "| 划分 | clean | temporal |", "|---|---:|---:|"]
    lines.extend(f"| {s} | {counts['clean'][s]['records']} | {counts['temporal'][s]['records']} |" for s in ("train", "dev", "test"))
    lines += ["", f"隔离记录：{len(result['quarantine'])}；去重/跨集合排除记录：{len(result['excluded'])}。",
              f"全部 {len(result['id_mapping']):,} 条原始记录的去向均在 data/id_mapping.json 中；理由可重叠，理由计数不能直接求和当作记录数。", "",
              "## 隔离原因", ""]
    lines.extend(f"- {k}: {v}" for k, v in manifest["quarantine_reason_counts"].items())
    lines += ["", "## 排除原因", ""]
    lines.extend(f"- {k}: {v}" for k, v in manifest["exclusion_reason_counts"].items())
    lines += ["", "## 验证边界", "", "已验证严格公历日期、单一金标、gold/reward 一致、ID 完整追踪、精确归一正文去重和跨集合无重叠、原测试集不进入训练/开发集、时间子集约束。",
              "不能从缺少 URL、日期来源证据和事件簇的本地导出确认 source-disjoint、event-disjoint、语义近重复消除或 gold 真实性。所有冲突记录仅隔离，未猜测或修补日期。",
              "新版划分必须重新训练；在旧 checkpoint 上运行新版测试集不能宣称已经消除历史训练污染。Parquet 和 JSON 等价检查状态见 dataset_build.json。", ""]
    (reports / "dataset_build.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"validation": validation, "quarantined": len(result["quarantine"]),
                      "excluded": len(result["excluded"]), "reasons": manifest["quarantine_reason_counts"]}, indent=2))


if __name__ == "__main__":
    main()
