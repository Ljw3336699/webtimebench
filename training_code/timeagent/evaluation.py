#!/usr/bin/env python3
"""Strict, independent temporal-date evaluator (Python standard library only).

Training rewards are never imported. See ../EVALUATION_PROTOCOL.md for denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

REGIME_ALIASES = {'easy': 'EXP', 'medium': 'IMP', 'hard': 'DIS',
                  'exp': 'EXP', 'imp': 'IMP', 'dis': 'DIS'}


class EvaluationError(ValueError):
    pass


def parse_date(value):
    """Accept an actual calendar date in exactly YYYY-MM-DD, without whitespace."""
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
        return None
    try:
        return date.fromisoformat(value).toordinal()
    except ValueError:
        return None


def load_records(path, kind):
    path = Path(path)
    try:
        if path.suffix.lower() == '.jsonl':
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            records = json.loads(path.read_text())
            if isinstance(records, dict):
                field = 'items' if kind == 'gold' else 'predictions'
                records = records.get(field)
        if not isinstance(records, list):
            raise EvaluationError(f'{kind}: expected a JSON array or a {"items" if kind == "gold" else "predictions"} array')
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f'{kind}: cannot read {path}: {exc}') from exc
    return records


def index_records(records, kind, allow_empty=False):
    indexed = {}
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise EvaluationError(f'{kind} row {i}: expected object')
        item_id = record.get('id')
        if not isinstance(item_id, str) or not item_id.strip():
            raise EvaluationError(f'{kind} row {i}: id must be a nonempty string')
        if item_id in indexed:
            raise EvaluationError(f'{kind}: duplicate id {item_id!r}')
        indexed[item_id] = record
    if not indexed and not allow_empty:
        raise EvaluationError(f'{kind}: empty input')
    return indexed


def extract_gold(record):
    # No oracle minimum across several acceptable answers: this task uses one anchor.
    for field in ('gold', 'gold_date', 'target', 'golden_answers', 'golds'):
        if field in record:
            value = record[field]
            if isinstance(value, list):
                if len(value) != 1:
                    return None, f'{field} must contain exactly one gold date'
                value = value[0]
            day = parse_date(value)
            return day, None if day is not None else f'{field} is not a valid YYYY-MM-DD calendar date'
    return None, 'no gold field (gold/gold_date/target/golden_answers/golds)'


def prepare_gold(records, exclude_invalid_gold=False):
    all_gold = index_records(records, 'gold')
    eligible, excluded = {}, []
    for item_id, row in all_gold.items():
        day, error = extract_gold(row)
        if error:
            excluded.append({'id': item_id, 'reason': error})
            continue
        regime = row.get('regime', row.get('difficulty', 'ALL'))
        if not isinstance(regime, str) or not regime.strip():
            raise EvaluationError(f'gold {item_id}: regime/difficulty must be a nonempty string')
        regime = REGIME_ALIASES.get(regime.casefold(), regime)
        eligible[item_id] = {'day': day, 'regime': regime}
    if excluded and not exclude_invalid_gold:
        ids = ', '.join(x['id'] for x in excluded[:8])
        raise EvaluationError(f'{len(excluded)} invalid gold date(s): {ids}; correct the data or explicitly use --exclude-invalid-gold')
    if not eligible:
        raise EvaluationError('no eligible gold dates remain')
    return all_gold, eligible, excluded


def prepare_predictions(records, label, all_gold, eligible, allow_partial=False):
    indexed = index_records(records, label, allow_empty=allow_partial)
    extras = sorted(set(indexed) - set(all_gold))
    if extras:
        raise EvaluationError(f'{label}: {len(extras)} unknown id(s), e.g. {extras[:5]}')
    missing = sorted(set(eligible) - set(indexed))
    if missing and not allow_partial:
        raise EvaluationError(f'{label}: missing {len(missing)} eligible id(s), e.g. {missing[:5]}; use --allow-partial only to retain these as invalid predictions in the full denominator')
    for item_id, row in indexed.items():
        if 'executed_searches' in row:
            count = row['executed_searches']
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise EvaluationError(f'{label} {item_id}: executed_searches must be a nonnegative integer')
    return indexed, missing


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] * (high-position) + values[high] * (position-low) if low != high else values[low]


def score_rows(eligible, predictions, alpha):
    rows = []
    for item_id in sorted(eligible):
        gold = eligible[item_id]
        record = predictions.get(item_id)
        pred_day = parse_date(record.get('prediction')) if record is not None else None
        error = abs(pred_day - gold['day']) if pred_day is not None else None
        rows.append({'id': item_id, 'regime': gold['regime'], 'error_days': error,
                     'utility': math.exp(-alpha*error/30.0) if error is not None else 0.0,
                     'acc1': int(error is not None and error <= 1),
                     'acc7': int(error is not None and error <= 7),
                     'acc30': int(error is not None and error <= 30),
                     'exact': int(error == 0), 'missing_record': record is None,
                     'executed_searches': record.get('executed_searches') if record is not None else None})
    return rows


def summarize(rows):
    n = len(rows)
    errors = [r['error_days'] for r in rows if r['error_days'] is not None]
    searches = [r['executed_searches'] for r in rows if r['executed_searches'] is not None]
    n_valid = len(errors)
    search_counts = Counter(searches)
    return {'n_eligible_gold': n, 'n_valid_predictions': n_valid,
            'n_missing_records': sum(r['missing_record'] for r in rows),
            'prediction_coverage': n_valid/n,
            'invalid_prediction_rate_including_missing': (n-n_valid)/n,
            'score_micro': sum(r['utility'] for r in rows)/n,
            'exact_match': sum(r['exact'] for r in rows)/n,
            'acc_at_1': sum(r['acc1'] for r in rows)/n,
            'acc_at_7': sum(r['acc7'] for r in rows)/n,
            'acc_at_30': sum(r['acc30'] for r in rows)/n,
            'mae_valid_predictions_days': statistics.mean(errors) if errors else None,
            'error_quantiles_valid_predictions_days': {
                'p50': percentile(errors, .50), 'p75': percentile(errors, .75),
                'p90': percentile(errors, .90), 'p95': percentile(errors, .95)},
            'search': {
                'n_with_executed_count': len(searches), 'count_coverage': len(searches)/n,
                'search_rate_among_counted': sum(s > 0 for s in searches)/len(searches) if searches else None,
                'mean_turns_among_counted': statistics.mean(searches) if searches else None,
                'turn_probability_among_counted': {
                    '0': search_counts[0]/len(searches) if searches else None,
                    '1': search_counts[1]/len(searches) if searches else None,
                    '2': search_counts[2]/len(searches) if searches else None,
                    'gt2': sum(s > 2 for s in searches)/len(searches) if searches else None}}}


def metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['regime']].append(row)
    regimes = {name: summarize(part) for name, part in sorted(grouped.items())}
    overall = summarize(rows)
    overall['score_macro'] = statistics.mean(x['score_micro'] for x in regimes.values())
    overall['n_regimes'] = len(regimes)
    return {'overall': overall, 'by_regime': regimes}


def paired_bootstrap(rows, baseline_rows, iterations=2000, seed=20261103):
    if [r['id'] for r in rows] != [r['id'] for r in baseline_rows]:
        raise EvaluationError('paired comparison requires exactly the same eligible ID sequence')
    if iterations < 100:
        raise EvaluationError('--bootstrap-samples must be at least 100 (default 2000)')
    n = len(rows)
    grouped = defaultdict(list)
    joint_errors = []
    for a, b in zip(rows, baseline_rows):
        grouped[a['regime']].append((a['utility']-b['utility'], a['acc30']-b['acc30']))
        if a['error_days'] is not None and b['error_days'] is not None:
            joint_errors.append(a['error_days']-b['error_days'])
    point_macro = statistics.mean(statistics.mean(p[0] for p in ps) for ps in grouped.values())
    point_micro = sum(p[0] for ps in grouped.values() for p in ps)/n
    point_acc = sum(p[1] for ps in grouped.values() for p in ps)/n
    rng = random.Random(seed)
    samples = {'score_macro_delta': [], 'score_micro_delta': [], 'acc_at_30_delta': [],
               'mae_delta_joint_valid_days': []}
    for _ in range(iterations):
        macro_parts, total_s, total_a = [], 0., 0.
        # Preserve each observed regime's size; paired draw shares IDs across systems.
        for name in sorted(grouped):
            pairs = grouped[name]
            sum_s, sum_a = 0., 0.
            for _ in range(len(pairs)):
                s, a = pairs[rng.randrange(len(pairs))]
                sum_s += s
                sum_a += a
            macro_parts.append(sum_s/len(pairs))
            total_s += sum_s
            total_a += sum_a
        samples['score_macro_delta'].append(statistics.mean(macro_parts))
        samples['score_micro_delta'].append(total_s/n)
        samples['acc_at_30_delta'].append(total_a/n)
        if joint_errors:
            samples['mae_delta_joint_valid_days'].append(
                sum(joint_errors[rng.randrange(len(joint_errors))] for _ in range(len(joint_errors)))/len(joint_errors))
    points = {'score_macro_delta': point_macro, 'score_micro_delta': point_micro,
              'acc_at_30_delta': point_acc,
              'mae_delta_joint_valid_days': statistics.mean(joint_errors) if joint_errors else None}
    out = {'direction': 'prediction minus baseline; Score/accuracy positive favors prediction; MAE negative favors prediction',
           'bootstrap_samples': iterations, 'seed': seed, 'confidence_level': 0.95,
           'score_accuracy_population': 'all eligible gold IDs; invalid/missing predictions have utility=0 and count as misses',
           'score_accuracy_resampling': 'paired, stratified by observed regime with fixed per-regime counts',
           'mae_population': 'only IDs with valid predictions from BOTH systems; pooled paired resampling of this fixed joint-valid subset',
           'n_all_eligible_pairs': n, 'n_joint_valid_pairs': len(joint_errors),
           'joint_valid_pair_coverage': len(joint_errors)/n,
           'limitation': 'CIs quantify test-example sampling under fixed regimes, not training-seed, event-cluster, or retrieval-snapshot uncertainty.',
           'metrics': {}}
    for key, vals in samples.items():
        out['metrics'][key] = {'estimate': points[key],
                              'ci95_percentile': [percentile(vals, .025), percentile(vals, .975)] if vals else None}
    return out


def evaluate(gold_records, prediction_records, *, baseline_records=None, alpha=.05,
             allow_partial=False, exclude_invalid_gold=False, bootstrap_samples=2000, seed=20261103):
    if not math.isfinite(alpha) or alpha <= 0:
        raise EvaluationError('--alpha must be finite and positive')
    all_gold, eligible, excluded = prepare_gold(gold_records, exclude_invalid_gold)
    pred, missing = prepare_predictions(prediction_records, 'prediction', all_gold, eligible, allow_partial)
    rows = score_rows(eligible, pred, alpha)
    result = {'protocol_version': '1.0', 'protocol': {
        'score': 'exp(-alpha*absolute_day_error/30); invalid/missing prediction=0; no regime weights, search bonuses, or training-format penalties',
        'alpha': alpha, 'macro': 'equal mean of per-regime mean utility across observed eligible regimes',
        'micro': 'mean utility across all eligible gold IDs',
        'accuracy_denominator': 'all eligible gold IDs, including invalid/missing predictions as misses',
        'error_denominator': 'valid predictions only; always interpret with prediction_coverage',
        'quantiles': 'linear interpolation at (n-1)*q',
        'partial_mode': allow_partial,
        'partial_behavior': 'missing eligible IDs remain in all score/accuracy denominators as invalid predictions',
        'exclude_invalid_gold': exclude_invalid_gold,
        'n_input_gold': len(all_gold), 'n_eligible_gold': len(eligible),
        'n_excluded_gold': len(excluded), 'excluded_gold': excluded,
        'missing_prediction_ids': missing,
        'partial_input': bool(missing),
        'prediction_ids_excluded_with_invalid_gold': sorted(set(pred)-set(eligible))},
        'prediction': metrics(rows)}
    if baseline_records is not None:
        baseline, baseline_missing = prepare_predictions(baseline_records, 'baseline', all_gold, eligible, allow_partial)
        if set(pred) != set(baseline):
            raise EvaluationError('prediction and baseline files must contain exactly the same ID set, including in partial mode')
        baseline_rows = score_rows(eligible, baseline, alpha)
        result['baseline'] = metrics(baseline_rows)
        result['paired_comparison'] = paired_bootstrap(rows, baseline_rows, bootstrap_samples, seed)
    return result


def fingerprint(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True, help='JSON array (or {items: [...]}) of gold records')
    parser.add_argument('--predictions', required=True, help='JSON/JSONL records: id, prediction, optional executed_searches')
    parser.add_argument('--baseline', help='Optional paired baseline JSON/JSONL with exactly the same IDs')
    parser.add_argument('--output', help='Save JSON report; default stdout')
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--exclude-invalid-gold', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261103)
    args = parser.parse_args(argv)
    try:
        result = evaluate(load_records(args.gold, 'gold'), load_records(args.predictions, 'prediction'),
            baseline_records=load_records(args.baseline, 'prediction') if args.baseline else None,
            alpha=args.alpha, allow_partial=args.allow_partial,
            exclude_invalid_gold=args.exclude_invalid_gold,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        result['inputs'] = {'gold': fingerprint(args.gold), 'predictions': fingerprint(args.predictions)}
        if args.baseline:
            result['inputs']['baseline'] = fingerprint(args.baseline)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text)
            print(f'Wrote {args.output}', file=sys.stderr)
        else:
            print(text, end='')
        return 0
    except EvaluationError as exc:
        print(f'Evaluation failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
