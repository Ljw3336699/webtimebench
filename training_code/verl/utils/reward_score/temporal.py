"""Strict date scoring shared by training and validation (no GPU dependencies)."""
import datetime as dt
import math
import re
from collections import Counter
from pathlib import Path
import json

REGIMES = {'easy': 'EXP', 'medium': 'IMP', 'hard': 'DIS', 'EXP': 'EXP', 'IMP': 'IMP', 'DIS': 'DIS'}


def parse_date(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def reference_date(ground_truth):
    target = ground_truth.get('target') if isinstance(ground_truth, dict) else ground_truth
    if not isinstance(target, str):
        try:
            if target is None or len(target) != 1:
                raise ValueError('Exactly one reference date is required')
            target = target[0]
        except (TypeError, KeyError, IndexError) as exc:
            raise ValueError('Reference date must be a string or single-element sequence') from exc
    if parse_date(target) is None:
        raise ValueError(f'Invalid reference date: {target!r}')
    return target


def normalize_regime(value):
    return REGIMES.get(value, 'unknown')


def extract_answer(response):
    if isinstance(response, bytes):
        response = response.decode('utf-8', errors='replace')
    if not isinstance(response, str):
        return None
    # Retrieved observations cannot supply the model's final answer.
    response = re.sub(r'<information>.*?(?:</information>|$)', '', response, flags=re.DOTALL)
    answers = re.findall(r'<answer>(.*?)</answer>', response, flags=re.DOTALL)
    return answers[-1].strip() if answers else None


def utility(prediction, target, alpha=0.05):
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    gold = parse_date(target)
    if gold is None:
        raise ValueError(f'Invalid reference date: {target!r}')
    pred = parse_date(prediction)
    return math.exp(-alpha * abs((pred - gold).days) / 30) if pred else 0.0


def validate_search_count(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('executed_searches must be a nonnegative integer or None')
    return value


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _summary(predictions, targets, counts, prompts, responses, latencies):
    n = len(targets)
    errors = [abs((parse_date(p) - parse_date(g)).days) if parse_date(p) else None
              for p, g in zip(predictions, targets)]
    valid = [e for e in errors if e is not None]
    known_counts = [validate_search_count(c) for c in counts if c is not None]
    result = {'count': n, 'valid_prediction_count': len(valid),
              'prediction_coverage': len(valid) / n if n else 0.0,
              'invalid_rate': (n - len(valid)) / n if n else 0.0,
              'score_micro': sum(utility(p, g) for p, g in zip(predictions, targets)) / n if n else 0.0,
              'exact_match': sum(e == 0 for e in valid) / n if n else 0.0,
              'mae': sum(valid) / len(valid) if valid else None,
              'median_ae': quantile(valid, .5),
              'p75_ae': quantile(valid, .75), 'p90_ae': quantile(valid, .9), 'p95_ae': quantile(valid, .95),
              'search_count_coverage': len(known_counts) / n if n else 0.0,
              'search_rate': sum(c > 0 for c in known_counts) / len(known_counts) if known_counts else None,
              'avg_search_turns': sum(known_counts) / len(known_counts) if known_counts else None}
    for days in (1, 7, 30):
        result[f'acc_at_{days}'] = sum(e <= days for e in valid) / n if n else 0.0
    for key, values in [('avg_prompt_tokens', prompts), ('avg_response_tokens', responses),
                        ('latency', latencies)]:
        known = [v for v in values if v is not None]
        result[key] = sum(known) / len(known) if known else None
        result[key + '_coverage'] = len(known) / n if n else 0.0
    totals = [p + r for p, r in zip(prompts, responses) if p is not None and r is not None]
    result['avg_total_tokens'] = sum(totals) / len(totals) if totals else None
    for name, predicate in [('0', lambda c: c == 0), ('1', lambda c: c == 1),
                            ('2', lambda c: c == 2), ('gt2', lambda c: c > 2)]:
        result['search_turns_p_' + name] = sum(predicate(c) for c in known_counts) / len(known_counts) if known_counts else None
    return result, errors


def compute_metrics(predictions, targets, search_turns=None, prompt_tokens=None,
                    response_tokens=None, timing_info=None, difficulties=None):
    n = len(targets)
    if len(predictions) != n:
        raise ValueError('Prediction and reference lengths differ')
    for target in targets:
        reference_date(target)
    arrays = []
    for name, values in [('search_turns', search_turns), ('prompt_tokens', prompt_tokens),
                         ('response_tokens', response_tokens), ('timing_info', timing_info),
                         ('difficulties', difficulties)]:
        if values is None:
            values = [None] * n
        if len(values) != n:
            raise ValueError(f'{name} length differs from reference length')
        arrays.append(list(values))
    counts, prompts, responses, latencies, regimes = arrays
    counts = [validate_search_count(c) for c in counts]
    regimes = [normalize_regime(c) for c in regimes]
    metrics, errors = _summary(predictions, targets, counts, prompts, responses, latencies)
    group_scores = []
    for regime in sorted(set(regimes)):
        indexes = [i for i, value in enumerate(regimes) if value == regime]
        inputs = [[values[i] for i in indexes] for values in [predictions, targets, counts, prompts, responses, latencies]]
        subset, _ = _summary(*inputs)
        group_scores.append(subset['score_micro'])
        metrics.update({f'{regime}/{key}': value for key, value in subset.items()})
    metrics['score_macro'] = sum(group_scores) / len(group_scores) if group_scores else 0.0
    metrics['regime_count'] = len(group_scores)
    return metrics, [i for i, e in enumerate(errors) if e == 0], [i for i, e in enumerate(errors) if e != 0], errors


def validate_prediction_ids(expected, actual):
    expected, actual = list(expected), list(actual)
    if not expected:
        raise ValueError('Expected validation manifest is empty')
    for name, ids in [('expected', expected), ('actual', actual)]:
        if any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ValueError(f'{name} IDs must be nonempty strings')
        duplicates = [i for i, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise ValueError(f'Duplicate {name} IDs: {duplicates[:5]}')
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(f'Validation coverage mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}')


def write_prediction_manifest(path, expected_ids, rows):
    validate_prediction_ids(expected_ids, [row['id'] for row in rows])
    for row in rows:
        reference_date(row['gold'])
        validate_search_count(row.get('executed_searches'))
    rows = [{k: v for k, v in row.items() if not (k in ('executed_searches', 'failed_searches') and v is None)} for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
