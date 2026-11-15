"""Compatibility entry points for revised temporal training and evaluation."""
from .temporal import (parse_date, reference_date, normalize_regime, extract_answer,
                       utility, validate_search_count, compute_metrics)
from .reward_objectives import cost_aware_reward


def date_prediction_reward(prediction, target, alpha=0.05, use_days=True):
    if not use_days:
        raise ValueError('Month-based scoring is not part of the revised protocol')
    return utility(prediction, target, alpha=alpha)


def extract_solution(solution_str):
    return extract_answer(solution_str)


def is_valid_date_format(date_str):
    return parse_date(date_str) is not None


def normalize_answer(text):
    return text.strip().lower() if isinstance(text, str) else ''


def compute_date_metrics(predictions, targets, search_turns=None, prompt_tokens=None,
                         response_tokens=None, timing_info=None, difficulties=None):
    return compute_metrics(predictions, targets, search_turns, prompt_tokens,
                           response_tokens, timing_info, difficulties)


def compute_score_em(solution_str, ground_truth, use_search_bonus=True,
                     reward_type='continuous', reward_tag='full',
                     eval_mode=False, executed_searches=None, cost_aware=None,
                     return_details=False):
    """Evaluation is pure utility; optional original shaping is training-only.

    ``cost_aware`` is a configurable, regime-independent quality/cost candidate.
    ``continuous`` is the matched objective shared by TIME-Agent/Search-R1
    controls. ``timeagent`` explicitly enables the original regime shaping.
    ``searchr1`` retains exact-match training as an objective ablation only.
    Search bonuses use executor counters, never transcript tags.
    """
    target = reference_date(ground_truth)
    answer = extract_solution(solution_str)
    if reward_type == 'cost_aware' and not eval_mode:
        details = cost_aware_reward(solution_str, target, executed_searches, cost_aware)
        return details if return_details else details['total_reward']

    def output(score):
        if not return_details:
            return score
        return {'reward_type': 'evaluation_utility' if eval_mode else reward_type,
                'evaluation_mode': bool(eval_mode), 'prediction': answer,
                'target': target, 'format_valid': parse_date(answer) is not None,
                'total_reward': score, 'executed_searches': executed_searches}

    if eval_mode or reward_type in ('continuous', 'naiverag'):
        return output(utility(answer, target))
    if reward_type == 'searchr1':
        return output(float(parse_date(answer) is not None and answer == target))
    if reward_type != 'timeagent':
        raise ValueError(f'Unknown reward_type: {reward_type}')
    if reward_tag not in ('full', 'use_search_reward', 'use_date_reward'):
        raise ValueError(f'Unknown reward_tag: {reward_tag}')
    regime = normalize_regime(ground_truth.get('difficulty') if isinstance(ground_truth, dict) else None)
    if regime == 'unknown':
        raise ValueError('Regime-shaped training requires EXP/IMP/DIS')
    if parse_date(answer) is None:
        return output(-0.5)
    bonus = 0.0
    if use_search_bonus and reward_tag != 'use_date_reward':
        count = validate_search_count(executed_searches)
        if count is None:
            raise ValueError('Search-shaped reward requires an executed_searches counter')
        if count > 0:
            bonus = -0.1 if regime == 'EXP' else 0.25
    date_reward = {'EXP': 1.0, 'IMP': 1.2, 'DIS': 1.5}[regime] * utility(answer, target)
    if reward_tag == 'use_search_reward':
        return output(bonus)
    return output(date_reward + bonus if reward_tag == 'full' else date_reward)
