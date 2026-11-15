"""Configurable training objectives, separate from publication metrics.

The defaults are uncalibrated experiment settings, not validated optimal values.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re

from .temporal import parse_date, reference_date, validate_search_count


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return float(value)


@dataclass(frozen=True)
class CostAwareConfig:
    scales_days: tuple = (7.0, 30.0, 365.0)
    weights: tuple = (0.2, 0.3, 0.5)
    search_cost: float = 0.01
    invalid_penalty: float = 0.25
    cost_by_regime: tuple = None

    def __post_init__(self):
        for name, values in [('scales_days', self.scales_days), ('weights', self.weights)]:
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
                raise ValueError(f'{name} must be a nonempty sequence')
        if len(self.scales_days) != len(self.weights):
            raise ValueError('scales_days and weights must have equal lengths')
        scales = tuple(_number(v, 'scales_days', positive=True) for v in self.scales_days)
        weights = tuple(_number(v, 'weights') for v in self.weights)
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('weights must sum to 1; they are not silently normalized')
        object.__setattr__(self, 'scales_days', scales)
        object.__setattr__(self, 'weights', weights)
        object.__setattr__(self, 'search_cost', _number(self.search_cost, 'search_cost'))
        object.__setattr__(self, 'invalid_penalty', _number(self.invalid_penalty, 'invalid_penalty'))
        cbr = self.cost_by_regime
        if cbr is not None:
            if not isinstance(cbr, Mapping) or not cbr:
                raise ValueError('cost_by_regime must be a nonempty mapping')
            unknown = set(cbr) - {'EXP', 'IMP', 'DIS'}
            if unknown:
                raise ValueError(f'Unknown cost_by_regime regimes: {sorted(unknown)}')
            cbr = {k: _number(v, 'cost_by_regime.' + k) for k, v in cbr.items()}
            object.__setattr__(self, 'cost_by_regime', cbr)

    @classmethod
    def from_mapping(cls, options=None):
        if isinstance(options, cls):
            return options
        if options is None:
            options = {}
        if not isinstance(options, Mapping):
            raise ValueError('cost_aware must be a mapping')
        unknown = set(options) - {'scales_days', 'weights', 'search_cost', 'invalid_penalty', 'cost_by_regime'}
        if unknown:
            raise ValueError(f'Unknown cost_aware settings: {sorted(unknown)}')
        return cls(**dict(options))

    def as_dict(self):
        return {'scales_days': list(self.scales_days), 'weights': list(self.weights),
                'search_cost': self.search_cost, 'invalid_penalty': self.invalid_penalty,
                'cost_by_regime': dict(self.cost_by_regime) if self.cost_by_regime else None}


def strict_training_answer(model_response):
    """Require exactly one complete calendar-date answer outside observations.

    The caller must already exclude prompt/observation token IDs. Removing
    observation spans here is only a secondary check, not a token provenance API.
    """
    if not isinstance(model_response, str):
        return None, 'response_not_text'
    text = re.sub(r'<information>.*?(?:</information>|$)', '', model_response, flags=re.DOTALL)
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return None, 'expected_single_answer'
    matches = re.findall(r'<answer>(.*?)</answer>', text, flags=re.DOTALL)
    if len(matches) != 1:
        return None, 'malformed_answer_tags'
    answer = matches[0].strip()
    if parse_date(answer) is None:
        return None, 'invalid_calendar_date'
    return answer, None


def cost_aware_reward(model_response, ground_truth, executed_searches, options=None):
    """Return auditable components of Q(error) - c*N - invalid_penalty.

    N counts dispatched per-document attempts, including failed attempts.
    When cost_by_regime is configured, c is selected by the difficulty label
    (EXP/IMP/DIS) in ground_truth, falling back to search_cost otherwise.
    Date metadata or transcript search tags never affect the reward.
    """
    config = CostAwareConfig.from_mapping(options)
    target = reference_date(ground_truth)
    count = validate_search_count(executed_searches)
    if count is None:
        raise ValueError('cost_aware requires an executed_searches counter, including zero for no search')
    answer, reason = strict_training_answer(model_response)
    error = abs((parse_date(answer) - parse_date(target)).days) if answer is not None else None
    components = [weight * math.exp(-error / scale) for scale, weight in zip(config.scales_days, config.weights)] if error is not None else [0.0] * len(config.weights)
    quality = sum(components)
    if config.cost_by_regime:
        regime = ground_truth.get('difficulty') if isinstance(ground_truth, Mapping) else None
        cost_per_search = config.cost_by_regime.get(regime, config.search_cost)
    else:
        cost_per_search = config.search_cost
    cost = cost_per_search * count
    penalty = config.invalid_penalty if answer is None else 0.0
    return {'reward_type': 'cost_aware', 'evaluation_mode': False,
            'prediction': answer, 'target': target, 'format_valid': answer is not None,
            'invalid_reason': reason, 'error_days': error,
            'quality': quality, 'quality_components': components,
            'executed_searches': count, 'search_cost_total': cost,
            'invalid_penalty_applied': penalty, 'total_reward': quality - cost - penalty,
            'objective_config': config.as_dict()}
