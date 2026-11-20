#!/usr/bin/env python3
"""Regression tests for the cost-aware training reward. Standard library only.

Run: PYTHONPATH=src python3 -m unittest tests.test_reward -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verl.utils.reward_score.qa_em import compute_score_em  # noqa: E402

ANSWER = "<answer>2024-03-11</answer>"
OPTIONS = {"scales_days": [7, 30, 365], "weights": [0.4, 0.4, 0.2], "search_cost": 0.01,
           "invalid_penalty": 0.3, "cost_by_regime": {"EXP": 0.2, "IMP": 0.01, "DIS": 0.05}}


def reward(regime, searches, options=OPTIONS):
    truth = {"target": ["2024-03-11"], "difficulty": regime}
    return compute_score_em(ANSWER, truth, reward_type="cost_aware", executed_searches=searches,
                            cost_aware=options, return_details=True)


class PerRegimeCost(unittest.TestCase):
    def test_each_regime_is_charged_its_own_cost(self):
        for regime, cost in (("EXP", 0.2), ("IMP", 0.01), ("DIS", 0.05)):
            with self.subTest(regime=regime):
                details = reward(regime, 2)
                self.assertAlmostEqual(details["search_cost_total"], 2 * cost)
                self.assertAlmostEqual(details["total_reward"], 1.0 - 2 * cost)

    def test_unlabelled_regime_falls_back_to_the_default_cost(self):
        self.assertAlmostEqual(reward("unknown", 3)["search_cost_total"], 3 * 0.01)

    def test_without_cost_by_regime_the_uniform_cost_applies_to_every_regime(self):
        options = dict(OPTIONS, cost_by_regime=None, search_cost=0.1)
        for regime in ("EXP", "IMP", "DIS"):
            with self.subTest(regime=regime):
                self.assertAlmostEqual(reward(regime, 1, options)["search_cost_total"], 0.1)

    def test_no_search_costs_nothing(self):
        self.assertEqual(reward("EXP", 0)["search_cost_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
