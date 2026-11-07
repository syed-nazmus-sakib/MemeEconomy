import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from memeeconomy.run_single_agent import _stable_round_seed
from memeeconomy.prompts import (
    PRESSURE_FACTORIAL_CONDITIONS,
    build_system_prompt,
    get_condition_metadata,
)
from run_experiment import build_tasks, load_manifest, validate_manifest
from analyze import factorial_effect_bootstrap, paired_bootstrap


def dashboard_state():
    return {
        "karma": {"balance": 15_000},
        "current_round": 0,
        "total_rounds": 33,
        "performance": {
            "win_rate": 0,
            "current_streak": 0,
            "avg_karma_per_round": 0,
        },
        "behavioral": {"ethical_violations": 0},
        "removals": {"total": 0},
    }


class PressureDesignTests(unittest.TestCase):
    def test_manifest_and_complete_factorial(self):
        manifest = load_manifest()
        validate_manifest(manifest)
        self.assertEqual(len(manifest["core_conditions"]), 8)
        cells = {
            (
                x["target_karma"],
                x["elimination_threshold"] is not None,
                x["aggressive"],
            )
            for x in PRESSURE_FACTORIAL_CONDITIONS.values()
        }
        self.assertEqual(len(cells), 8)

    def test_prompt_factor_fidelity(self):
        for condition, spec in PRESSURE_FACTORIAL_CONDITIONS.items():
            prompt = build_system_prompt(condition, dashboard_state())
            self.assertIn(f"Target            : {spec['target_karma']:,}", prompt)
            self.assertEqual("# TERMINATION RULE" in prompt, spec["elimination_threshold"] is not None)
            self.assertEqual("Maximize karma aggressively" in prompt, spec["aggressive"])
            meta = get_condition_metadata(condition)
            self.assertEqual(meta["target_karma"], spec["target_karma"])

    def test_common_random_numbers_ignore_condition_and_model(self):
        seed_a = _stable_round_seed(3530401, 7, "dark_brandon")
        seed_b = _stable_round_seed(3530401, 7, "dark_brandon")
        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, _stable_round_seed(3530402, 7, "dark_brandon"))

    def test_each_replicate_has_identical_order_in_all_cells(self):
        tasks = build_tasks(load_manifest(), repeats=[1])
        orders = {tuple(task["instances"]) for task in tasks}
        seeds = {task["seed"] for task in tasks}
        self.assertEqual(len(orders), 1)
        self.assertEqual(seeds, {3530401})

    def test_paired_analysis_preserves_matching(self):
        rows = []
        for condition, target, elim, aggressive, outcome in [
            ("pf_t25_e0_a0", 0, 0, 0, 0),
            ("pf_t50_e0_a0", 1, 0, 0, 1),
        ]:
            rows.append({
                "condition": condition,
                "model": "gpt5_1",
                "replicate": 1,
                "instance": "dark_brandon",
                "high_target": target,
                "elimination": elim,
                "aggressive": aggressive,
                "t5": outcome,
            })
        contrast = paired_bootstrap(rows, "pf_t50_e0_a0", "pf_t25_e0_a0", draws=100)
        self.assertEqual(contrast["n_pairs"], 1)
        self.assertEqual(contrast["difference"], 1.0)
        effect = factorial_effect_bootstrap(rows, "high_target", draws=100)
        self.assertEqual(effect["n_pairs"], 1)
        self.assertEqual(effect["risk_difference"], 1.0)


if __name__ == "__main__":
    unittest.main()
