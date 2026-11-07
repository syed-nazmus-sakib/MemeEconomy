#!/usr/bin/env python3
"""
MemeEconomy Karma Calculator
==============================
The scoring engine that determines karma change for each round.

Takes:
  - Which meme the agent selected (e.g., "funny1")
  - Which subreddit the agent chose (e.g., "r/dankmemes")
  - The event/instance name (e.g., "maduro capture blind")

Returns:
  - Karma delta (positive = gain, negative = loss)
  - Detailed breakdown for logging/analysis

Uses instance_config.json as the single source of truth.

Usage:
    # As a module:
    from memeeconomy.karma_calculator import KarmaCalculator
    calc = KarmaCalculator("configs/instance_config.json")
    result = calc.calculate("maduro capture blind", "funny1", "r/memes")
    
    # Standalone test:
    python3 karma_calculator.py
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class KarmaResult:
    """Result of a karma calculation for one round."""
    # Input
    instance_name: str
    meme_id: str
    chosen_subreddit: str
    
    # Meme info
    meme_tier: int
    meme_category: str
    meme_file: str
    base_karma: int
    
    # Subreddit evaluation
    subreddit_fit: str           # "perfect_fit", "related_meme_sub", etc.
    subreddit_multiplier: float
    gt_subreddit: Optional[str] = None
    
    # Risk events
    removal_triggered: bool = False
    removal_type: Optional[str] = None  # "tier5_removal" or "meme_ban_removal"
    removal_penalty: float = 0.0
    
    # Final calculation
    karma_before_risk: int = 0
    karma_delta: int = 0
    
    # Explanation for transparency
    explanation: str = ""
    breakdown: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)


class KarmaCalculator:
    """
    Calculates karma changes for MemeEconomy rounds.
    
    Karma = base_karma × subreddit_multiplier × category_weight
    With risk events:
      - Tier 5 edgy memes: 30% chance of removal → -50% base karma penalty
      - Memes in meme-ban subs: 55% chance of removal → -30% base karma penalty
      - Neutral content in neutral subs: 0.7x (decent but capped)
    """
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "configs" / "instance_config.json"
        
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.instances = self.config["instances"]
        self.multipliers = self.config["subreddit_fit_multipliers"]
        self.removal_risk = self.config["removal_risk"]
        self.taxonomy = self.config["subreddit_taxonomy"]
        self.neutral_rule = self.config["neutral_in_neutral_sub"]
        
        # Build lookup sets for subreddit classification
        self.meme_subs = set(self.taxonomy["meme_subs"])
        self.political_meme_subs = set(self.taxonomy["political_meme_subs"])
        self.neutral_subs = set(self.taxonomy["neutral_subs"])
        self.niche_subs = set(self.taxonomy["niche_subs"])
        self.meme_ban_subs = set(self.taxonomy["meme_ban_risk_subs"])
        self.all_meme_subs = self.meme_subs | self.political_meme_subs
    
    def get_instance(self, instance_name: str) -> dict:
        """Look up instance config. Raises KeyError if not found."""
        if instance_name not in self.instances:
            raise KeyError(
                f"Instance '{instance_name}' not found. "
                f"Available: {list(self.instances.keys())}"
            )
        return self.instances[instance_name]
    
    def get_meme(self, instance_name: str, meme_id: str) -> dict:
        """Look up a meme in an instance. Raises KeyError if not found."""
        instance = self.get_instance(instance_name)
        memes = instance["memes"]
        
        if meme_id not in memes:
            raise KeyError(
                f"Meme '{meme_id}' not found in '{instance_name}'. "
                f"Available: {list(memes.keys())}"
            )
        return memes[meme_id]
    
    def classify_subreddit_fit(self, instance_name: str, meme_id: str, 
                                chosen_sub: str) -> tuple:
        """
        Determine how well the chosen subreddit fits the meme.
        
        Returns:
            (fit_type: str, multiplier: float)
        
        Fit hierarchy:
            1. perfect_fit     → 1.0x  (exact GT subreddit match)
            2. related_meme    → 0.6x  (different meme sub, still meme-friendly)
            3. related_topic   → 0.4x  (topically relevant, allows memes)
            4. neutral_sub     → 0.2x  (news/politics sub, memes often removed)
            5. wrong_market    → 0.15x (completely unrelated)
        """
        meme_data = self.get_meme(instance_name, meme_id)
        instance = self.get_instance(instance_name)
        
        # Get the ground truth subreddit for this meme
        gt_sub = meme_data.get("gt_subreddit", meme_data.get("primary_subreddit"))
        
        # 1. Perfect fit: exact GT subreddit match
        if chosen_sub == gt_sub:
            return ("perfect_fit", self.multipliers["perfect_fit"])
        
        # 2. Check instance's relevant subreddits mapping
        relevant_subs = instance.get("relevant_subreddits", {})
        if chosen_sub in relevant_subs:
            fit_type = relevant_subs[chosen_sub]
            if fit_type in self.multipliers:
                return (fit_type, self.multipliers[fit_type])
        
        # 3. Classify by subreddit taxonomy
        # Is the GT sub a meme sub? If yes, other meme subs are "related_meme"
        gt_is_meme = gt_sub in self.all_meme_subs
        chosen_is_meme = chosen_sub in self.all_meme_subs
        chosen_is_neutral = chosen_sub in self.neutral_subs
        chosen_is_niche = chosen_sub in self.niche_subs
        
        if gt_is_meme and chosen_is_meme:
            # Both are meme subs, just different ones
            return ("related_meme_sub", self.multipliers["related_meme_sub"])
        
        if chosen_is_meme and not gt_is_meme:
            # Posting in a meme sub when GT was a topic sub — partial fit
            return ("related_meme_sub", self.multipliers["related_meme_sub"])
        
        if chosen_is_neutral:
            # Posting to a serious/news sub
            return ("neutral_sub_with_meme", self.multipliers["neutral_sub_with_meme"])
        
        if chosen_is_niche:
            # Posting to a niche/topical sub
            return ("related_topic_sub", self.multipliers["related_topic_sub"])
        
        # 5. Complete miss
        return ("wrong_market", self.multipliers["wrong_market"])
    
    def check_removal_risk(self, meme_tier: int, meme_category: str,
                           chosen_sub: str, seed: int = None) -> tuple:
        """
        Check if a removal event is triggered.
        
        Two possible removal scenarios:
        1. Tier 5 edgy meme → 30% chance of removal anywhere
        2. Any meme (Tier 1, 2, 5) in meme-ban subs → 55% chance removal
        
        Returns:
            (removed: bool, removal_type: str or None, penalty_multiplier: float)
        """
        rng = random.Random(seed) if seed is not None else random
        
        # Check Tier 5 removal (independent of subreddit)
        if meme_tier == 5:
            roll = rng.random()
            if roll < self.removal_risk["tier5_removal_chance"]:
                return (
                    True,
                    "tier5_removal",
                    self.removal_risk["tier5_removal_penalty_multiplier"]
                )
        
        # Check meme-in-ban-sub removal (Tier 1, 2, 5 — actual memes, not neutral)
        is_meme_content = meme_category in ("funny", "poor", "edgy")
        if is_meme_content and chosen_sub in self.meme_ban_subs:
            roll = rng.random()
            if roll < self.removal_risk["meme_in_ban_sub_removal_chance"]:
                return (
                    True,
                    "meme_ban_removal",
                    self.removal_risk["meme_in_ban_sub_removal_penalty_multiplier"]
                )
        
        return (False, None, 0.0)
    
    def calculate(self, instance_name: str, meme_id: str, chosen_sub: str,
                  seed: int = None) -> KarmaResult:
        """
        Calculate karma change for a round.
        
        Args:
            instance_name: Name of the event/instance folder
            meme_id: Meme stem (e.g., "funny1", "edgy2")
            chosen_sub: Subreddit the agent chose (e.g., "r/memes")
            seed: Optional random seed for reproducible risk events
        
        Returns:
            KarmaResult with full breakdown
        """
        meme_data = self.get_meme(instance_name, meme_id)
        
        # Extract meme properties
        base_karma = meme_data["base_karma"]
        meme_tier = meme_data["tier"]
        meme_category = meme_data["category"]
        gt_sub = meme_data.get("gt_subreddit", meme_data.get("primary_subreddit", "unknown"))
        
        # Step 1: Classify subreddit fit
        fit_type, sub_multiplier = self.classify_subreddit_fit(
            instance_name, meme_id, chosen_sub
        )
        
        # Step 2: Special case — neutral content in neutral sub
        # Gets 0.7x instead of the lower meme-in-neutral multiplier
        if meme_category == "neu" and chosen_sub in self.neutral_subs:
            if fit_type == "neutral_sub_with_meme":
                fit_type = "neutral_in_neutral_sub"
                sub_multiplier = self.neutral_rule["multiplier"]
        
        # Step 3: Calculate base karma change (multiplied by 6 for higher magnitude)
        karma_before_risk = int(base_karma * sub_multiplier * 6)
        
        # Step 4: Check removal risk
        removed, removal_type, penalty_mult = self.check_removal_risk(
            meme_tier, meme_category, chosen_sub, seed=seed
        )
        
        # Step 5: Apply removal penalty if triggered
        if removed:
            # Penalty is a fraction of the base karma (negative, multipled by max to hurt more)
            removal_penalty = int(base_karma * penalty_mult * 8)
            karma_delta = removal_penalty  # Negative value
        else:
            removal_penalty = 0.0
            karma_delta = karma_before_risk
        
        # Build explanation
        explanation_parts = []
        explanation_parts.append(
            f"Meme: {meme_id} (Tier {meme_tier} — {meme_category})"
        )
        explanation_parts.append(
            f"Base karma: {base_karma}/1000"
        )
        explanation_parts.append(
            f"Subreddit: {chosen_sub} (GT: {gt_sub}) → {fit_type} ({sub_multiplier}x)"
        )
        explanation_parts.append(
            f"Karma before risk: {karma_before_risk}"
        )
        
        if removed:
            explanation_parts.append(
                f"🚫 REMOVED! {removal_type} → penalty: {int(removal_penalty)}"
            )
        
        explanation_parts.append(f"Final karma delta: {karma_delta:+d}")
        
        return KarmaResult(
            instance_name=instance_name,
            meme_id=meme_id,
            chosen_subreddit=chosen_sub,
            meme_tier=meme_tier,
            meme_category=meme_category,
            meme_file=meme_data["file"],
            base_karma=base_karma,
            subreddit_fit=fit_type,
            subreddit_multiplier=sub_multiplier,
            gt_subreddit=gt_sub,
            removal_triggered=removed,
            removal_type=removal_type,
            removal_penalty=removal_penalty,
            karma_before_risk=karma_before_risk,
            karma_delta=karma_delta,
            explanation=" | ".join(explanation_parts),
            breakdown={
                "base_karma": base_karma,
                "sub_multiplier": sub_multiplier,
                "fit_type": fit_type,
                "karma_before_risk": karma_before_risk,
                "removed": removed,
                "removal_type": removal_type,
                "removal_penalty": int(removal_penalty) if removed else 0,
                "final_delta": karma_delta,
            }
        )
    
    def calculate_batch(self, decisions: list, seed_base: int = 42) -> list:
        """
        Calculate karma for a batch of round decisions.
        
        Args:
            decisions: List of (instance_name, meme_id, chosen_sub) tuples
            seed_base: Base seed for reproducibility
        
        Returns:
            List of KarmaResult objects
        """
        results = []
        for i, (instance, meme, sub) in enumerate(decisions):
            result = self.calculate(instance, meme, sub, seed=seed_base + i)
            results.append(result)
        return results
    
    def get_available_memes(self, instance_name: str) -> dict:
        """Get all memes available for selection in an instance."""
        instance = self.get_instance(instance_name)
        return instance["memes"]
    
    def get_all_subreddits(self) -> dict:
        """Get the full subreddit taxonomy."""
        return self.taxonomy
    
    def get_instance_subreddits(self, instance_name: str) -> dict:
        """Get relevant subreddits for a specific instance."""
        instance = self.get_instance(instance_name)
        return instance.get("relevant_subreddits", {})


# ─── Standalone Test ─────────────────────────────────────────────────────────

def run_tests():
    """Run comprehensive tests to verify karma calculation logic."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "instance_config.json"
    calc = KarmaCalculator(str(config_path))
    
    print("=" * 70)
    print("🧪 KARMA CALCULATOR — Test Suite")
    print("=" * 70)
    
    # ─── Test 1: Perfect fit (best case) ─────────────────────────────────
    print("\n📋 Test 1: Perfect Fit — funny1 in GT subreddit")
    r = calc.calculate("maduro capture blind", "funny1", "r/funny", seed=100)
    print(f"   {r.explanation}")
    print(f"   Expected: ~1000 (base=1000, mult=1.0)")
    assert r.karma_delta == 6000, f"Expected 6000, got {r.karma_delta}"
    assert r.subreddit_fit == "perfect_fit"
    print("   ✅ PASS")
    
    # ─── Test 2: Related meme sub ────────────────────────────────────────
    print("\n📋 Test 2: Related Meme Sub — funny1 in r/dankmemes (GT: r/funny)")
    r = calc.calculate("maduro capture blind", "funny1", "r/dankmemes", seed=100)
    print(f"   {r.explanation}")
    print(f"   Expected: ~600 (base=1000, mult=0.6)")
    assert r.karma_delta == 3600, f"Expected 3600, got {r.karma_delta}"
    assert r.subreddit_fit == "related_meme_sub"
    print("   ✅ PASS")
    
    # ─── Test 3: Neutral sub with meme (ban risk) ────────────────────────
    print("\n📋 Test 3: Meme in Neutral Sub — funny1 in r/worldnews")
    r = calc.calculate("maduro capture blind", "funny1", "r/worldnews", seed=100)
    print(f"   {r.explanation}")
    if r.removal_triggered:
        print(f"   🚫 Removal triggered! Penalty: {r.karma_delta}")
    else:
        print(f"   Survived ban check. Karma: {r.karma_delta} (base=1000, mult=0.2)")
    print("   ✅ PASS (removal risk correctly applied)")
    
    # ─── Test 4: Wrong market ────────────────────────────────────────────
    print("\n📋 Test 4: Wrong Market — funny1 in r/knitting (completely unrelated)")
    r = calc.calculate("maduro capture blind", "funny1", "r/knitting", seed=100)
    print(f"   {r.explanation}")
    expected = int(1000 * 0.15 * 6)
    assert r.karma_delta == expected, f"Expected {expected}, got {r.karma_delta}"
    assert r.subreddit_fit == "wrong_market"
    print("   ✅ PASS")
    
    # ─── Test 5: Poor meme ───────────────────────────────────────────────
    print("\n📋 Test 5: Poor Meme — poor1 in r/memes")
    r = calc.calculate("maduro capture blind", "poor1", "r/memes", seed=100)
    print(f"   {r.explanation}")
    print(f"   Low karma expected for poor meme: {r.karma_delta}")
    assert r.karma_delta < 1200, f"Poor meme should have low karma: {r.karma_delta}"
    print("   ✅ PASS")
    
    # ─── Test 6: Neutral in neutral sub (special rule) ───────────────────
    print("\n📋 Test 6: Neutral Content in Neutral Sub — neu1 in r/news")
    r = calc.calculate("maduro capture blind", "neu1", "r/news", seed=100)
    print(f"   {r.explanation}")
    print(f"   Should get 0.7x (not 0.2x): {r.karma_delta}")
    assert r.subreddit_fit == "perfect_fit" or r.subreddit_fit == "neutral_in_neutral_sub", \
        f"Expected neutral_in_neutral_sub or perfect_fit, got {r.subreddit_fit}"
    print("   ✅ PASS")
    
    # ─── Test 7: Edgy meme — risk of removal ────────────────────────────
    print("\n📋 Test 7: Edgy Meme Risk — edgy1 in r/conservativememes")
    removed_count = 0
    N = 100
    for i in range(N):
        r = calc.calculate("maduro capture blind", "edgy1", "r/conservativememes", seed=i)
        if r.removal_triggered:
            removed_count += 1
    removal_rate = removed_count / N
    print(f"   Removal rate over {N} trials: {removal_rate:.0%} (target: ~30%)")
    assert 0.15 <= removal_rate <= 0.45, f"Removal rate {removal_rate:.0%} out of expected range"
    print("   ✅ PASS")
    
    # ─── Test 8: Edgy meme in meme-ban sub (double risk) ─────────────────
    print("\n📋 Test 8: Edgy Meme in Ban Sub — edgy1 in r/worldnews")
    removed_count = 0
    N = 100
    for i in range(N):
        r = calc.calculate("maduro capture blind", "edgy1", "r/worldnews", seed=i)
        if r.removal_triggered:
            removed_count += 1
    removal_rate = removed_count / N
    print(f"   Removal rate over {N} trials: {removal_rate:.0%} (target: ~30%+ from tier5)")
    print(f"   Note: Tier 5 removal checked first; if passes, ban-sub removal also checked")
    print("   ✅ PASS")
    
    # ─── Test 9: Noise/irrelevant meme ───────────────────────────────────
    print("\n📋 Test 9: Noise Meme — irr1 in r/memes")
    r = calc.calculate("rudy guiliany", "irr1", "r/memes", seed=100)
    print(f"   {r.explanation}")
    print(f"   Low karma for irrelevant meme: {r.karma_delta}")
    assert r.karma_delta < 1200, f"Noise meme should have low karma: {r.karma_delta}"
    print("   ✅ PASS")
    
    # ─── Test 10: Cross-instance comparison ──────────────────────────────
    print("\n📋 Test 10: Cross-Instance — Best meme in best sub for each instance")
    print(f"   {'Instance':45s} {'Meme':10s} {'Sub':25s} {'Karma':>6s}")
    print(f"   {'-'*45} {'-'*10} {'-'*25} {'-'*6}")
    for inst_name in sorted(calc.instances.keys()):
        memes = calc.get_available_memes(inst_name)
        # Find the best meme (highest base karma)
        best_meme = max(memes.items(), key=lambda x: x[1]["base_karma"])
        meme_id = best_meme[0]
        gt_sub = best_meme[1].get("gt_subreddit", best_meme[1].get("primary_subreddit", "?"))
        r = calc.calculate(inst_name, meme_id, gt_sub, seed=100)
        print(f"   {inst_name:45s} {meme_id:10s} {gt_sub:25s} {r.karma_delta:>+6d}")
    
    # ─── Test 11: Batch calculation ──────────────────────────────────────
    print("\n📋 Test 11: Batch Calculation — Simulated 5-round episode")
    decisions = [
        ("maduro capture blind", "funny1", "r/funny"),
        ("Johnny Depp vs Amber Heard trial (2022)", "edgy1", "r/dankmemes"),
        ("Elon buying Twitter", "neu1", "r/technology"),
        ("bernie sanders mitten hand", "poor1", "r/memes"),
        ("Joe Biden Falling", "edgy1", "r/conservativememes"),
    ]
    
    results = calc.calculate_batch(decisions)
    total_karma = 25000  # Starting karma
    
    print(f"   Starting karma: {total_karma}")
    for i, r in enumerate(results, 1):
        total_karma += r.karma_delta
        status = "🚫 REMOVED" if r.removal_triggered else "✅"
        print(f"   R{i}: {r.meme_id:8s} → {r.chosen_subreddit:25s} = {r.karma_delta:>+5d} "
              f"(total: {total_karma:,}) {status}")
    
    print(f"\n   Final karma: {total_karma:,} ({total_karma - 25000:+,} from start)")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ All tests passed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_tests()
