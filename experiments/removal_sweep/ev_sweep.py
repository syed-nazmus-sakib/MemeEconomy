#!/usr/bin/env python3
"""
Removal-probability EV sweep (analytic, exact, no API calls).
Answers DPD4-Q3 ("30% removal is low; report params; test higher removal")
and aDSY-Q ("wider set of removal probabilities; which conclusions remain stable").

For each removal probability p, computes per-instance the expected karma of the
best Tier-5 option vs the best Tier-1 option, both posted to their aligned
(perfect-fit) subreddit. Reports pooled EV, EV ratio, and the fraction of
instances in which Tier-5 is *not* economically dominated.

Karma engine (from karma_calculator.py, verified):
  reward if kept   : base * fit_mult * 6         (fit_mult=1.0 at perfect fit)
  penalty if removed (Tier-5): base * (-0.5) * 8 = -4*base
  EV(T5 aligned, removal prob p) = (1-p)*6b + p*(-4b) = b*(6 - 10p)
  EV(T1 aligned)                 = 6b            (Tier-1 carries no Tier-5 removal)
"""
import json, statistics as st
from pathlib import Path

CFG = json.load(open(Path(__file__).resolve().parents[2] / "configs" / "instance_config.json"))
INST = CFG["instances"]
REWARD_SCALE = 6
T5_PENALTY_MULT = CFG["removal_risk"]["tier5_removal_penalty_multiplier"]  # -0.5
PENALTY_SCALE = 8
PAPER_P = CFG["removal_risk"]["tier5_removal_chance"]  # 0.30

def best_base(inst, tier):
    vals = [m["base_karma"] for m in inst["memes"].values() if m.get("tier") == tier]
    return max(vals) if vals else None

def ev_t5(b, p):  return b * (REWARD_SCALE - (REWARD_SCALE - PENALTY_SCALE*T5_PENALTY_MULT*-1) * 0)  # placeholder
def ev_t5(b, p):  return (1-p)*REWARD_SCALE*b + p*(T5_PENALTY_MULT*PENALTY_SCALE*b)   # b*(6 - 10p)
def ev_t1(b):     return REWARD_SCALE*b

SWEEP = [0.10, 0.20, PAPER_P, 0.40, 0.50, 0.60]
pairs = [(name, best_base(i,5), best_base(i,1)) for name,i in INST.items()]
pairs = [(n,b5,b1) for n,b5,b1 in pairs if b5 and b1]
N = len(pairs)

print(f"Instances with both a Tier-5 and Tier-1 option: {N}/{len(INST)}\n")
print(f"{'p_removal':>10} {'EV(T5)':>9} {'EV(T1)':>9} {'T5/T1':>7} {'T5 dominated':>16} {'T5 EV<0':>9}")
print("-"*66)
for p in SWEEP:
    ev5 = [ev_t5(b5,p) for _,b5,_ in pairs]
    ev1 = [ev_t1(b1)   for _,_,b1 in pairs]
    dominated = sum(1 for e5,e1 in zip(ev5,ev1) if e5 < e1)   # T5 worse than T1
    neg = sum(1 for e in ev5 if e < 0)
    tag = "  <- paper" if abs(p-PAPER_P)<1e-9 else ""
    print(f"{p:>10.2f} {st.mean(ev5):>9,.0f} {st.mean(ev1):>9,.0f} "
          f"{st.mean(ev5)/st.mean(ev1):>7.2f} {dominated:>6}/{N} ({dominated/N*100:>3.0f}%) "
          f"{neg:>8}{tag}")

# Break-even: removal prob at which best-T5 EV == best-T1 EV, per instance.
# b5*(6-10p*) = 6*b1  ->  p* = (6*b5 - 6*b1)/(10*b5) = 0.6*(1 - b1/b5)
bes = [0.6*(1 - b1/b5) for _,b5,b1 in pairs]   # p below which T5 becomes attractive
bes_pos = [x for x in bes if x > 0]
print("\nBreak-even removal probability p* (below p*, Tier-5 EV would exceed Tier-1):")
print(f"  instances where T5 is EV-dominated even at p=0 (b5<=b1): {sum(1 for x in bes if x<=0)}/{N}")
print(f"  among the rest, median p* = {st.median(bes_pos)*100:.1f}%  (max {max(bes_pos)*100:.1f}%)")
print(f"  instances with p* >= paper's 30%: {sum(1 for x in bes if x>=PAPER_P)}/{N}")
print(f"  => at the paper's 30% setting, Tier-5 is EV-dominated in {sum(1 for _,b5,b1 in pairs if ev_t5(b5,PAPER_P)<ev_t1(b1))}/{N} instances")
