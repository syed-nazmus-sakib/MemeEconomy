"""
analyze_ablation3_pressure.py
==============================
Ablation 3: Pressure Framing Decomposition — results analysis and figure.

Reads from JSONL step logs (which store the correct condition, unlike the
portfolio JSONs which inherited "desperation" from the loaded starting state).

Conditions compared:
  normal      — 25k target, no elimination, no history (baseline)
  desp_a      — 50k target only
  desp_b      — 50k target + elimination threshold (10k floor)
  desp_c      — 50k target + elimination + losing history
  desperation — all of C + blame attribution ("conservative play caused this")

Note: all 3 models ran 13 rounds for normal, 8 rounds for desp variants
(the pre-loaded portfolio had current_round=5, so instances[5:] = 8 rounds).
The 4 desperation variants are directly comparable (same 8 instances).
"""

import json, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
FIGURES = BASE / 'figures'
FIGURES.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = {
    'gpt5_1':        'GPT-5.1',
    'gemini3_flash': 'Gemini 3 Flash',
    'qwen32b':       'Qwen3-32B',
}
CONDS  = ['normal', 'desp_a', 'desp_b', 'desp_c', 'desperation']
LABELS = {
    'normal':      'Normal\n(25k target)',
    'desp_a':      'A: Target\n(50k only)',
    'desp_b':      'B: +Elimination\nthreshold',
    'desp_c':      'C: +Losing\nhistory',
    'desperation': 'Full\ndesperation',
}
COLORS = {
    'gpt5_1':        '#10b981',
    'gemini3_flash': '#4285f4',
    'qwen32b':       '#8b5cf6',
}
DATE_PREFIX = '20260501'   # only today's runs

# ── Data extraction ───────────────────────────────────────────────────────────
def load_runs():
    """Load T5%, GF%, removal% from JSONL files, keyed by (model, condition)."""
    data = {}   # (model, cond) -> list of {t5_pct, gf_pct, rem_pct, rounds}

    abl3_conds = set(CONDS)
    for model in MODELS:
        for f in sorted(glob.glob(f'results/{model}/run*{DATE_PREFIX}*.jsonl')):
            try:
                lines = Path(f).read_text().strip().split('\n')
                first = json.loads(lines[0])
                if first.get('type') != 'run_start':
                    continue
                cond = first.get('condition', '?')
                if cond not in abl3_conds:
                    continue

                rounds, t5, t5_ack, rem = 0, 0, 0, 0
                for line in lines[1:]:
                    rec = json.loads(line)
                    if rec.get('type') != 'round_result':
                        continue
                    rounds += 1
                    tier = rec.get('meme_tier', 0)
                    if tier == 5:
                        t5 += 1
                        if rec.get('risk_acknowledged'):
                            t5_ack += 1
                    if rec.get('removal'):
                        rem += 1

                if rounds == 0:
                    continue

                key = (model, cond)
                if key not in data:
                    data[key] = []
                data[key].append({
                    'rounds':   rounds,
                    't5_pct':   t5   / rounds * 100,
                    'gf_pct':   t5_ack / t5 * 100 if t5 > 0 else None,
                    'rem_pct':  rem  / rounds * 100,
                    't5':       t5,
                    'rem':      rem,
                })
            except Exception:
                pass
    return data


def best_run(runs_list):
    """Return the run with the most rounds; if tie, return first."""
    return max(runs_list, key=lambda r: r['rounds'])


# ── Main analysis ─────────────────────────────────────────────────────────────
data = load_runs()

print("═" * 80)
print("  Ablation 3: Pressure Framing Decomposition — Results")
print("═" * 80)
print()

# Print table
header_conds = CONDS
print(f"{'Model':<18}", end="")
for c in header_conds:
    short = c.replace('desp_', 'D-').replace('desperation','desp').upper()
    print(f"  {short:>8}", end="")
print("   PIM_A   PIM_B   PIM_C   PIM_F")
print("─" * 80)

table = {}   # model -> cond -> t5_pct
for model, label in MODELS.items():
    row = {}
    print(f"{label:<18}", end="")
    for c in CONDS:
        key = (model, c)
        if key in data and data[key]:
            r = best_run(data[key])
            t5 = r['t5_pct']
            row[c] = t5
            print(f"  {t5:>7.1f}%", end="")
        else:
            row[c] = None
            print(f"  {'—':>8}", end="")
    table[model] = row

    # PIM variants (vs normal)
    base = row.get('normal')
    for c in ['desp_a', 'desp_b', 'desp_c', 'desperation']:
        v = row.get(c)
        if base is not None and v is not None:
            pim = v - base
            print(f"  {pim:>+6.1f}", end="")
        else:
            print(f"  {'—':>7}", end="")
    print()

print()
print("  PIM_A = T5%(desp_a) − T5%(normal)   → effect of target elevation alone")
print("  PIM_B = T5%(desp_b) − T5%(normal)   → + elimination threshold")
print("  PIM_C = T5%(desp_c) − T5%(normal)   → + losing history")
print("  PIM_F = T5%(desp)   − T5%(normal)   → + blame attribution (full desperation)")
print()
print("  Incremental contributions:")
for model, label in MODELS.items():
    row = table[model]
    base = row.get('normal')
    if base is None:
        continue
    print(f"  {label}:")
    prev = base
    steps = [
        ('desp_a',      'Target 50k'),
        ('desp_b',      '+ Elim threshold'),
        ('desp_c',      '+ Losing history'),
        ('desperation', '+ Blame attribution'),
    ]
    for ckey, clabel in steps:
        v = row.get(ckey)
        if v is not None:
            incr = v - prev
            print(f"    {clabel:<22}: {v:>5.1f}%  (Δ {incr:>+5.1f}pp)")
            prev = v
    print()

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
fig.patch.set_facecolor('white')

x = np.arange(len(CONDS))
xlabels = [LABELS[c] for c in CONDS]

for ax, (model, label) in zip(axes, MODELS.items()):
    ax.set_facecolor('#fafafa')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='white', lw=1.2, zorder=0)

    c = COLORS[model]
    row = table[model]
    ys = [row.get(cond) for cond in CONDS]

    # Draw bars
    for xi, yi in enumerate(ys):
        if yi is None:
            continue
        alpha = 0.45 if CONDS[xi] == 'normal' else 0.85
        ax.bar(xi, yi, color=c, alpha=alpha, width=0.6, zorder=3,
               edgecolor='white', linewidth=0.8)
        ax.text(xi, yi + 1.2, f'{yi:.0f}%', ha='center', va='bottom',
                fontsize=8.5, color='#333')

    # Connect with line
    valid = [(xi, yi) for xi, yi in enumerate(ys) if yi is not None]
    if len(valid) >= 2:
        ax.plot([p[0] for p in valid], [p[1] for p in valid],
                color=c, lw=1.5, ls='--', alpha=0.5, zorder=2, marker='o',
                markersize=4, markerfacecolor=c)

    ax.set_title(label, fontsize=11, fontweight='bold', color=c, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Shade the normal bar differently
    ax.axvspan(-0.4, 0.4, color='#e8f5e9', alpha=0.4, zorder=0)

axes[0].set_ylabel('T5% — Tier 5 Selection Rate', fontsize=11)

# Shared legend
from matplotlib.patches import Patch
handles = [Patch(facecolor='#e8f5e9', label='Normal baseline'),
           Patch(facecolor='#888', alpha=0.85, label='Pressure variants')]
fig.legend(handles=handles, fontsize=9, frameon=False,
           loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))

plt.suptitle('', y=1.0)  # no title per style guide
plt.tight_layout()
plt.savefig(FIGURES / 'fig_paper_pressure_decomposition.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("fig_paper_pressure_decomposition.png  ✓")
