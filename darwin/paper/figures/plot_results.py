import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# Full results (7 benchmarks):
#   GSM8K  MATH-500   AMC   Minerva  Olympiad  AIME24  AIME25
# Base:  72.6   68.2      47.5   42.3     34.8      6.7     10.3   → avg 40.3
# R-Zero:92.12  79.6      57.27  52.94    44.59     13.4    9.6    → avg 49.9
# Prism: 93.45  81.02     61.25  56.62    45.58     16.77   12.92  → avg 52.5
benchmarks = ['MATH-500', 'AMC', 'Minerva', 'Average']

models = {
    'Qwen3-4B-Base': [68.20, 47.50, 42.30, 40.34],
    'R-Zero':        [79.60, 57.27, 52.94, 49.93],
    'Prism':         [81.02, 61.25, 56.62, 52.52],
}

# ── Style ──────────────────────────────────────────────────────────────────────
colors = {
    'Qwen3-4B-Base': '#A8C4E0',   # muted blue  (like R-Zero bar in R-Few fig)
    'R-Zero':        '#F4A97F',   # muted orange
    'Prism':         '#E05C4B',   # bold red    (highlight our method)
}
hatches = {
    'Qwen3-4B-Base': '',
    'R-Zero':        '',
    'Prism':         '',
}

n_benchmarks = len(benchmarks)
n_models = len(models)
bar_width = 0.18
x = np.arange(n_benchmarks)

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor('white')

# ── Bars ───────────────────────────────────────────────────────────────────────
offsets = np.array([-1, 0, 1]) * (bar_width + 0.01)
for i, (model_name, scores) in enumerate(models.items()):
    bars = ax.bar(
        x + offsets[i], scores,
        width=bar_width,
        color=colors[model_name],
        label=model_name,
        edgecolor='white',
        linewidth=0.6,
        zorder=3,
    )
    # value labels on top of each bar
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{score:.1f}',
            ha='center', va='bottom',
            fontsize=7.5,
            color='#333333',
            fontweight='bold' if model_name == 'Prism' else 'normal',
        )


# ── Grid & axes ───────────────────────────────────────────────────────────────
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#DDDDDD', zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=11)
ax.set_ylabel('Pass@1 Accuracy (%)', fontsize=11)
ax.set_ylim(0, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.tick_params(axis='both', which='both', length=0)

# ── Title & legend ─────────────────────────────────────────────────────────────
ax.set_title('Prism vs. Baselines Across Mathematical Reasoning Benchmarks',
             fontsize=13, fontweight='bold', pad=14)

legend_patches = [
    mpatches.Patch(color=colors[m], label=m) for m in models
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=10,
          framealpha=0.9, edgecolor='#CCCCCC')

plt.tight_layout()
out = '/Users/vaibhav/Desktop/brahma/tree/euclid/darwin/paper/figures/main_results.pdf'
out_png = '/Users/vaibhav/Desktop/brahma/tree/euclid/darwin/paper/figures/main_results.png'
plt.savefig(out, bbox_inches='tight', dpi=300)
plt.savefig(out_png, bbox_inches='tight', dpi=300)
print(f'Saved → {out}')
print(f'Saved → {out_png}')
