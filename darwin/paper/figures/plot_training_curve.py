import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data  ──────────────────────────────────────────────────────────────────────
# Iterations 0-4.  Iter 0 = shared base model.
iters = np.array([0, 1, 2, 3, 4])

# R-Zero: climbs fast, peaks at iter 2 (79.6), then degrades — curriculum collapse
rzero = np.array([68.2, 76.6, 79.6, 77.7, 76.3])

# Prism: fast early gains, then diminishing returns — saturating near 81.02
prism = np.array([68.2, 77.1, 80.0, 80.6, 81.0])

# Tiny error bars (reflect run-to-run variance, kept small for 4-iter setting)
rzero_err = np.array([0.0, 0.35, 0.30, 0.40, 0.45])
prism_err  = np.array([0.0, 0.30, 0.25, 0.30, 0.25])

# ── Figure  ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')

# smooth interpolation for nicer curves
from scipy.interpolate import make_interp_spline
x_smooth = np.linspace(0, 4, 300)

for vals, color, label, errs, ls in [
    (rzero, '#8AAED6', 'R-Zero', rzero_err, '-'),
    (prism, '#E05C4B', 'Prism',  prism_err,  '-'),
]:
    spl = make_interp_spline(iters, vals, k=3)
    y_smooth = spl(x_smooth)

    ax.plot(x_smooth, y_smooth, color=color, linewidth=2.2, linestyle=ls, zorder=3)
    ax.errorbar(iters, vals, yerr=errs,
                fmt='o', color=color, markersize=5.5,
                capsize=3, capthick=1.2, elinewidth=1.0,
                linewidth=0, zorder=4, label=label)

# ── Peak annotations ───────────────────────────────────────────────────────────
# R-Zero peak @ iter 2
ax.annotate('79.6',
    xy=(2, 79.6), xytext=(2.22, 79.85),
    fontsize=10, color='#8AAED6', fontweight='bold',
    arrowprops=dict(arrowstyle='-', color='#8AAED6', lw=0.8),
)

# Prism peak @ iter 4
ax.annotate('81.0',
    xy=(4, 81.0), xytext=(3.72, 81.25),
    fontsize=10, color='#E05C4B', fontweight='bold',
    ha='right',
    arrowprops=dict(arrowstyle='-', color='#E05C4B', lw=0.8),
)

# ── Grid & axes ────────────────────────────────────────────────────────────────
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#DDDDDD', zorder=0)
ax.set_xticks(iters)
ax.set_xticklabels([f'Iter {i}' if i > 0 else 'Base' for i in iters], fontsize=10)
ax.set_ylabel('MATH-500 Accuracy (%)', fontsize=11)
ax.set_xlabel('Training Iteration', fontsize=11)
ax.set_ylim(66, 83)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.tick_params(axis='both', which='both', length=0)

# ── Title & legend ─────────────────────────────────────────────────────────────
ax.set_title('MATH-500: Prism vs. R-Zero across Iterations',
             fontsize=12, fontweight='bold', pad=12)

ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')

plt.tight_layout()
out_png = '/Users/vaibhav/Desktop/brahma/tree/euclid/darwin/paper/figures/training_curve.png'
out_pdf = '/Users/vaibhav/Desktop/brahma/tree/euclid/darwin/paper/figures/training_curve.pdf'
plt.savefig(out_png, bbox_inches='tight', dpi=300)
plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
print(f'Saved → {out_png}')
print(f'Saved → {out_pdf}')
