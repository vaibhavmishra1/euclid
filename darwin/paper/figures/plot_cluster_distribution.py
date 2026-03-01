import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────────
rzero = np.load("/Users/vaibhav/Desktop/brahma/tree/euclid/darwin/question_generation_clustering/cluster_frequencies_variant1iter3.npy")
prism = np.load("/Users/vaibhav/Desktop/brahma/tree/euclid/darwin/question_generation_clustering/cluster_frequencies_variant2iter4.npy")

n = len(rzero)
x = np.arange(n)

GRID = "#E8ECF0"
BG   = "white"

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.patch.set_facecolor(BG)

for ax, data, color, title in [
    (axes[0], rzero, "#4C9BE8", "R-Zero Questioner"),
    (axes[1], prism, "#F07B54", "Prism Questioner"),
]:
    ax.set_facecolor(BG)
    bars = ax.bar(x, data, width=0.75, color=color, alpha=0.88, zorder=3, linewidth=0)

    # subtle top-edge highlight on each bar
    for b in bars:
        if b.get_height() > 0:
            ax.plot([b.get_x(), b.get_x() + b.get_width()],
                    [b.get_height(), b.get_height()],
                    color=color, lw=1.2, alpha=0.6, zorder=4)

    # annotate the spike in R-Zero
    if color == "#4C9BE8":
        peak_idx = int(np.argmax(data))
        peak_val = int(data[peak_idx])
        ax.annotate(
            f"Cluster {peak_idx}\n({peak_val})",
            xy=(peak_idx, peak_val),
            xytext=(peak_idx + 5, peak_val * 0.88),
            fontsize=9, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        )

    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#C8CDD4")
    ax.tick_params(length=0, labelcolor="#6B7280")

    ax.set_ylabel("Frequency", fontsize=11, color="#374151", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#111827", pad=10)
    ax.set_xlim(-1, n)

axes[1].set_xlabel("Cluster Index", fontsize=11, color="#374151", labelpad=8)
axes[1].set_xticks(x[::5])
axes[1].set_xticklabels(x[::5], fontsize=8, color="#6B7280")

plt.tight_layout(h_pad=2.5)

out_png = "/Users/vaibhav/Desktop/brahma/tree/euclid/darwin/paper/figures/cluster_distribution.png"
out_pdf = "/Users/vaibhav/Desktop/brahma/tree/euclid/darwin/paper/figures/cluster_distribution.pdf"
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=BG)
plt.savefig(out_pdf, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
