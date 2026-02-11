import numpy as np
import matplotlib.pyplot as plt

# Load cluster frequencies
cluster_freq = np.load("/workspace/euclid/darwin/question_generation_clustering/cluster_frequencies_darwin_iter2.npy")

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(cluster_freq)), cluster_freq, alpha=0.7, edgecolor='black')
plt.xlabel("Cluster ID", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title(f"Cluster Frequency Distribution ({len(cluster_freq)} clusters)", fontsize=14)
plt.grid(axis='y', alpha=0.3)

# Add statistics
stats_text = f"Min: {cluster_freq.min()}\nMax: {cluster_freq.max()}\nMean: {cluster_freq.mean():.2f}\nMedian: {np.median(cluster_freq):.2f}"
plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("cluster_frequencies_plot.png", dpi=300, bbox_inches='tight')
print(f"Plot saved to cluster_frequencies_plot.png")
print(f"\nStatistics:")
print(f"  Total clusters: {len(cluster_freq)}")
print(f"  Min frequency: {cluster_freq.min()}")
print(f"  Max frequency: {cluster_freq.max()}")
print(f"  Mean frequency: {cluster_freq.mean():.2f}")
print(f"  Median frequency: {np.median(cluster_freq):.2f}")
print(f"  Empty clusters: {np.sum(cluster_freq == 0)}")

plt.show()
