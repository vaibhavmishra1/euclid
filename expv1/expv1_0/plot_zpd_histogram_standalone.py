#!/usr/bin/env python3
"""
Standalone script to plot histogram of ZPD p_succ values from accepted.jsonl
Can be run directly without module imports to avoid naming conflicts.
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required. Install with: pip install matplotlib numpy")
    sys.exit(1)


def load_p_succ_values(accepted_jsonl_path: str):
    """Load all p_succ values from accepted.jsonl."""
    p_succ_values = []
    
    with open(accepted_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                zpd = data.get("zpd", {})
                p_succ = zpd.get("p_succ")
                if p_succ is not None:
                    p_succ_values.append(float(p_succ))
            except json.JSONDecodeError as e:
                if line_num <= 5:  # Only warn for first few errors
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    return p_succ_values


def plot_histogram(p_succ_values, output_path: str, title: str = "ZPD p_succ Distribution"):
    """Plot histogram of p_succ values."""
    if not p_succ_values:
        print("No p_succ values found!")
        return
    
    p_succ_array = np.array(p_succ_values)
    
    # Calculate statistics
    mean = np.mean(p_succ_array)
    median = np.median(p_succ_array)
    std = np.std(p_succ_array)
    min_val = np.min(p_succ_array)
    max_val = np.max(p_succ_array)
    q25 = np.percentile(p_succ_array, 25)
    q75 = np.percentile(p_succ_array, 75)
    
    print(f"\nZPD p_succ Statistics:")
    print(f"  Count: {len(p_succ_values)}")
    print(f"  Mean: {mean:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  Std: {std:.3f}")
    print(f"  Min: {min_val:.3f}")
    print(f"  Max: {max_val:.3f}")
    print(f"  25th percentile: {q25:.3f}")
    print(f"  75th percentile: {q75:.3f}")
    
    # Create histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram 1: Full distribution
    n_bins = min(50, len(set(p_succ_values)))  # Adaptive bin count
    ax1.hist(p_succ_array, bins=n_bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.3f}")
    ax1.axvline(median, color="green", linestyle="--", linewidth=2, label=f"Median: {median:.3f}")
    ax1.set_xlabel("p_succ (Solver Success Probability)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(f"{title} - Full Distribution (n={len(p_succ_values)})", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.0)
    
    # Histogram 2: Binned by difficulty zones
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["Very Hard\n(0.0-0.2)", "Hard\n(0.2-0.4)", "Medium\n(0.4-0.6)", "Easy\n(0.6-0.8)", "Very Easy\n(0.8-1.0)"]
    counts, _ = np.histogram(p_succ_array, bins=bins)
    
    colors = ["#d62728", "#ff7f0e", "#ffbb78", "#2ca02c", "#1f77b4"]
    bars = ax2.bar(range(len(labels)), counts, color=colors, edgecolor="black", alpha=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("ZPD Distribution by Difficulty Zones", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        percentage = (count / len(p_succ_values)) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nHistogram saved to: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_zpd_histogram_standalone.py <accepted.jsonl> [output.png] [title]")
        print("\nExample:")
        print("  python plot_zpd_histogram_standalone.py output_*/accepted.jsonl")
        sys.exit(1)
    
    accepted_path = Path(sys.argv[1])
    if not accepted_path.exists():
        print(f"Error: File not found: {accepted_path}")
        sys.exit(1)
    
    # Determine output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = accepted_path.parent / "zpd_histogram.png"
    
    # Generate title if not provided
    if len(sys.argv) >= 4:
        title = sys.argv[3]
    else:
        model_name = accepted_path.parent.name.replace("_accepted", "").replace("_", " ")
        title = f"ZPD p_succ Distribution - {model_name}"
    
    # Load and plot
    print(f"Loading p_succ values from: {accepted_path}")
    p_succ_values = load_p_succ_values(str(accepted_path))
    
    if not p_succ_values:
        print("Error: No p_succ values found in the file!")
        sys.exit(1)
    
    plot_histogram(p_succ_values, str(output_path), title)


if __name__ == "__main__":
    main()
