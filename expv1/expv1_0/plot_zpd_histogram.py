#!/usr/bin/env python3
"""
Plot histogram of ZPD p_succ values from accepted.jsonl
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List


def load_p_succ_values(accepted_jsonl_path: str) -> List[float]:
    """Load all p_succ values from accepted.jsonl."""
    p_succ_values = []
    
    with open(accepted_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                zpd = data.get("zpd", {})
                p_succ = zpd.get("p_succ")
                if p_succ is not None:
                    p_succ_values.append(float(p_succ))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
    
    return p_succ_values


def plot_histogram(p_succ_values: List[float], output_path: str, title: str = "ZPD p_succ Distribution") -> None:
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
    
    print(f"\nZPD p_succ Statistics:")
    print(f"  Count: {len(p_succ_values)}")
    print(f"  Mean: {mean:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  Std: {std:.3f}")
    print(f"  Min: {min_val:.3f}")
    print(f"  Max: {max_val:.3f}")
    print(f"  25th percentile: {np.percentile(p_succ_array, 25):.3f}")
    print(f"  75th percentile: {np.percentile(p_succ_array, 75):.3f}")
    
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
    parser = argparse.ArgumentParser(description="Plot ZPD p_succ histogram from accepted.jsonl")
    parser.add_argument(
        "--accepted-path",
        type=str,
        required=True,
        help="Path to accepted.jsonl file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for histogram (default: same directory as accepted.jsonl)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the plot (default: auto-generated)"
    )
    args = parser.parse_args()
    
    accepted_path = Path(args.accepted_path)
    if not accepted_path.exists():
        raise FileNotFoundError(f"File not found: {accepted_path}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = accepted_path.parent / "zpd_histogram.png"
    
    # Generate title if not provided
    if args.title is None:
        model_name = accepted_path.parent.name.replace("_accepted", "").replace("_", " ")
        args.title = f"ZPD p_succ Distribution - {model_name}"
    
    # Load and plot
    print(f"Loading p_succ values from: {accepted_path}")
    p_succ_values = load_p_succ_values(str(accepted_path))
    
    if not p_succ_values:
        print("Error: No p_succ values found in the file!")
        return
    
    plot_histogram(p_succ_values, str(output_path), args.title)


if __name__ == "__main__":
    main()
