#!/usr/bin/env python3
"""
Plot concept performance over iterations from cumulative evaluation results.
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cumulative_file", required=True, help="Path to cumulative_evaluation.json")
    parser.add_argument("--output_dir", required=True, help="Directory to save plots")
    args = parser.parse_args()
    
    # Load cumulative results
    with open(args.cumulative_file, 'r') as f:
        results = json.load(f)
    
    # Group by concept and iteration
    concept_data = defaultdict(lambda: defaultdict(list))  # concept -> iteration -> [scores]
    
    for result in results:
        concept = result["concept"]
        iteration = result["iteration"]
        avg_score = result["avg_score"]
        concept_data[concept][iteration].append(avg_score)
    
    # Compute average per concept per iteration
    concept_avg_scores = {}
    for concept, iter_scores in concept_data.items():
        concept_avg_scores[concept] = {}
        for iteration, scores in sorted(iter_scores.items()):
            concept_avg_scores[concept][iteration] = np.mean(scores)
    
    # Create plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Individual concept lines
    plt.figure(figsize=(12, 8))
    for concept, scores_by_iter in sorted(concept_avg_scores.items()):
        iterations = sorted(scores_by_iter.keys())
        avg_scores = [scores_by_iter[i] for i in iterations]
        plt.plot(iterations, avg_scores, marker='o', label=concept, linewidth=2, markersize=8)
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Average Majority Voting Score", fontsize=12)
    plt.title("Model Performance on Held-Out Questions by Concept", fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_dir / "concept_performance_over_iterations.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    plt.close()
    
    # Plot 2: Bar chart showing improvement (first vs last iteration)
    concepts = sorted(concept_avg_scores.keys())
    first_scores = []
    last_scores = []
    
    for concept in concepts:
        scores_by_iter = concept_avg_scores[concept]
        if len(scores_by_iter) > 0:
            first_iter = min(scores_by_iter.keys())
            last_iter = max(scores_by_iter.keys())
            first_scores.append(scores_by_iter[first_iter])
            last_scores.append(scores_by_iter[last_iter])
        else:
            first_scores.append(0)
            last_scores.append(0)
    
    x = np.arange(len(concepts))
    width = 0.35
    
    plt.figure(figsize=(14, 8))
    plt.bar(x - width/2, first_scores, width, label='First Iteration', alpha=0.8)
    plt.bar(x + width/2, last_scores, width, label='Last Iteration', alpha=0.8)
    
    plt.xlabel("Concept", fontsize=12)
    plt.ylabel("Average Majority Voting Score", fontsize=12)
    plt.title("Performance Improvement: First vs Last Iteration", fontsize=14, fontweight='bold')
    plt.xticks(x, concepts, rotation=45, ha='right', fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot_file2 = output_dir / "concept_improvement_comparison.png"
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file2}")
    plt.close()
    
    # Plot 3: Heatmap (concept x iteration)
    concepts_list = sorted(concept_avg_scores.keys())
    max_iter = max((max(scores_by_iter.keys()) for scores_by_iter in concept_avg_scores.values()), default=0)
    
    heatmap_data = []
    for concept in concepts_list:
        row = []
        scores_by_iter = concept_avg_scores[concept]
        for iter_num in range(max_iter + 1):
            row.append(scores_by_iter.get(iter_num, np.nan))
        heatmap_data.append(row)
    
    plt.figure(figsize=(max(10, max_iter + 1), max(8, len(concepts_list) * 0.5)))
    im = plt.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, label='Average Score')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Concept", fontsize=12)
    plt.title("Concept Performance Heatmap", fontsize=14, fontweight='bold')
    plt.yticks(range(len(concepts_list)), concepts_list, fontsize=10)
    plt.xticks(range(max_iter + 1), range(max_iter + 1), fontsize=10)
    plt.tight_layout()
    
    plot_file3 = output_dir / "concept_performance_heatmap.png"
    plt.savefig(plot_file3, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file3}")
    plt.close()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for concept in concepts:
        scores_by_iter = concept_avg_scores[concept]
        if len(scores_by_iter) > 0:
            first_iter = min(scores_by_iter.keys())
            last_iter = max(scores_by_iter.keys())
            first_score = scores_by_iter[first_iter]
            last_score = scores_by_iter[last_iter]
            improvement = last_score - first_score
            print(f"{concept}:")
            print(f"  First iteration (iter {first_iter}): {first_score:.3f}")
            print(f"  Last iteration (iter {last_iter}): {last_score:.3f}")
            print(f"  Improvement: {improvement:+.3f}")
            print()


if __name__ == "__main__":
    main()
