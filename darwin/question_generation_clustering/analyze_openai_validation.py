#!/usr/bin/env python3
"""
Analyze OpenAI validation results.

Usage:
    python analyze_openai_validation.py \
        --input_file balanced_questions_darwin_iter2_openai_validated.json
"""

import argparse
import json
import numpy as np
from collections import defaultdict


def analyze_validation_results(input_file: str):
    """Analyze the OpenAI validation results."""
    
    print(f"[Load] Reading from {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Statistics
    total = len(data)
    validated = sum(1 for item in data if "openai_answer" in item)
    matched = sum(1 for item in data if item.get("openai_match") == 1)
    wrong_questions = sum(1 for item in data if item.get("openai_answer", "").lower() == "wrong")
    errors = sum(1 for item in data if item.get("openai_answer") == "ERROR")
    
    # Per-cluster statistics
    cluster_stats = defaultdict(lambda: {"total": 0, "matched": 0, "wrong": 0, "mismatched": 0, "error": 0})
    
    for item in data:
        if "openai_answer" not in item:
            continue
        
        cluster_id = item.get("cluster_id", -1)
        cluster_stats[cluster_id]["total"] += 1
        
        openai_answer = item.get("openai_answer", "")
        
        if openai_answer == "ERROR":
            cluster_stats[cluster_id]["error"] += 1
        elif openai_answer.lower() == "wrong":
            cluster_stats[cluster_id]["wrong"] += 1
        elif item.get("openai_match") == 1:
            cluster_stats[cluster_id]["matched"] += 1
        else:
            cluster_stats[cluster_id]["mismatched"] += 1
    
    # Per-score statistics
    score_ranges = {
        "0.0-0.3": (0.0, 0.3),
        "0.3-0.5": (0.3, 0.5),
        "0.5-0.7": (0.5, 0.7),
        "0.7-0.9": (0.7, 0.9),
        "0.9-1.0": (0.9, 1.0),
    }
    
    score_stats = {range_name: {"total": 0, "matched": 0, "wrong": 0, "mismatched": 0, "error": 0} 
                   for range_name in score_ranges}
    
    for item in data:
        if "openai_answer" not in item:
            continue
        
        score = item.get("score", 0)
        for range_name, (min_s, max_s) in score_ranges.items():
            if min_s <= score < max_s or (range_name == "0.9-1.0" and score >= 0.9):
                score_stats[range_name]["total"] += 1
                
                openai_answer = item.get("openai_answer", "")
                
                if openai_answer == "ERROR":
                    score_stats[range_name]["error"] += 1
                elif openai_answer.lower() == "wrong":
                    score_stats[range_name]["wrong"] += 1
                elif item.get("openai_match") == 1:
                    score_stats[range_name]["matched"] += 1
                else:
                    # OpenAI gave an answer, but it didn't match the majority answer
                    score_stats[range_name]["mismatched"] += 1
                break
    
    # Print results
    print(f"\n{'='*70}")
    print("VALIDATION ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Total questions: {total}")
    print(f"Validated: {validated} ({validated/total*100:.1f}%)")
    print(f"Matched (correct): {matched} ({matched/max(validated,1)*100:.1f}%)")
    print(f"Wrong questions: {wrong_questions} ({wrong_questions/max(validated,1)*100:.1f}%)")
    print(f"API errors: {errors} ({errors/max(validated,1)*100:.1f}%)")
    
    # Score-based analysis
    print(f"\n{'='*100}")
    print("VALIDATION BY SCORE RANGE")
    print(f"{'='*100}\n")
    print(f"{'Range':<12} {'Total':<8} {'Match':<8} {'Match%':<8} {'Wrong':<8} {'Wrong%':<8} {'Mismatch':<10} {'Mism%':<8} {'Error':<8} {'Err%':<8}")
    print("-" * 100)
    
    for range_name in ["0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]:
        stats = score_stats[range_name]
        if stats["total"] > 0:
            match_pct = stats["matched"] / stats["total"] * 100
            wrong_pct = stats["wrong"] / stats["total"] * 100
            mismatch_pct = stats["mismatched"] / stats["total"] * 100
            error_pct = stats["error"] / stats["total"] * 100
            print(f"{range_name:<12} {stats['total']:<8} {stats['matched']:<8} {match_pct:<8.1f} "
                  f"{stats['wrong']:<8} {wrong_pct:<8.1f} {stats['mismatched']:<10} {mismatch_pct:<8.1f} "
                  f"{stats['error']:<8} {error_pct:<8.1f}")
    
    # Cluster analysis
    print(f"\n{'='*90}")
    print("VALIDATION BY CLUSTER (Top 10 worst)")
    print(f"{'='*90}\n")
    
    cluster_match_rates = []
    for cid, stats in cluster_stats.items():
        if stats["total"] > 0:
            match_rate = stats["matched"] / stats["total"]
            cluster_match_rates.append((cid, match_rate, stats))
    
    cluster_match_rates.sort(key=lambda x: x[1])  # Sort by match rate (ascending)
    
    print(f"{'Cluster':<10} {'Total':<8} {'Match':<8} {'Match%':<8} {'Wrong':<8} {'Mismatch':<10} {'Error':<8}")
    print("-" * 90)
    
    for cid, match_rate, stats in cluster_match_rates[:10]:
        print(f"{cid:<10} {stats['total']:<8} {stats['matched']:<8} {match_rate*100:<8.1f} "
              f"{stats['wrong']:<8} {stats['mismatched']:<10} {stats['error']:<8}")
    
    # Overall cluster statistics
    cluster_match_rates_values = [rate for _, rate, stats in cluster_match_rates if stats["total"] >= 5]
    if cluster_match_rates_values:
        print(f"\nCluster match rate statistics (clusters with ≥5 questions):")
        print(f"  Mean: {np.mean(cluster_match_rates_values)*100:.1f}%")
        print(f"  Std: {np.std(cluster_match_rates_values)*100:.1f}%")
        print(f"  Min: {min(cluster_match_rates_values)*100:.1f}%")
        print(f"  Max: {max(cluster_match_rates_values)*100:.1f}%")
    
    # Total questions per cluster
    print(f"\n{'='*70}")
    print("TOTAL QUESTIONS PER CLUSTER")
    print(f"{'='*70}\n")
    
    # Sort by cluster ID
    cluster_totals = [(cid, stats["total"]) for cid, stats in cluster_stats.items()]
    cluster_totals.sort(key=lambda x: x[0])  # Sort by cluster ID
    
    print(f"{'Cluster ID':<12} {'Total Questions':<20}")
    print("-" * 35)
    
    for cid, total in cluster_totals:
        print(f"{cid:<12} {total:<20}")
    
    # Summary statistics
    total_counts = [total for _, total in cluster_totals]
    if total_counts:
        print(f"\nCluster distribution statistics:")
        print(f"  Total clusters with questions: {len(cluster_totals)}")
        print(f"  Mean questions per cluster: {np.mean(total_counts):.1f}")
        print(f"  Std: {np.std(total_counts):.1f}")
        print(f"  Min: {min(total_counts)}")
        print(f"  Max: {max(total_counts)}")
        print(f"  Median: {np.median(total_counts):.1f}")
    
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Analyze OpenAI validation results")
    parser.add_argument("--input_file", type=str, required=True, help="Validated JSON file")
    args = parser.parse_args()
    
    analyze_validation_results(args.input_file)


if __name__ == "__main__":
    main()
