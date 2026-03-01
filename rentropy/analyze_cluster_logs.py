#!/usr/bin/env python3
"""
Analyze cluster statistics logged during Rentropy training.

Usage:
    python analyze_cluster_logs.py --log_dir /path/to/storage/cluster_logs
    python analyze_cluster_logs.py --log_dir /path/to/storage/cluster_logs --output summary.json
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_logs(log_dir):
    """Load all cluster log files from directory."""
    log_files = sorted(Path(log_dir).glob("cluster_stats_step_*.json"))
    
    logs = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            logs.append(json.load(f))
    
    return logs


def analyze_logs(logs):
    """Analyze cluster logs and compute summary statistics."""
    if not logs:
        return {"error": "No logs found"}
    
    num_clusters = logs[0]["cluster_stats"]["num_clusters"]
    
    # Track per-cluster statistics across all steps
    cluster_question_counts = defaultdict(int)  # Total questions per cluster
    cluster_score_sums = defaultdict(float)     # Sum of scores per cluster
    cluster_score_counts = defaultdict(int)     # Count of scores per cluster
    
    # Track evolution over time
    steps = []
    cluster_counts_over_time = []
    
    for log in logs:
        step = log["step"]
        steps.append(step)
        
        # Cluster counts (EMA-tracked visit frequency)
        cluster_counts_over_time.append(log["cluster_stats"]["cluster_counts"])
        
        # Per-batch cluster distribution and scores
        batch_dist = log.get("batch_cluster_distribution", {})
        batch_scores = log.get("batch_cluster_avg_scores", {})
        
        for cid_str, count in batch_dist.items():
            cid = int(cid_str)
            cluster_question_counts[cid] += count
            
            if cid_str in batch_scores:
                score = batch_scores[cid_str]
                cluster_score_sums[cid] += score * count
                cluster_score_counts[cid] += count
    
    # Compute overall averages
    cluster_avg_scores = {
        cid: cluster_score_sums[cid] / cluster_score_counts[cid]
        for cid in cluster_score_counts if cluster_score_counts[cid] > 0
    }
    
    # Compute final cluster probabilities (from last step)
    final_probs = logs[-1]["cluster_stats"]["cluster_probabilities"]
    
    # Summary statistics
    summary = {
        "num_steps": len(logs),
        "num_clusters": num_clusters,
        "total_questions": sum(cluster_question_counts.values()),
        "clusters_visited": len(cluster_question_counts),
        "clusters_never_visited": num_clusters - len(cluster_question_counts),
        
        "per_cluster_stats": {
            cid: {
                "total_questions": cluster_question_counts[cid],
                "avg_majority_vote_score": cluster_avg_scores.get(cid, 0.0),
                "final_probability": final_probs[cid],
                "question_percentage": 100 * cluster_question_counts[cid] / sum(cluster_question_counts.values())
            }
            for cid in range(num_clusters)
        },
        
        "top_10_most_visited": sorted(
            [(cid, cluster_question_counts[cid]) for cid in cluster_question_counts],
            key=lambda x: x[1],
            reverse=True
        )[:10],
        
        "top_10_highest_scores": sorted(
            [(cid, cluster_avg_scores[cid]) for cid in cluster_avg_scores],
            key=lambda x: x[1],
            reverse=True
        )[:10],
        
        "evolution": {
            "steps": steps,
            "cluster_counts": cluster_counts_over_time,
        }
    }
    
    return summary


def print_summary(summary):
    """Print human-readable summary."""
    print("=" * 80)
    print("RENTROPY CLUSTER STATISTICS SUMMARY")
    print("=" * 80)
    print()
    
    print(f"Training Steps: {summary['num_steps']}")
    print(f"Total Clusters: {summary['num_clusters']}")
    print(f"Clusters Visited: {summary['clusters_visited']}")
    print(f"Clusters Never Visited: {summary['clusters_never_visited']}")
    print(f"Total Questions Generated: {summary['total_questions']}")
    print()
    
    print("-" * 80)
    print("TOP 10 MOST VISITED CLUSTERS")
    print("-" * 80)
    print(f"{'Cluster ID':<12} {'Questions':<12} {'% of Total':<12} {'Avg Score':<12}")
    print("-" * 80)
    
    for cid, count in summary['top_10_most_visited']:
        pct = summary['per_cluster_stats'][cid]['question_percentage']
        avg_score = summary['per_cluster_stats'][cid]['avg_majority_vote_score']
        print(f"{cid:<12} {count:<12} {pct:<12.2f} {avg_score:<12.3f}")
    print()
    
    print("-" * 80)
    print("TOP 10 HIGHEST SCORING CLUSTERS")
    print("-" * 80)
    print(f"{'Cluster ID':<12} {'Avg Score':<12} {'Questions':<12} {'% of Total':<12}")
    print("-" * 80)
    
    for cid, score in summary['top_10_highest_scores']:
        count = summary['per_cluster_stats'][cid]['total_questions']
        pct = summary['per_cluster_stats'][cid]['question_percentage']
        print(f"{cid:<12} {score:<12.3f} {count:<12} {pct:<12.2f}")
    print()
    
    # Distribution statistics
    all_counts = [summary['per_cluster_stats'][cid]['total_questions'] 
                  for cid in range(summary['num_clusters'])]
    print("-" * 80)
    print("CLUSTER VISIT DISTRIBUTION")
    print("-" * 80)
    print(f"Mean questions per cluster: {np.mean(all_counts):.2f}")
    print(f"Std dev: {np.std(all_counts):.2f}")
    print(f"Min: {np.min(all_counts)}")
    print(f"Max: {np.max(all_counts)}")
    print(f"Median: {np.median(all_counts):.2f}")
    print()
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze Rentropy cluster logs")
    parser.add_argument("--log_dir", required=True, help="Directory containing cluster log files")
    parser.add_argument("--output", help="Optional: Save summary to JSON file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"ERROR: Log directory not found: {args.log_dir}")
        return 1
    
    print(f"Loading logs from: {args.log_dir}")
    logs = load_logs(args.log_dir)
    print(f"Found {len(logs)} log files")
    print()
    
    if not logs:
        print("ERROR: No log files found")
        return 1
    
    summary = analyze_logs(logs)
    print_summary(summary)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
