#!/usr/bin/env python3
"""
exp0_0: Results Analysis Script
===============================
Computes aggregate metrics and generates visualizations from baseline results.

Usage:
    python analyze_results.py --results_dir results/
    python analyze_results.py --results_dir results/ --plots
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import numpy as np

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.")


# =============================================================================
# Data Loading
# =============================================================================

def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_results(results_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load all result files."""
    raw_path = os.path.join(results_dir, "raw_outputs.json")
    traces_path = os.path.join(results_dir, "iteration_traces.json")
    
    raw_outputs = load_json(raw_path) if os.path.exists(raw_path) else []
    traces = load_json(traces_path) if os.path.exists(traces_path) else []
    
    return raw_outputs, traces


# =============================================================================
# Metrics Computation
# =============================================================================

@dataclass
class AggregateMetrics:
    """Aggregate metrics from baseline experiment."""
    # Overall counts
    total_seeds: int
    total_iterations: int
    total_questions: int
    
    # Format/validity rates
    format_success_rate: float
    question_validity_rate: float
    solvability_rate: float
    
    # Score distributions
    mean_r_m: float
    std_r_m: float
    mean_novelty: float
    std_novelty: float
    
    # Iteration statistics
    mean_iteration_depth: float
    std_iteration_depth: float
    max_iteration_depth: int
    
    # Termination reasons
    termination_distribution: Dict[str, float]
    
    # Per-iteration breakdown
    per_iteration_r_m: List[float]
    per_iteration_novelty: List[float]


def compute_metrics(raw_outputs: List[Dict], traces: List[Dict]) -> AggregateMetrics:
    """Compute aggregate metrics from results."""
    
    # Initialize counters
    total_seeds = len(traces)
    total_iterations = 0
    total_questions = 0
    
    format_successes = 0
    validity_checks = []
    solvability_checks = []
    
    all_r_m = []
    all_novelty = []
    
    iteration_depths = []
    termination_counts = defaultdict(int)
    
    per_iter_r_m = defaultdict(list)
    per_iter_novelty = defaultdict(list)
    
    # Process traces
    for trace in traces:
        iteration_depths.append(trace['final_iteration'])
        termination_counts[trace['termination_reason']] += 1
        
        for iteration in trace['iterations']:
            total_iterations += 1
            iter_idx = iteration['iteration']
            
            # Record per-iteration metrics
            per_iter_r_m[iter_idx].append(iteration['avg_r_m'])
            per_iter_novelty[iter_idx].append(iteration['avg_novelty'])
            
            all_r_m.append(iteration['avg_r_m'])
            if iteration['avg_novelty'] > 0:
                all_novelty.append(iteration['avg_novelty'])
    
    # Process raw outputs for detailed metrics
    for seed_data in raw_outputs:
        for iteration in seed_data.get('iterations', []):
            for q in iteration.get('questions', []):
                total_questions += 1
                
                q_data = q.get('question_data', {})
                if q_data.get('valid_format', False):
                    format_successes += 1
                
                # Check verification results
                for ver in q.get('verification_results', []):
                    if ver.get('r2_question_correct', 0) > 0.5:
                        validity_checks.append(1)
                    else:
                        validity_checks.append(0)
                
                # Check solvability
                if q.get('r_m', 0) > 0.5:
                    solvability_checks.append(1)
                else:
                    solvability_checks.append(0)
    
    # Compute rates
    format_rate = format_successes / total_questions if total_questions > 0 else 0
    validity_rate = sum(validity_checks) / len(validity_checks) if validity_checks else 0
    solvability_rate = sum(solvability_checks) / len(solvability_checks) if solvability_checks else 0
    
    # Compute score statistics
    mean_r_m = np.mean(all_r_m) if all_r_m else 0
    std_r_m = np.std(all_r_m) if all_r_m else 0
    mean_novelty = np.mean(all_novelty) if all_novelty else 0
    std_novelty = np.std(all_novelty) if all_novelty else 0
    
    # Compute iteration statistics
    mean_depth = np.mean(iteration_depths) if iteration_depths else 0
    std_depth = np.std(iteration_depths) if iteration_depths else 0
    max_depth = max(iteration_depths) if iteration_depths else 0
    
    # Normalize termination distribution
    total_terms = sum(termination_counts.values())
    term_dist = {k: v / total_terms for k, v in termination_counts.items()} if total_terms > 0 else {}
    
    # Compute per-iteration averages
    max_iter = max(per_iter_r_m.keys()) + 1 if per_iter_r_m else 0
    per_iter_r_m_avg = [np.mean(per_iter_r_m[i]) if i in per_iter_r_m else 0 for i in range(max_iter)]
    per_iter_novelty_avg = [np.mean(per_iter_novelty[i]) if i in per_iter_novelty else 0 for i in range(max_iter)]
    
    return AggregateMetrics(
        total_seeds=total_seeds,
        total_iterations=total_iterations,
        total_questions=total_questions,
        format_success_rate=format_rate,
        question_validity_rate=validity_rate,
        solvability_rate=solvability_rate,
        mean_r_m=mean_r_m,
        std_r_m=std_r_m,
        mean_novelty=mean_novelty,
        std_novelty=std_novelty,
        mean_iteration_depth=mean_depth,
        std_iteration_depth=std_depth,
        max_iteration_depth=max_depth,
        termination_distribution=term_dist,
        per_iteration_r_m=per_iter_r_m_avg,
        per_iteration_novelty=per_iter_novelty_avg
    )


def compute_failure_analysis(traces: List[Dict]) -> Dict[str, Any]:
    """Analyze failure modes in detail."""
    
    analysis = {
        "by_termination_reason": defaultdict(lambda: {"count": 0, "avg_depth": 0, "depths": []}),
        "by_iteration": defaultdict(lambda: {"terminations": defaultdict(int), "total": 0}),
        "depth_histogram": defaultdict(int)
    }
    
    for trace in traces:
        reason = trace['termination_reason']
        depth = trace['final_iteration']
        
        analysis["by_termination_reason"][reason]["count"] += 1
        analysis["by_termination_reason"][reason]["depths"].append(depth)
        
        analysis["depth_histogram"][depth] += 1
        
        if trace['iterations']:
            last_iter = trace['iterations'][-1]
            iter_idx = last_iter['iteration']
            analysis["by_iteration"][iter_idx]["terminations"][reason] += 1
            analysis["by_iteration"][iter_idx]["total"] += 1
    
    # Compute averages
    for reason, data in analysis["by_termination_reason"].items():
        if data["depths"]:
            data["avg_depth"] = np.mean(data["depths"])
        del data["depths"]  # Remove raw data for cleaner output
    
    # Convert defaultdicts to regular dicts for JSON serialization
    analysis["by_termination_reason"] = dict(analysis["by_termination_reason"])
    analysis["by_iteration"] = {k: dict(v) for k, v in analysis["by_iteration"].items()}
    analysis["depth_histogram"] = dict(analysis["depth_histogram"])
    
    return analysis


# =============================================================================
# Visualization
# =============================================================================

def create_plots(metrics: AggregateMetrics, traces: List[Dict], output_dir: str):
    """Create visualization plots."""
    if not HAS_PLOTTING:
        print("Plotting not available. Skipping.")
        return
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # 1. Termination Reasons Pie Chart
    fig, ax = plt.subplots()
    if metrics.termination_distribution:
        labels = list(metrics.termination_distribution.keys())
        sizes = list(metrics.termination_distribution.values())
        colors = sns.color_palette("husl", len(labels))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title("Termination Reasons Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "termination_reasons.png"), dpi=150)
    plt.close()
    
    # 2. Iteration Depth Histogram
    depths = [t['final_iteration'] for t in traces]
    fig, ax = plt.subplots()
    ax.hist(depths, bins=range(1, max(depths) + 2), edgecolor='black', alpha=0.7)
    ax.set_xlabel("Iteration Depth")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Iteration Depths")
    ax.set_xticks(range(1, max(depths) + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "iteration_depth_histogram.png"), dpi=150)
    plt.close()
    
    # 3. Per-Iteration Metrics
    if metrics.per_iteration_r_m:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        iterations = range(1, len(metrics.per_iteration_r_m) + 1)
        
        ax1.bar(iterations, metrics.per_iteration_r_m, color='steelblue', alpha=0.7)
        ax1.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Average r_m (Solvability)")
        ax1.set_title("Solvability by Iteration")
        ax1.legend()
        
        ax2.bar(iterations, metrics.per_iteration_novelty, color='forestgreen', alpha=0.7)
        ax2.axhline(y=0.4, color='red', linestyle='--', label='Threshold')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Average Novelty Score")
        ax2.set_title("Novelty by Iteration")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "per_iteration_metrics.png"), dpi=150)
        plt.close()
    
    # 4. Score Distributions
    all_r_m = []
    all_novelty = []
    for trace in traces:
        for iteration in trace.get('iterations', []):
            all_r_m.append(iteration['avg_r_m'])
            if iteration['avg_novelty'] > 0:
                all_novelty.append(iteration['avg_novelty'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if all_r_m:
        ax1.hist(all_r_m, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        ax1.set_xlabel("r_m (Solvability Score)")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of Solvability Scores")
        ax1.legend()
    
    if all_novelty:
        ax2.hist(all_novelty, bins=20, edgecolor='black', alpha=0.7, color='forestgreen')
        ax2.axvline(x=0.4, color='red', linestyle='--', label='Threshold')
        ax2.set_xlabel("Novelty Score")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Novelty Scores")
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "score_distributions.png"), dpi=150)
    plt.close()
    
    print(f"Plots saved to {plots_dir}/")


# =============================================================================
# Question Progression Analysis
# =============================================================================

def print_question_progression(raw_outputs: List[Dict], output_dir: str = None, min_iterations: int = 3):
    """
    Print and save the sequence of questions for seeds that went beyond min_iterations.
    This helps visualize how questions evolve through the iteration chain.
    
    Args:
        raw_outputs: List of seed trace dictionaries
        output_dir: Directory to save the output file (if None, only prints)
        min_iterations: Minimum iterations threshold
    """
    lines = []
    
    def add_line(text=""):
        lines.append(text)
        print(text)
    
    add_line("\n" + "=" * 80)
    add_line(f"QUESTION PROGRESSION ANALYSIS (Seeds with > {min_iterations} iterations)")
    add_line("=" * 80)
    
    deep_seeds = []
    
    for seed_data in raw_outputs:
        num_iterations = len(seed_data.get('iterations', []))
        if num_iterations > min_iterations:
            deep_seeds.append(seed_data)
    
    if not deep_seeds:
        add_line(f"\nNo seeds found with more than {min_iterations} iterations.")
        add_line("=" * 80)
        
        # Save even empty report
        if output_dir:
            output_path = os.path.join(output_dir, "question_progression_report.txt")
            with open(output_path, 'w') as f:
                f.write("\n".join(lines))
            print(f"\nProgression report saved to {output_path}")
        return
    
    add_line(f"\nFound {len(deep_seeds)} seeds with > {min_iterations} iterations.\n")
    
    for seed_idx, seed_data in enumerate(deep_seeds):
        seed_question = seed_data.get('seed_question', 'N/A')
        seed_answer = seed_data.get('seed_answer', 'N/A')
        num_iterations = len(seed_data.get('iterations', []))
        termination = seed_data.get('termination_reason', 'unknown')
        
        add_line("-" * 80)
        add_line(f"SEED #{seed_data.get('seed_idx', seed_idx)} ({num_iterations} iterations, terminated: {termination})")
        add_line("-" * 80)
        
        # Print seed question (truncated)
        add_line(f"\n[SEED QUESTION]")
        add_line(f"  Q: {seed_question[:500]}{'...' if len(seed_question) > 500 else ''}")
        add_line(f"  A: {seed_answer}")
        
        # Print each iteration's questions
        for iteration in seed_data.get('iterations', []):
            iter_idx = iteration.get('iteration', 0)
            iter_termination = iteration.get('termination_reason', 'continue')
            
            add_line(f"\n  [ITERATION {iter_idx + 1}] (status: {iter_termination})")
            add_line(f"  " + "-" * 40)
            
            questions = iteration.get('questions', [])
            
            for q_idx, q in enumerate(questions):
                q_data = q.get('question_data', {})
                question_text = q_data.get('question', 'N/A')
                claimed_answer = q_data.get('claimed_answer', 'N/A')
                transformation = q_data.get('transformation', 'unknown')
                difficulty = q_data.get('difficulty_estimate', 'N/A')
                valid_format = q_data.get('valid_format', False)
                
                r_m = q.get('r_m', 0)
                novelty_result = q.get('novelty_result', {})
                r3_novelty = novelty_result.get('r3_novelty', 0) if novelty_result else 0
                
                # Format status indicator
                status = "[OK]" if valid_format else "[X]"
                
                add_line(f"\n    {status} Question {q_idx + 1}:")
                add_line(f"        Transformation: {str(transformation)[:100]}{'...' if len(str(transformation)) > 100 else ''}")
                add_line(f"        Q: {question_text[:300]}{'...' if len(str(question_text)) > 300 else ''}")
                add_line(f"        A: {claimed_answer}")
                add_line(f"        Difficulty: {difficulty}/10 | r_m: {r_m:.3f} | novelty: {r3_novelty:.3f}")
        
        add_line("\n")
    
    add_line("=" * 80)
    add_line("END OF QUESTION PROGRESSION ANALYSIS")
    add_line("=" * 80)
    
    # Save to file
    if output_dir:
        output_path = os.path.join(output_dir, "question_progression_report.txt")
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"\nProgression report saved to {output_path}")


def dump_full_analysis(raw_outputs: List[Dict], traces: List[Dict], metrics: AggregateMetrics, 
                       failure_analysis: Dict, output_dir: str):
    """
    Dump comprehensive analysis including all metrics, rewards, and progressions to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save detailed reward analysis
    reward_analysis = {
        "reward_types": {
            "r1_solution_correct": {
                "description": "Solution correctness score from V1 verifier",
                "range": "[0, 1]",
                "source": "Answer Verifier (V1)"
            },
            "r2_question_correct": {
                "description": "Question validity score (well-posed, solvable, determinate)",
                "range": "[0, 1]",
                "source": "Answer Verifier (V1)"
            },
            "r_m_majority_voting": {
                "description": "Fraction of M solutions agreeing on answer",
                "range": "[0, 1]",
                "source": "Aggregation over M solutions"
            },
            "r3_novelty": {
                "description": "Semantic distance from seed question",
                "range": "[0, 1]",
                "source": "Novelty Verifier (V2)",
                "rubric": {
                    "0.0-0.2": "Trivial/No meaningful change",
                    "0.2-0.4": "Minor variation (changed numbers)",
                    "0.4-0.6": "Moderate novelty (different approach)",
                    "0.6-0.8": "Significant novelty (new concepts)",
                    "0.8-1.0": "Exceptional novelty (creative extension)"
                }
            },
            "r3_difficulty": {
                "description": "Estimated difficulty of generated question",
                "range": "[0, 1]",
                "source": "Novelty Verifier (V2)"
            },
            "r3_transformation": {
                "description": "Quality of transformation applied",
                "range": "[0, 1]",
                "source": "Novelty Verifier (V2)"
            }
        },
        "aggregate_statistics": {
            "mean_r_m": metrics.mean_r_m,
            "std_r_m": metrics.std_r_m,
            "mean_novelty": metrics.mean_novelty,
            "std_novelty": metrics.std_novelty,
            "solvability_rate": metrics.solvability_rate,
            "question_validity_rate": metrics.question_validity_rate
        }
    }
    
    reward_path = os.path.join(output_dir, "reward_analysis.json")
    with open(reward_path, 'w') as f:
        json.dump(reward_analysis, f, indent=2)
    print(f"Reward analysis saved to {reward_path}")
    
    # 2. Save per-question detailed scores
    question_scores = []
    for seed_data in raw_outputs:
        seed_idx = seed_data.get('seed_idx', 0)
        for iteration in seed_data.get('iterations', []):
            iter_idx = iteration.get('iteration', 0)
            for q_idx, q in enumerate(iteration.get('questions', [])):
                q_data = q.get('question_data', {})
                novelty_result = q.get('novelty_result', {}) or {}
                
                # Extract all verification scores
                r1_scores = [v.get('r1_solution_correct', 0) for v in q.get('verification_results', [])]
                r2_scores = [v.get('r2_question_correct', 0) for v in q.get('verification_results', [])]
                
                question_scores.append({
                    "seed_idx": seed_idx,
                    "iteration": iter_idx,
                    "question_idx": q_idx,
                    "question": q_data.get('question', '')[:200],
                    "transformation": q_data.get('transformation', 'unknown'),
                    "valid_format": q_data.get('valid_format', False),
                    "scores": {
                        "r1_mean": np.mean(r1_scores) if r1_scores else 0,
                        "r2_mean": np.mean(r2_scores) if r2_scores else 0,
                        "r_m": q.get('r_m', 0),
                        "r3_novelty": novelty_result.get('r3_novelty', 0),
                        "r3_difficulty": novelty_result.get('r3_difficulty', 0),
                        "r3_transformation": novelty_result.get('r3_transformation', 0)
                    },
                    "solution_stats": {
                        "avg_length": q.get('avg_solution_length', 0),
                        "num_solutions": len(q.get('solutions', []))
                    }
                })
    
    scores_path = os.path.join(output_dir, "per_question_scores.json")
    with open(scores_path, 'w') as f:
        json.dump(question_scores, f, indent=2)
    print(f"Per-question scores saved to {scores_path}")
    
    # 3. Save iteration-level summary
    iteration_summary = []
    for trace in traces:
        for iteration in trace.get('iterations', []):
            iteration_summary.append({
                "seed_idx": trace.get('seed_idx', 0),
                "iteration": iteration.get('iteration', 0),
                "num_valid": iteration.get('num_valid', 0),
                "num_solvable": iteration.get('num_solvable', 0),
                "avg_r_m": iteration.get('avg_r_m', 0),
                "avg_novelty": iteration.get('avg_novelty', 0),
                "termination_reason": iteration.get('termination_reason', 'unknown')
            })
    
    iter_summary_path = os.path.join(output_dir, "iteration_summary.json")
    with open(iter_summary_path, 'w') as f:
        json.dump(iteration_summary, f, indent=2)
    print(f"Iteration summary saved to {iter_summary_path}")
    
    print(f"\nFull analysis dumped to {output_dir}/")


def save_question_progressions(raw_outputs: List[Dict], output_dir: str, min_iterations: int = 3):
    """
    Save the question progression data to a JSON file for further analysis.
    """
    progressions = []
    
    for seed_data in raw_outputs:
        num_iterations = len(seed_data.get('iterations', []))
        if num_iterations > min_iterations:
            progression = {
                "seed_idx": seed_data.get('seed_idx'),
                "seed_question": seed_data.get('seed_question'),
                "seed_answer": seed_data.get('seed_answer'),
                "num_iterations": num_iterations,
                "termination_reason": seed_data.get('termination_reason'),
                "question_chain": []
            }
            
            for iteration in seed_data.get('iterations', []):
                iter_questions = []
                for q in iteration.get('questions', []):
                    q_data = q.get('question_data', {})
                    novelty_result = q.get('novelty_result', {})
                    
                    iter_questions.append({
                        "question": q_data.get('question'),
                        "answer": q_data.get('claimed_answer'),
                        "transformation": q_data.get('transformation'),
                        "difficulty": q_data.get('difficulty_estimate'),
                        "valid_format": q_data.get('valid_format'),
                        "r_m": q.get('r_m'),
                        "r3_novelty": novelty_result.get('r3_novelty') if novelty_result else None
                    })
                
                progression["question_chain"].append({
                    "iteration": iteration.get('iteration'),
                    "termination_reason": iteration.get('termination_reason'),
                    "questions": iter_questions
                })
            
            progressions.append(progression)
    
    # Save to file
    output_path = os.path.join(output_dir, "question_progressions.json")
    with open(output_path, 'w') as f:
        json.dump(progressions, f, indent=2)
    
    print(f"Question progressions saved to {output_path}")
    return progressions


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(metrics: AggregateMetrics, failure_analysis: Dict) -> str:
    """Generate a text report of the results."""
    
    report = []
    report.append("=" * 60)
    report.append("exp0_0: BASELINE EXPERIMENT RESULTS")
    report.append("=" * 60)
    report.append("")
    
    report.append("## SUMMARY STATISTICS")
    report.append("-" * 40)
    report.append(f"Total Seeds Processed:    {metrics.total_seeds}")
    report.append(f"Total Iterations:         {metrics.total_iterations}")
    report.append(f"Total Questions Generated:{metrics.total_questions}")
    report.append("")
    
    report.append("## QUALITY METRICS")
    report.append("-" * 40)
    report.append(f"Format Success Rate:      {metrics.format_success_rate:.1%}")
    report.append(f"Question Validity Rate:   {metrics.question_validity_rate:.1%}")
    report.append(f"Solvability Rate:         {metrics.solvability_rate:.1%}")
    report.append("")
    
    report.append("## SCORE STATISTICS")
    report.append("-" * 40)
    report.append(f"Mean r_m (Solvability):   {metrics.mean_r_m:.3f} (+/- {metrics.std_r_m:.3f})")
    report.append(f"Mean Novelty Score:       {metrics.mean_novelty:.3f} (+/- {metrics.std_novelty:.3f})")
    report.append("")
    
    report.append("## ITERATION DEPTH")
    report.append("-" * 40)
    report.append(f"Mean Iteration Depth:     {metrics.mean_iteration_depth:.2f} (+/- {metrics.std_iteration_depth:.2f})")
    report.append(f"Max Iteration Depth:      {metrics.max_iteration_depth}")
    report.append("")
    
    report.append("## TERMINATION REASONS")
    report.append("-" * 40)
    for reason, pct in sorted(metrics.termination_distribution.items(), key=lambda x: -x[1]):
        report.append(f"  {reason:25s} {pct:.1%}")
    report.append("")
    
    report.append("## PER-ITERATION BREAKDOWN")
    report.append("-" * 40)
    for i, (r_m, nov) in enumerate(zip(metrics.per_iteration_r_m, metrics.per_iteration_novelty)):
        report.append(f"  Iteration {i+1}: r_m={r_m:.3f}, novelty={nov:.3f}")
    report.append("")
    
    report.append("## FAILURE ANALYSIS")
    report.append("-" * 40)
    for reason, data in failure_analysis.get("by_termination_reason", {}).items():
        report.append(f"  {reason}:")
        report.append(f"    Count: {data['count']}")
        report.append(f"    Avg Depth: {data['avg_depth']:.2f}")
    report.append("")
    
    report.append("=" * 60)
    report.append("END OF REPORT")
    report.append("=" * 60)
    
    return "\n".join(report)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="exp0_0: Analyze Baseline Results")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--plots", action="store_true", help="Generate visualization plots")
    parser.add_argument("--output", type=str, default=None, help="Output directory for analysis")
    parser.add_argument("--min_iterations", type=int, default=3, help="Min iterations for progression analysis")
    parser.add_argument("--show_progressions", action="store_true", help="Show question progressions for deep seeds")
    args = parser.parse_args()
    
    output_dir = args.output or args.results_dir
    
    print(f"Loading results from {args.results_dir}...")
    raw_outputs, traces = load_results(args.results_dir)
    
    if not traces:
        print("Error: No results found. Run baseline_pipeline.py first.")
        return
    
    print(f"Found {len(traces)} seed traces.")
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(raw_outputs, traces)
    
    # Compute failure analysis
    print("Analyzing failures...")
    failure_analysis = compute_failure_analysis(traces)
    
    # Generate report
    report = generate_report(metrics, failure_analysis)
    print("\n" + report)
    
    # Save metrics summary
    summary = {
        "total_seeds": metrics.total_seeds,
        "total_iterations": metrics.total_iterations,
        "total_questions": metrics.total_questions,
        "format_success_rate": metrics.format_success_rate,
        "question_validity_rate": metrics.question_validity_rate,
        "solvability_rate": metrics.solvability_rate,
        "mean_r_m": metrics.mean_r_m,
        "std_r_m": metrics.std_r_m,
        "mean_novelty": metrics.mean_novelty,
        "std_novelty": metrics.std_novelty,
        "mean_iteration_depth": metrics.mean_iteration_depth,
        "std_iteration_depth": metrics.std_iteration_depth,
        "max_iteration_depth": metrics.max_iteration_depth,
        "termination_distribution": metrics.termination_distribution,
        "per_iteration_r_m": metrics.per_iteration_r_m,
        "per_iteration_novelty": metrics.per_iteration_novelty
    }
    
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {summary_path}")
    
    # Save failure analysis
    failure_path = os.path.join(output_dir, "failure_analysis.json")
    with open(failure_path, 'w') as f:
        json.dump(failure_analysis, f, indent=2)
    print(f"Failure analysis saved to {failure_path}")
    
    # Save report
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    
    # Dump full analysis (all rewards, per-question scores, iteration summary)
    print("\nDumping full analysis...")
    dump_full_analysis(raw_outputs, traces, metrics, failure_analysis, output_dir)
    
    # Generate plots
    if args.plots:
        print("\nGenerating plots...")
        create_plots(metrics, traces, output_dir)
    
    # Question progression analysis for deep seeds
    if raw_outputs:
        # Always save progressions to JSON file
        save_question_progressions(raw_outputs, output_dir, min_iterations=args.min_iterations)
        
        # Print progressions if requested or if any deep seeds exist
        if args.show_progressions:
            # Print to console AND save to text file
            print_question_progression(raw_outputs, output_dir=output_dir, min_iterations=args.min_iterations)
        else:
            # Check if there are any deep seeds and notify user
            deep_count = sum(1 for s in raw_outputs if len(s.get('iterations', [])) > args.min_iterations)
            if deep_count > 0:
                print(f"\nFound {deep_count} seeds with > {args.min_iterations} iterations.")
                print(f"Use --show_progressions to see the question evolution chain.")
                print(f"Or check: {output_dir}/question_progressions.json")
                
                # Still save the text report even without --show_progressions
                print("\nGenerating progression report...")
                # Temporarily suppress print output
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                print_question_progression(raw_outputs, output_dir=output_dir, min_iterations=args.min_iterations)
                sys.stdout = old_stdout
                print(f"Progression report saved to {output_dir}/question_progression_report.txt")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
