#!/usr/bin/env python3
"""
Standalone script to analyze cluster distributions for already-generated questions.
Use this if you've generated questions separately (e.g., using question_generate.bash).
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import Counter
from typing import List, Dict
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "rentropy" / "cluster_space"))

from cluster_assigner import ClusterAssigner


def load_generated_questions(storage_path: str, save_name: str, num_gpus: int = 8) -> List[Dict]:
    """Load all generated questions from multiple GPU output files."""
    questions = []
    base_path = Path(storage_path) / "generated_question"
    
    for suffix in range(num_gpus):
        file_path = base_path / f"{save_name}_{suffix}.json"
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                questions.extend(data)
                print(f"Loaded {len(data)} questions from {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Total questions loaded: {len(questions)}")
    return questions


def assign_clusters_to_questions(
    questions: List[Dict],
    centroids_path: str,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    use_local_files: bool = False
) -> np.ndarray:
    """Assign cluster IDs to questions using ClusterAssigner."""
    import time
    
    print(f"\nAssigning clusters to {len(questions)} questions...")
    if use_local_files:
        print("Using local files only (no downloads)")
    
    question_texts = [q["question"] for q in questions]
    
    # Set environment variables before importing (if using local files)
    original_transformers_offline = None
    original_hf_hub_offline = None
    if use_local_files:
        import os
        original_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE", None)
        original_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE", None)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    
    assigner = None
    max_retries = 3
    retry_delay = 5
    
    try:
        for attempt in range(max_retries):
            try:
                assigner = ClusterAssigner(
                    centroids_path=centroids_path,
                    embedding_model=embedding_model
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\nWarning: Failed to load embedding model (attempt {attempt + 1}/{max_retries})")
                    print(f"Error: {str(e)[:200]}...")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"\nError: Failed to load embedding model after {max_retries} attempts")
                    print(f"Last error: {e}")
                    print("\nSuggestions:")
                    print("1. Check your internet connection")
                    print("2. Try again later (Hugging Face servers may be busy)")
                    print("3. Pre-download the model: python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"Qwen/Qwen3-Embedding-0.6B\")'")
                    print("4. Use --use_local_files flag if model is already cached")
                    raise
    finally:
        if use_local_files:
            import os
            if original_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = original_transformers_offline
            if original_hf_hub_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_hf_hub_offline
    
    cluster_ids = assigner.assign_clusters(question_texts)
    
    print(f"Assigned {len(cluster_ids)} questions to clusters")
    print(f"Cluster ID range: {cluster_ids.min()} to {cluster_ids.max()}")
    
    return cluster_ids


def save_questions_with_clusters(
    questions: List[Dict],
    cluster_ids: np.ndarray,
    storage_path: str,
    save_name: str
) -> None:
    """
    Save questions with their assigned cluster IDs to a single JSON file.
    
    Args:
        questions: List of question dictionaries
        cluster_ids: Array of cluster IDs (one per question)
        storage_path: Base storage path
        save_name: Save name prefix for output file
    """
    output_dir = Path(storage_path) / "generated_question_with_clusters"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add cluster IDs to questions
    questions_with_clusters = []
    for i, (question, cluster_id) in enumerate(zip(questions, cluster_ids)):
        question_copy = question.copy()
        question_copy["cluster_id"] = int(cluster_id)
        questions_with_clusters.append(question_copy)
    
    # Save to single file
    output_file = output_dir / f"{save_name}_with_clusters.json"
    with open(output_file, 'w') as f:
        json.dump(questions_with_clusters, f, indent=2)
    
    print(f"Saved {len(questions_with_clusters)} questions with cluster IDs to: {output_file}")


def compute_cluster_statistics(cluster_ids: np.ndarray, model_name: str) -> Dict:
    """Compute detailed statistics about cluster distribution."""
    cluster_counts = Counter(cluster_ids)
    counts_array = np.array(list(cluster_counts.values()))
    
    total_questions = len(cluster_ids)
    num_unique_clusters = len(cluster_counts)
    num_total_clusters = cluster_ids.max() + 1
    
    stats = {
        "model_name": model_name,
        "total_questions": total_questions,
        "num_unique_clusters": num_unique_clusters,
        "num_total_clusters": num_total_clusters,
        "cluster_coverage": num_unique_clusters / num_total_clusters if num_total_clusters > 0 else 0,
        "min_cluster_count": int(counts_array.min()) if len(counts_array) > 0 else 0,
        "max_cluster_count": int(counts_array.max()) if len(counts_array) > 0 else 0,
        "mean_cluster_count": float(counts_array.mean()) if len(counts_array) > 0 else 0,
        "median_cluster_count": float(np.median(counts_array)) if len(counts_array) > 0 else 0,
        "std_cluster_count": float(counts_array.std()) if len(counts_array) > 0 else 0,
        "cluster_counts": dict(cluster_counts),
    }
    
    if len(counts_array) > 0:
        stats["p25_cluster_count"] = float(np.percentile(counts_array, 25))
        stats["p75_cluster_count"] = float(np.percentile(counts_array, 75))
        stats["p90_cluster_count"] = float(np.percentile(counts_array, 90))
        stats["p95_cluster_count"] = float(np.percentile(counts_array, 95))
        stats["p99_cluster_count"] = float(np.percentile(counts_array, 99))
    
    probabilities = counts_array / total_questions
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    stats["entropy"] = float(entropy)
    stats["max_entropy"] = float(np.log(num_unique_clusters)) if num_unique_clusters > 0 else 0
    stats["normalized_entropy"] = float(entropy / stats["max_entropy"]) if stats["max_entropy"] > 0 else 0
    
    top_clusters = cluster_counts.most_common(10)
    stats["top_10_clusters"] = [{"cluster_id": int(cid), "count": int(count)} for cid, count in top_clusters]
    
    bottom_clusters = cluster_counts.most_common()[-10:]
    stats["bottom_10_clusters"] = [{"cluster_id": int(cid), "count": int(count)} for cid, count in bottom_clusters]
    
    return stats


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print(f"\n{'='*80}")
    print(f"CLUSTER DISTRIBUTION STATISTICS: {stats['model_name']}")
    print(f"{'='*80}")
    print(f"\nBasic Statistics:")
    print(f"  Total Questions: {stats['total_questions']:,}")
    print(f"  Unique Clusters Used: {stats['num_unique_clusters']:,} / {stats['num_total_clusters']:,}")
    print(f"  Cluster Coverage: {stats['cluster_coverage']:.2%}")
    
    print(f"\nCluster Count Distribution:")
    print(f"  Minimum: {stats['min_cluster_count']}")
    print(f"  Maximum: {stats['max_cluster_count']}")
    print(f"  Mean: {stats['mean_cluster_count']:.2f}")
    print(f"  Median: {stats['median_cluster_count']:.2f}")
    print(f"  Std Dev: {stats['std_cluster_count']:.2f}")
    
    if "p25_cluster_count" in stats:
        print(f"\nPercentiles:")
        print(f"  25th: {stats['p25_cluster_count']:.2f}")
        print(f"  75th: {stats['p75_cluster_count']:.2f}")
        print(f"  90th: {stats['p90_cluster_count']:.2f}")
        print(f"  95th: {stats['p95_cluster_count']:.2f}")
        print(f"  99th: {stats['p99_cluster_count']:.2f}")
    
    print(f"\nDiversity Metrics:")
    print(f"  Entropy: {stats['entropy']:.4f}")
    print(f"  Max Entropy: {stats['max_entropy']:.4f}")
    print(f"  Normalized Entropy: {stats['normalized_entropy']:.4f}")
    
    print(f"\nTop 10 Most Frequent Clusters:")
    for item in stats['top_10_clusters']:
        print(f"  Cluster {item['cluster_id']:4d}: {item['count']:5d} questions ({100*item['count']/stats['total_questions']:.2f}%)")
    
    print(f"\nBottom 10 Least Frequent Clusters:")
    for item in stats['bottom_10_clusters']:
        print(f"  Cluster {item['cluster_id']:4d}: {item['count']:5d} questions ({100*item['count']/stats['total_questions']:.2f}%)")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cluster distributions for generated questions"
    )
    parser.add_argument(
        "--save_name1",
        type=str,
        required=True,
        help="Save name prefix for model 1 results"
    )
    parser.add_argument(
        "--save_name2",
        type=str,
        required=True,
        help="Save name prefix for model 2 results"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs used for generation (default: 8)"
    )
    parser.add_argument(
        "--centroids_path",
        type=str,
        default="/workspace/euclid/rentropy/cluster_space/cluster_data/centroids.npy",
        help="Path to centroids.npy file"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model to use (default: Qwen/Qwen3-Embedding-0.6B)"
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default=None,
        help="Storage path for generated questions (default: uses STORAGE_PATH env var)"
    )
    parser.add_argument(
        "--model1_name",
        type=str,
        default="Model 1",
        help="Display name for model 1"
    )
    parser.add_argument(
        "--model2_name",
        type=str,
        default="Model 2",
        help="Display name for model 2"
    )
    parser.add_argument(
        "--use_local_files",
        action="store_true",
        help="Use only locally cached models (no downloads from Hugging Face)"
    )
    
    args = parser.parse_args()
    
    storage_path = args.storage_path or os.getenv("STORAGE_PATH")
    if storage_path is None:
        raise ValueError("STORAGE_PATH must be provided via --storage_path or environment variable")
    
    # Load questions
    print("Loading questions for Model 1...")
    questions1 = load_generated_questions(storage_path, args.save_name1, args.num_gpus)
    
    print("\nLoading questions for Model 2...")
    questions2 = load_generated_questions(storage_path, args.save_name2, args.num_gpus)
    
    if len(questions1) == 0:
        raise ValueError(f"No questions found for model 1 (save_name: {args.save_name1})")
    if len(questions2) == 0:
        raise ValueError(f"No questions found for model 2 (save_name: {args.save_name2})")
    
    # Assign clusters
    print("\n" + "="*80)
    print("ASSIGNING CLUSTERS")
    print("="*80)
    
    cluster_ids1 = assign_clusters_to_questions(
        questions1,
        centroids_path=args.centroids_path,
        embedding_model=args.embedding_model,
        use_local_files=args.use_local_files
    )
    
    cluster_ids2 = assign_clusters_to_questions(
        questions2,
        centroids_path=args.centroids_path,
        embedding_model=args.embedding_model,
        use_local_files=args.use_local_files
    )
    
    # Save questions with cluster IDs
    print("\nSaving questions with cluster IDs...")
    save_questions_with_clusters(questions1, cluster_ids1, storage_path, args.save_name1)
    save_questions_with_clusters(questions2, cluster_ids2, storage_path, args.save_name2)
    
    # Compute statistics
    print("\n" + "="*80)
    print("COMPUTING STATISTICS")
    print("="*80)
    
    stats1 = compute_cluster_statistics(cluster_ids1, args.model1_name)
    stats2 = compute_cluster_statistics(cluster_ids2, args.model2_name)
    
    print_statistics(stats1)
    print_statistics(stats2)
    
    # Save statistics
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "cluster_statistics.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model1_stats": stats1,
            "model2_stats": stats2,
        }, f, indent=2)
    
    print(f"\nStatistics saved to: {output_file}")
    
    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{args.model1_name} Coverage: {stats1['cluster_coverage']:.2%}")
    print(f"{args.model2_name} Coverage: {stats2['cluster_coverage']:.2%}")
    print(f"\n{args.model1_name} Entropy: {stats1['entropy']:.4f} (normalized: {stats1['normalized_entropy']:.4f})")
    print(f"{args.model2_name} Entropy: {stats2['entropy']:.4f} (normalized: {stats2['normalized_entropy']:.4f})")
    print(f"\n{args.model1_name} Mean Cluster Count: {stats1['mean_cluster_count']:.2f}")
    print(f"{args.model2_name} Mean Cluster Count: {stats2['mean_cluster_count']:.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
