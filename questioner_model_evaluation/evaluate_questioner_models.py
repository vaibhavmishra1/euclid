#!/usr/bin/env python3
"""
Evaluate two question generation models by:
1. Generating 10,000 questions from each model
2. Assigning each question to clusters using Qwen 0.6B embeddings
3. Computing and printing cluster frequency distribution statistics
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
sys.path.insert(0, str(Path(__file__).parent.parent / "rentropy" / "R-Zero-main"))

from cluster_assigner import ClusterAssigner


def load_generated_questions(storage_path: str, save_name: str, num_gpus: int = 8) -> List[Dict]:
    """
    Load all generated questions from multiple GPU output files.
    
    Args:
        storage_path: Base storage path (from STORAGE_PATH env var)
        save_name: Save name prefix used during generation
        num_gpus: Number of GPUs used (default 8)
    
    Returns:
        List of question dictionaries with 'question', 'answer', 'score' fields
    """
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


def generate_questions_for_model(
    model_path: str,
    save_name: str,
    num_samples: int = 10000,
    num_gpus: int = 8,
    storage_path: str = None
):
    """
    Generate questions using the question_generate.py script.
    
    Args:
        model_path: Path to the model
        save_name: Name to save results under
        num_samples: Total number of samples to generate (will be split across GPUs)
        num_gpus: Number of GPUs to use
        storage_path: Storage path (if None, uses STORAGE_PATH env var)
    """
    import subprocess
    
    if storage_path is None:
        storage_path = os.getenv("STORAGE_PATH")
        if storage_path is None:
            raise ValueError("STORAGE_PATH environment variable not set")
    
    # Create output directory if it doesn't exist
    output_dir = Path(storage_path) / "generated_question"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Calculate samples per GPU (round up to ensure we get at least num_samples total)
    samples_per_gpu = (num_samples + num_gpus - 1) // num_gpus
    
    print(f"\n{'='*80}")
    print(f"Generating {num_samples} questions using model: {model_path}")
    print(f"Using {num_gpus} GPUs, {samples_per_gpu} samples per GPU")
    print(f"Results will be saved with prefix: {save_name}")
    print(f"{'='*80}\n")
    
    # Import the question generation script
    script_path = Path(__file__).parent.parent / "rentropy" / "R-Zero-main" / "question_generate" / "question_generate.py"
    
    # Set environment variables
    env = os.environ.copy()
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    env["STORAGE_PATH"] = storage_path
    pythonpath = str(Path(__file__).parent.parent / "rentropy" / "R-Zero-main")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{pythonpath}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = pythonpath
    
    # Run generation on each GPU
    processes = []
    for gpu_id in range(num_gpus):
        cmd = [
            sys.executable,
            str(script_path),
            "--model", model_path,
            "--suffix", str(gpu_id),
            "--num_samples", str(samples_per_gpu),
            "--save_name", save_name
        ]
        
        gpu_env = env.copy()
        gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        print(f"Starting GPU {gpu_id}...")
        proc = subprocess.Popen(cmd, env=gpu_env)
        processes.append(proc)
    
    # Wait for all processes to complete
    print("\nWaiting for all GPU processes to complete...")
    for i, proc in enumerate(processes):
        proc.wait()
        if proc.returncode != 0:
            print(f"Warning: GPU {i} process exited with code {proc.returncode}")
        else:
            print(f"GPU {i} completed successfully")
    
    print("\nQuestion generation completed!")


def assign_clusters_to_questions(
    questions: List[Dict],
    centroids_path: str,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    use_local_files: bool = False
) -> np.ndarray:
    """
    Assign cluster IDs to questions using ClusterAssigner.
    
    Args:
        questions: List of question dictionaries
        centroids_path: Path to centroids.npy file
        embedding_model: Embedding model to use
        use_local_files: If True, only use locally cached models (no downloads)
    
    Returns:
        Array of cluster IDs (one per question)
    """
    import time
    
    print(f"\n{'='*80}")
    print(f"Assigning clusters to {len(questions)} questions")
    print(f"Using embedding model: {embedding_model}")
    print(f"Centroids path: {centroids_path}")
    if use_local_files:
        print("Using local files only (no downloads)")
    print(f"{'='*80}\n")
    
    # Extract question texts
    question_texts = [q["question"] for q in questions]
    
    # Initialize cluster assigner with retry logic
    max_retries = 3
    retry_delay = 5  # seconds
    
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
    try:
        for attempt in range(max_retries):
            try:
                # Initialize cluster assigner
                assigner = ClusterAssigner(
                    centroids_path=centroids_path,
                    embedding_model=embedding_model
                )
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\nWarning: Failed to load embedding model (attempt {attempt + 1}/{max_retries})")
                    print(f"Error: {str(e)[:200]}...")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
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
        # Restore original environment variables
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
    
    # Assign clusters
    cluster_ids = assigner.assign_clusters(question_texts)
    
    # Assign clusters
    cluster_ids = assigner.assign_clusters(question_texts)
    
    print(f"Assigned {len(cluster_ids)} questions to clusters")
    print(f"Cluster ID range: {cluster_ids.min()} to {cluster_ids.max()}")
    
    return cluster_ids


def compute_cluster_statistics(cluster_ids: np.ndarray, model_name: str) -> Dict:
    """
    Compute detailed statistics about cluster distribution.
    
    Args:
        cluster_ids: Array of cluster IDs
        model_name: Name of the model (for logging)
    
    Returns:
        Dictionary with statistics
    """
    # Count occurrences of each cluster
    cluster_counts = Counter(cluster_ids)
    
    # Convert to numpy array for easier computation
    counts_array = np.array(list(cluster_counts.values()))
    
    # Basic statistics
    total_questions = len(cluster_ids)
    num_unique_clusters = len(cluster_counts)
    num_total_clusters = cluster_ids.max() + 1  # Assuming clusters are 0-indexed
    
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
    
    # Percentiles
    if len(counts_array) > 0:
        stats["p25_cluster_count"] = float(np.percentile(counts_array, 25))
        stats["p75_cluster_count"] = float(np.percentile(counts_array, 75))
        stats["p90_cluster_count"] = float(np.percentile(counts_array, 90))
        stats["p95_cluster_count"] = float(np.percentile(counts_array, 95))
        stats["p99_cluster_count"] = float(np.percentile(counts_array, 99))
    
    # Entropy (diversity measure)
    probabilities = counts_array / total_questions
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    stats["entropy"] = float(entropy)
    stats["max_entropy"] = float(np.log(num_unique_clusters)) if num_unique_clusters > 0 else 0
    stats["normalized_entropy"] = float(entropy / stats["max_entropy"]) if stats["max_entropy"] > 0 else 0
    
    # Top clusters
    top_clusters = cluster_counts.most_common(10)
    stats["top_10_clusters"] = [{"cluster_id": int(cid), "count": int(count)} for cid, count in top_clusters]
    
    # Bottom clusters (least frequent)
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
        description="Evaluate question generation models using cluster analysis"
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to first trained model"
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Path to second trained model"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of questions to generate per model (default: 10000)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for generation (default: 8)"
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
        "--skip_generation",
        action="store_true",
        help="Skip question generation and only analyze existing results"
    )
    parser.add_argument(
        "--save_name1",
        type=str,
        default="model1_eval",
        help="Save name prefix for model 1 results (default: model1_eval)"
    )
    parser.add_argument(
        "--save_name2",
        type=str,
        default="model2_eval",
        help="Save name prefix for model 2 results (default: model2_eval)"
    )
    parser.add_argument(
        "--use_local_files",
        action="store_true",
        help="Use only locally cached models (no downloads from Hugging Face)"
    )
    
    args = parser.parse_args()
    
    # Get storage path
    storage_path = args.storage_path or os.getenv("STORAGE_PATH")
    if storage_path is None:
        raise ValueError("STORAGE_PATH must be provided via --storage_path or environment variable")
    
    # Step 1: Generate questions for both models
    if not args.skip_generation:
        print("\n" + "="*80)
        print("STEP 1: GENERATING QUESTIONS")
        print("="*80)
        
        print("\nGenerating questions for Model 1...")
        generate_questions_for_model(
            model_path=args.model1,
            save_name=args.save_name1,
            num_samples=args.num_samples,
            num_gpus=args.num_gpus,
            storage_path=storage_path
        )
        
        print("\nGenerating questions for Model 2...")
        generate_questions_for_model(
            model_path=args.model2,
            save_name=args.save_name2,
            num_samples=args.num_samples,
            num_gpus=args.num_gpus,
            storage_path=storage_path
        )
    else:
        print("Skipping question generation (--skip_generation flag set)")
    
    # Step 2: Load generated questions
    print("\n" + "="*80)
    print("STEP 2: LOADING GENERATED QUESTIONS")
    print("="*80)
    
    questions1 = load_generated_questions(storage_path, args.save_name1, args.num_gpus)
    questions2 = load_generated_questions(storage_path, args.save_name2, args.num_gpus)
    
    if len(questions1) == 0:
        raise ValueError(f"No questions found for model 1 (save_name: {args.save_name1})")
    if len(questions2) == 0:
        raise ValueError(f"No questions found for model 2 (save_name: {args.save_name2})")
    
    # Step 3: Assign clusters
    print("\n" + "="*80)
    print("STEP 3: ASSIGNING CLUSTERS")
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
    
    # Step 4: Compute and print statistics
    print("\n" + "="*80)
    print("STEP 4: COMPUTING STATISTICS")
    print("="*80)
    
    stats1 = compute_cluster_statistics(cluster_ids1, "Model 1")
    stats2 = compute_cluster_statistics(cluster_ids2, "Model 2")
    
    print_statistics(stats1)
    print_statistics(stats2)
    
    # Save statistics to JSON
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "cluster_statistics.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model1_stats": stats1,
            "model2_stats": stats2,
            "model1_path": args.model1,
            "model2_path": args.model2,
        }, f, indent=2)
    
    print(f"\nStatistics saved to: {output_file}")
    
    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nModel 1 Coverage: {stats1['cluster_coverage']:.2%}")
    print(f"Model 2 Coverage: {stats2['cluster_coverage']:.2%}")
    print(f"\nModel 1 Entropy: {stats1['entropy']:.4f} (normalized: {stats1['normalized_entropy']:.4f})")
    print(f"Model 2 Entropy: {stats2['entropy']:.4f} (normalized: {stats2['normalized_entropy']:.4f})")
    print(f"\nModel 1 Mean Cluster Count: {stats1['mean_cluster_count']:.2f}")
    print(f"Model 2 Mean Cluster Count: {stats2['mean_cluster_count']:.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
