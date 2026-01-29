#!/usr/bin/env python3
"""
Build cluster space from a corpus of math questions.
This is run ONCE offline before training.

Usage:
    python build_clusters.py --corpus_file questions.json --output_dir ./cluster_data --num_clusters 128
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    raise

from sklearn.cluster import KMeans


def load_questions_from_file(filepath: str) -> List[str]:
    """Load questions from a JSON file. Supports multiple formats."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    questions = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict):
                # Try common keys
                for key in ['question', 'problem', 'text', 'input']:
                    if key in item and item[key]:
                        questions.append(item[key])
                        break
    elif isinstance(data, dict):
        # If it's a dict with a questions key
        if 'questions' in data:
            questions = data['questions']
    
    return [q for q in questions if q and len(q.strip()) > 10]


def load_questions_from_dir(corpus_dir: str) -> List[str]:
    """Load questions from all JSON files in a directory."""
    questions = []
    corpus_path = Path(corpus_dir)
    
    for json_file in corpus_path.glob("*.json"):
        print(f"Loading from {json_file}...")
        questions.extend(load_questions_from_file(str(json_file)))
    
    return questions


def embed_questions(questions: List[str], model_name: str, batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """Embed questions using a sentence transformer model."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    print(f"Embedding {len(questions)} questions...")
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    
    return embeddings


def fit_kmeans(embeddings: np.ndarray, num_clusters: int, n_init: int = 10, max_iter: int = 300, random_state: int = 42) -> KMeans:
    """Fit K-Means clustering on embeddings."""
    print(f"Fitting K-Means with {num_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
    )
    kmeans.fit(embeddings)
    print(f"K-Means inertia: {kmeans.inertia_:.4f}")
    return kmeans


def save_cluster_data(output_dir: str, centroids: np.ndarray, labels: np.ndarray, questions: List[str]):
    """Save cluster centroids and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save centroids
    centroids_path = os.path.join(output_dir, "centroids.npy")
    np.save(centroids_path, centroids)
    print(f"Saved centroids to {centroids_path}")
    
    # Save labels for analysis
    labels_path = os.path.join(output_dir, "labels.npy")
    np.save(labels_path, labels)
    
    # Save cluster stats
    unique, counts = np.unique(labels, return_counts=True)
    stats = {
        "num_clusters": len(unique),
        "num_questions": len(questions),
        "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
        "embedding_dim": centroids.shape[1],
    }
    stats_path = os.path.join(output_dir, "cluster_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build cluster space for Rentropy")
    parser.add_argument("--corpus_file", type=str, help="Single JSON file with questions")
    parser.add_argument("--corpus_dir", type=str, help="Directory with JSON files")
    parser.add_argument("--output_dir", type=str, default="./cluster_data")
    parser.add_argument("--num_clusters", type=int, default=128)
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true", default=True)
    args = parser.parse_args()
    
    # Load questions
    if args.corpus_file:
        questions = load_questions_from_file(args.corpus_file)
    elif args.corpus_dir:
        questions = load_questions_from_dir(args.corpus_dir)
    else:
        raise ValueError("Must provide either --corpus_file or --corpus_dir")
    
    print(f"Loaded {len(questions)} questions")
    
    if len(questions) < args.num_clusters:
        print(f"Warning: fewer questions ({len(questions)}) than clusters ({args.num_clusters})")
        args.num_clusters = max(10, len(questions) // 5)
        print(f"Reducing to {args.num_clusters} clusters")
    
    # Embed
    embeddings = embed_questions(questions, args.embedding_model, args.batch_size, args.normalize)
    
    # Cluster
    kmeans = fit_kmeans(embeddings, args.num_clusters)
    
    # Save
    save_cluster_data(args.output_dir, kmeans.cluster_centers_, kmeans.labels_, questions)
    
    print("Done!")


if __name__ == "__main__":
    main()
