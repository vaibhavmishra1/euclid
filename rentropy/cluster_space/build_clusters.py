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
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    raise

from sklearn.cluster import KMeans


def generate_question_id(question: str, index: int) -> str:
    """Generate a unique ID for a question based on content hash and index."""
    # Use hash of question + index for uniqueness
    content = f"{question}_{index}".encode('utf-8')
    hash_obj = hashlib.md5(content)
    return f"q_{hash_obj.hexdigest()[:12]}_{index}"


def load_questions_from_file(filepath: str, dataset_source: str = None) -> List[Tuple[str, str]]:
    """
    Load questions from a JSON file. Supports multiple formats.
    
    Returns:
        List of (question_text, dataset_source) tuples
    """
    if dataset_source is None:
        dataset_source = Path(filepath).stem  # Use filename as source
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    questions = []
    if isinstance(data, list):
        for item in data:
            question_text = None
            if isinstance(item, str):
                question_text = item
            elif isinstance(item, dict):
                # Try common keys
                for key in ['question', 'problem', 'text', 'input']:
                    if key in item and item[key]:
                        question_text = item[key]
                        break
            
            if question_text and len(question_text.strip()) > 10:
                questions.append((question_text, dataset_source))
    elif isinstance(data, dict):
        # If it's a dict with a questions key
        if 'questions' in data:
            for q in data['questions']:
                if q and len(q.strip()) > 10:
                    questions.append((q, dataset_source))
    
    return questions


def load_questions_from_dir(corpus_dir: str) -> List[Tuple[str, str]]:
    """
    Load questions from all JSON files in a directory.
    
    Returns:
        List of (question_text, dataset_source) tuples
    """
    questions = []
    corpus_path = Path(corpus_dir)
    
    for json_file in corpus_path.glob("*.json"):
        dataset_source = json_file.stem  # Use filename as source
        print(f"Loading from {json_file} (source: {dataset_source})...")
        questions.extend(load_questions_from_file(str(json_file), dataset_source))
    
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


def save_cluster_data(
    output_dir: str,
    centroids: np.ndarray,
    labels: np.ndarray,
    questions: List[Tuple[str, str]],
    embeddings: np.ndarray,
):
    """
    Save cluster centroids, labels, embeddings, and question metadata.
    
    Args:
        output_dir: Directory to save files
        centroids: Cluster centroids array (num_clusters, embedding_dim)
        labels: Cluster assignments array (num_questions,)
        questions: List of (question_text, dataset_source) tuples
        embeddings: Question embeddings array (num_questions, embedding_dim)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_questions = len(questions)
    assert len(labels) == num_questions, f"Mismatch: {len(labels)} labels vs {num_questions} questions"
    assert len(embeddings) == num_questions, f"Mismatch: {len(embeddings)} embeddings vs {num_questions} questions"
    
    # Save centroids
    centroids_path = os.path.join(output_dir, "centroids.npy")
    np.save(centroids_path, centroids)
    print(f"Saved centroids to {centroids_path} (shape: {centroids.shape})")
    
    # Save labels (cluster assignments)
    labels_path = os.path.join(output_dir, "labels.npy")
    np.save(labels_path, labels)
    print(f"Saved labels to {labels_path} (shape: {labels.shape})")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path} (shape: {embeddings.shape})")
    
    # Generate question IDs and create metadata
    question_metadata = []
    for index, (question_text, dataset_source) in enumerate(questions):
        question_id = generate_question_id(question_text, index)
        question_metadata.append({
            "question_id": question_id,
            "index": index,  # Index in labels.npy and embeddings.npy
            "question": question_text,
            "cluster_id": int(labels[index]),
            "dataset_source": dataset_source,
        })
    
    # Save question metadata (mapping from question_id to all info)
    metadata_path = os.path.join(output_dir, "question_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(question_metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved question metadata to {metadata_path} ({len(question_metadata)} questions)")
    
    # Also save a reverse mapping: index -> question_id (for quick lookup)
    index_to_id = {item["index"]: item["question_id"] for item in question_metadata}
    index_mapping_path = os.path.join(output_dir, "index_to_question_id.json")
    with open(index_mapping_path, 'w') as f:
        json.dump(index_to_id, f, indent=2)
    print(f"Saved index mapping to {index_mapping_path}")
    
    # Save cluster stats
    unique, counts = np.unique(labels, return_counts=True)
    stats = {
        "num_clusters": len(unique),
        "num_questions": num_questions,
        "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
        "embedding_dim": centroids.shape[1],
        "dataset_sources": list(set([q[1] for q in questions])),
    }
    stats_path = os.path.join(output_dir, "cluster_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build cluster space for Rentropy")
    parser.add_argument("--corpus_file", type=str, help="Single JSON file with questions")
    parser.add_argument("--corpus_dir", type=str, help="Directory with JSON files")
    parser.add_argument("--dataset_source", type=str, default=None,
                       help="Dataset source name (if using --corpus_file)")
    parser.add_argument("--output_dir", type=str, default="./cluster_data")
    parser.add_argument("--num_clusters", type=int, default=128)
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true", default=True)
    args = parser.parse_args()
    
    # Load questions (now returns tuples of (question, source))
    if args.corpus_file:
        dataset_source = args.dataset_source or Path(args.corpus_file).stem
        questions_with_source = load_questions_from_file(args.corpus_file, dataset_source)
    elif args.corpus_dir:
        questions_with_source = load_questions_from_dir(args.corpus_dir)
    else:
        raise ValueError("Must provide either --corpus_file or --corpus_dir")
    
    # Extract just question texts for embedding
    questions = [q[0] for q in questions_with_source]
    print(f"Loaded {len(questions)} questions from {len(set(q[1] for q in questions_with_source))} dataset(s)")
    
    if len(questions) < args.num_clusters:
        print(f"Warning: fewer questions ({len(questions)}) than clusters ({args.num_clusters})")
        args.num_clusters = max(10, len(questions) // 5)
        print(f"Reducing to {args.num_clusters} clusters")
    
    # Embed
    embeddings = embed_questions(questions, args.embedding_model, args.batch_size, args.normalize)
    
    # Cluster
    kmeans = fit_kmeans(embeddings, args.num_clusters)
    
    # Save everything (including embeddings and metadata)
    save_cluster_data(
        args.output_dir,
        kmeans.cluster_centers_,
        kmeans.labels_,
        questions_with_source,  # Pass tuples with source info
        embeddings,  # Pass embeddings to save
    )
    
    print("\n" + "="*60)
    print("Done! Files saved:")
    print(f"  - centroids.npy: Cluster centroids")
    print(f"  - labels.npy: Cluster assignments (indexed by question order)")
    print(f"  - embeddings.npy: Question embeddings (indexed by question order)")
    print(f"  - question_metadata.json: Full metadata with question_id mapping")
    print(f"  - index_to_question_id.json: Quick index -> question_id lookup")
    print(f"  - cluster_stats.json: Summary statistics")
    print("="*60)


if __name__ == "__main__":
    main()
