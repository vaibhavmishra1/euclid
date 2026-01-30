"""
Cluster space configuration for Rentropy.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClusterConfig:
    """Configuration for cluster space construction."""
    
    # Embedding model
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    
    # Number of clusters
    num_clusters: int = 2048
    
    # Paths
    corpus_dir: str = "./corpus"  # Directory containing question files
    output_dir: str = "./cluster_data"  # Where to save centroids
    centroids_file: str = "centroids.npy"
    
    # Processing
    batch_size: int = 1024 * 10
    normalize_embeddings: bool = True
    
    # K-means settings
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    random_state: int = 42
