"""
Cluster assignment and count tracking for Rentropy diversity reward.
This module is imported by the R-Zero reward function.
"""
import numpy as np
import os
from typing import List, Tuple, Optional
from collections import defaultdict
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class ClusterAssigner:
    """Assigns questions to clusters and tracks visit counts."""
    
    def __init__(
        self,
        centroids_path: str,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        ema_decay: float = 0.99,
        smoothing_alpha: float = 1.0,
    ):
        """
        Args:
            centroids_path: Path to centroids.npy file
            embedding_model: Sentence transformer model for embedding
            ema_decay: Decay factor for exponential moving average of counts
            smoothing_alpha: Smoothing constant for log-inverse-frequency reward
        """
        if SentenceTransformer is None:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        # Load centroids
        self.centroids = np.load(centroids_path)
        self.num_clusters = self.centroids.shape[0]
        print(f"[ClusterAssigner] Loaded {self.num_clusters} centroids from {centroids_path}")
        
        # Load embedding model
        # CUDA_VISIBLE_DEVICES should be set by caller before importing this module
        # to ensure we use the designated GPU (e.g., GPU 7)
        print(f"[ClusterAssigner] Loading embedding model: {embedding_model}")
        print(f"[ClusterAssigner] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        
        if torch.cuda.is_available():
            device = 'cuda:0'  # Use first visible GPU (should be the designated embedding GPU)
            print(f"[ClusterAssigner] Using device: {device}")
        else:
            device = 'cpu'
            print(f"[ClusterAssigner] CUDA not available, falling back to CPU")
        
        self.embed_model = SentenceTransformer(embedding_model, trust_remote_code=True, device=device)
        
        # Count tracking (EMA style)
        self.ema_decay = ema_decay
        self.smoothing_alpha = smoothing_alpha
        self.cluster_counts = np.ones(self.num_clusters) * smoothing_alpha  # Initialize with smoothing
        self.total_count = self.num_clusters * smoothing_alpha
    
    def embed(self, questions: List[str]) -> np.ndarray:
        """Embed a list of questions."""
        embeddings = self.embed_model.encode(
            questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
    
    def assign_clusters(self, questions: List[str]) -> np.ndarray:
        """Assign cluster IDs to a list of questions."""
        if not questions:
            return np.array([], dtype=np.int32)
        
        embeddings = self.embed(questions)
        # Compute distances to all centroids
        # Since embeddings are normalized, we can use dot product (higher = closer)
        similarities = embeddings @ self.centroids.T  # (n_questions, n_clusters)
        cluster_ids = np.argmax(similarities, axis=1)
        return cluster_ids
    
    def get_cluster_probabilities(self) -> np.ndarray:
        """Get current estimated cluster probabilities."""
        return self.cluster_counts / self.total_count
    
    def compute_rarity_reward(self, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Compute rarity reward: -log(p(cluster))
        Higher reward for rare clusters.
        """
        probs = self.get_cluster_probabilities()
        # Clip probabilities to prevent log(0) and ensure valid range
        probs = np.clip(probs, 1e-8, 1.0)
        rewards = -np.log(probs[cluster_ids])
        # Use fixed max_reward based on minimum possible probability (always positive)
        max_reward = -np.log(1e-8)  # = 18.4
        rewards = np.clip(rewards / max_reward, 0.0, 1.0)
        return rewards
    
    def compute_batch_uniqueness_reward(self, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Compute within-batch uniqueness reward.
        Reward is higher if cluster is different from other questions in batch.
        """
        n = len(cluster_ids)
        if n <= 1:
            return np.ones(n)
        
        rewards = np.zeros(n)
        for i, c in enumerate(cluster_ids):
            # Count how many others have the same cluster
            same_count = np.sum(cluster_ids == c) - 1  # Exclude self
            # Uniqueness = 1 - (same_count / (n-1))
            rewards[i] = 1.0 - (same_count / (n - 1))
        return rewards
    
    def compute_within_cluster_uniqueness(self, questions: List[str], cluster_ids: np.ndarray) -> np.ndarray:
        """
        Compute within-cluster uniqueness using embedding similarity.
        Questions that are far from their cluster centroid get higher reward.
        """
        if not questions:
            return np.array([])
        
        embeddings = self.embed(questions)
        rewards = np.zeros(len(questions))
        
        for i, (emb, cid) in enumerate(zip(embeddings, cluster_ids)):
            centroid = self.centroids[cid]
            # Cosine similarity (embeddings are normalized)
            similarity = np.dot(emb, centroid)
            # Convert to uniqueness: lower similarity = higher uniqueness
            # similarity is in [-1, 1], so (1 - similarity) / 2 gives [0, 1]
            rewards[i] = (1 - similarity) / 2
        
        return rewards
    
    def update_counts(self, cluster_ids: np.ndarray):
        """Update cluster counts with EMA.
        
        Fixed: Only apply EMA decay when we actually have questions to update.
        This prevents stale cluster distributions from decaying when no questions are generated.
        """
        if len(cluster_ids) == 0:
            # No questions to update, don't decay (preserves current distribution)
            return
        
        # Apply decay ONCE per batch, not per question (fixes reward explosion bug)
        # But only when we have actual questions to process
        self.cluster_counts *= self.ema_decay
        
        for cid in cluster_ids:
            self.cluster_counts[cid] += (1 - self.ema_decay)
        
        self.total_count = np.sum(self.cluster_counts)
    
    def get_stats(self) -> dict:
        """Get current cluster statistics for logging."""
        return {
            "cluster_counts": self.cluster_counts.tolist(),
            "cluster_probabilities": self.get_cluster_probabilities().tolist(),
            "total_count": float(self.total_count),
            "num_clusters": self.num_clusters,
        }


# Global instance (lazy loaded)
_assigner: Optional[ClusterAssigner] = None


def get_assigner(centroids_path: str = None, **kwargs) -> ClusterAssigner:
    """Get or create the global ClusterAssigner instance."""
    global _assigner
    if _assigner is None:
        if centroids_path is None:
            # Default path - adjust as needed
            centroids_path = os.path.join(
                os.path.dirname(__file__),
                "cluster_data",
                "centroids.npy"
            )
        _assigner = ClusterAssigner(centroids_path, **kwargs)
    return _assigner


def reset_assigner():
    """Reset the global assigner (useful for testing)."""
    global _assigner
    _assigner = None
