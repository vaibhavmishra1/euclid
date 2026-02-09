import argparse
import json
import os
import sys
from typing import List

import numpy as np

# Add rentropy root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# question_generate is at: R-Zero-main/question_generate
# rentropy root is two levels up from SCRIPT_DIR
RENTROPY_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, RENTROPY_ROOT)

from cluster_space.cluster_assigner import ClusterAssigner


def load_rentropy_config() -> dict:
    config_path = os.path.join(RENTROPY_ROOT, "rentropy_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.loads(json.dumps(__import__("yaml").safe_load(f)))
    return {
        "diversity_mode": 1,
        "centroids_path": None,
        "weights": {"rarity": 0.5, "batch_uniqueness": 0.2, "within_cluster_uniqueness": 0.2},
    }


def compute_diversity_score_readonly(question: str, assigner: ClusterAssigner, config: dict) -> float:
    if not question or not question.strip():
        return 0.0
    mode = config.get("diversity_mode", 1)
    if mode == 1:
        return 0.0
    weights = config.get("weights", {})
    cluster_ids = assigner.assign_clusters([question])
    if len(cluster_ids) == 0:
        return 0.0
    diversity_score = 0.0
    if mode == 5:
        rarity_rewards = assigner.compute_rarity_reward(cluster_ids)
        return float(rarity_rewards[0])
    if mode >= 2:
        rarity_rewards = assigner.compute_rarity_reward(cluster_ids)
        diversity_score += weights.get("rarity", 0.5) * rarity_rewards[0]
    if mode >= 3:
        diversity_score += weights.get("batch_uniqueness", 0.2) * 1.0
    if mode >= 4:
        within_cluster_rewards = assigner.compute_within_cluster_uniqueness([question], cluster_ids)
        diversity_score += weights.get("within_cluster_uniqueness", 0.2) * within_cluster_rewards[0]
    return float(diversity_score)


def process_file(path: str, assigner: ClusterAssigner, config: dict) -> int:
    with open(path, "r") as f:
        data = json.load(f)
    updated = 0
    for item in data:
        if item.get("score", -1) != 0:
            continue
        question = item.get("question", "")
        diversity_score = compute_diversity_score_readonly(question, assigner, config)
        item["diversity_score"] = diversity_score
        updated += 1
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True, help="Experiment name prefix")
    parser.add_argument("--storage_path", default=os.getenv("STORAGE_PATH"), help="Storage path")
    parser.add_argument("--device", default="cuda:0", help="Embedding device (e.g. cuda:0)")
    parser.add_argument("--num_shards", type=int, default=7, help="Number of generation shards")
    args = parser.parse_args()

    if not args.storage_path:
        raise ValueError("STORAGE_PATH is not set and --storage_path not provided.")

    config = load_rentropy_config()
    centroids_path = config.get("centroids_path")
    if not centroids_path:
        raise ValueError("centroids_path missing in rentropy_config.yaml")
    if not os.path.isabs(centroids_path):
        centroids_path = os.path.join(RENTROPY_ROOT, centroids_path)

    init_counts_path = config.get("init_cluster_counts_path")
    if init_counts_path and not os.path.isabs(init_counts_path):
        init_counts_path = os.path.join(RENTROPY_ROOT, init_counts_path)

    assigner = ClusterAssigner(
        centroids_path=centroids_path,
        embedding_model=config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        ema_decay=config.get("ema_decay", 0.99),
        smoothing_alpha=config.get("smoothing_alpha", 1.0),
        init_counts_path=init_counts_path,
        device=args.device,
    )

    total_updated = 0
    for i in range(args.num_shards):
        path = os.path.join(args.storage_path, "generated_question", f"{args.experiment_name}_{i}.json")
        if not os.path.exists(path):
            print(f"[Diversity] File not found: {path}")
            continue
        updated = process_file(path, assigner, config)
        total_updated += updated
        print(f"[Diversity] Updated {updated} items in {path}")

    print(f"[Diversity] Done. Total updated: {total_updated}")


if __name__ == "__main__":
    main()
