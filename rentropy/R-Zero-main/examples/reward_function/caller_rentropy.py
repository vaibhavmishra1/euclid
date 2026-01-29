# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modified for Rentropy: Adds cluster-entropy diversity reward to R-Zero
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Rentropy reward function with 4 configurable diversity modes:
  1: Vanilla majority voting reward only (R-Zero baseline)
  2: Mode 1 + reward for choosing a rare cluster
  3: Mode 2 + reward for uniqueness from other n-1 questions in batch
  4: Mode 3 + within-cluster uniqueness reward
"""

import regex as re
from typing import Dict, List
import json
import os
import time
import random
import requests
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from collections import Counter
from mathruler.grader import extract_boxed_content, grade_answer

# Add cluster_space to path
import sys
RENTROPY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, RENTROPY_ROOT)

from cluster_space.cluster_assigner import ClusterAssigner

STORAGE_PATH = os.getenv("STORAGE_PATH", "/apdcephfs_sh2/share_300000800/user/chengchuang")

# ============================================================================
# Rentropy Configuration Loading
# ============================================================================

def load_rentropy_config() -> dict:
    """Load rentropy configuration from yaml file."""
    config_path = os.path.join(RENTROPY_ROOT, "rentropy_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[Rentropy] Loaded config from {config_path}")
        print(f"[Rentropy] Diversity mode: {config.get('diversity_mode', 1)}")
        return config
    else:
        print(f"[Rentropy] Config not found at {config_path}, using defaults (mode 1)")
        return {
            "diversity_mode": 1,
            "centroids_path": None,
            "weights": {"rarity": 0.1, "batch_uniqueness": 0.05, "within_cluster_uniqueness": 0.05},
            "majority_vote_threshold": 0.3,
            "ema_decay": 0.99,
            "smoothing_alpha": 1.0,
        }

# Load config once at module import
RENTROPY_CONFIG = load_rentropy_config()

# Lazy-loaded cluster assigner
_cluster_assigner = None

def get_cluster_assigner() -> ClusterAssigner:
    """Get or create the cluster assigner instance."""
    global _cluster_assigner
    if _cluster_assigner is None and RENTROPY_CONFIG.get("diversity_mode", 1) > 1:
        centroids_path = RENTROPY_CONFIG.get("centroids_path")
        if centroids_path and not os.path.isabs(centroids_path):
            centroids_path = os.path.join(RENTROPY_ROOT, centroids_path)
        
        if centroids_path and os.path.exists(centroids_path):
            _cluster_assigner = ClusterAssigner(
                centroids_path=centroids_path,
                embedding_model=RENTROPY_CONFIG.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
                ema_decay=RENTROPY_CONFIG.get("ema_decay", 0.99),
                smoothing_alpha=RENTROPY_CONFIG.get("smoothing_alpha", 1.0),
            )
        else:
            print(f"[Rentropy] WARNING: centroids not found at {centroids_path}, falling back to mode 1")
    return _cluster_assigner

# ============================================================================
# Utility Functions (from original R-Zero)
# ============================================================================

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index, i):
    response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={i}")
    print(response)
    return True

def generate_results(data):
    """Call vLLM servers to get majority voting scores."""
    datas = split_list(data, 4)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(4)]
    for i in range(4):
        with open(random_names[i], 'w') as f:
            json.dump(datas[i], f, indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, i, random_names[i]) for i in range(4)]
        for future in as_completed(futures):
            print(future.result())

    for i in range(4):
        with open(random_names[i].replace('.json', '_results.json'), 'r') as f:
            final_results.extend(json.load(f))
    for i in range(4):
        os.remove(random_names[i].replace('.json', '_results.json'))
    return final_results

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0

def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

# ============================================================================
# Rentropy Diversity Reward
# ============================================================================

def compute_diversity_rewards(questions: List[str], base_scores: List[float]) -> List[float]:
    """
    Compute diversity rewards based on configured mode.
    
    Args:
        questions: List of question strings
        base_scores: Majority voting scores for each question
    
    Returns:
        List of diversity reward bonuses (0 if below threshold or mode=1)
    """
    mode = RENTROPY_CONFIG.get("diversity_mode", 1)
    threshold = RENTROPY_CONFIG.get("majority_vote_threshold", 0.3)
    weights = RENTROPY_CONFIG.get("weights", {})
    
    n = len(questions)
    diversity_rewards = np.zeros(n)
    
    if mode == 1:
        # Mode 1: No diversity reward
        return diversity_rewards.tolist()
    
    assigner = get_cluster_assigner()
    if assigner is None:
        print("[Rentropy] No cluster assigner available, returning zero diversity rewards")
        return diversity_rewards.tolist()
    
    # Filter to questions that pass majority vote threshold
    valid_indices = [i for i, s in enumerate(base_scores) if s >= threshold and questions[i]]
    valid_questions = [questions[i] for i in valid_indices]
    
    if not valid_questions:
        return diversity_rewards.tolist()
    
    # Assign clusters
    cluster_ids = assigner.assign_clusters(valid_questions)
    
    # Mode 2+: Rarity reward
    if mode >= 2:
        rarity_rewards = assigner.compute_rarity_reward(cluster_ids)
        rarity_weight = weights.get("rarity", 0.1)
        for idx, valid_idx in enumerate(valid_indices):
            diversity_rewards[valid_idx] += rarity_weight * rarity_rewards[idx]
    
    # Mode 3+: Batch uniqueness reward
    if mode >= 3:
        batch_uniqueness_rewards = assigner.compute_batch_uniqueness_reward(cluster_ids)
        batch_weight = weights.get("batch_uniqueness", 0.05)
        for idx, valid_idx in enumerate(valid_indices):
            diversity_rewards[valid_idx] += batch_weight * batch_uniqueness_rewards[idx]
    
    # Mode 4: Within-cluster uniqueness reward
    if mode >= 4:
        within_cluster_rewards = assigner.compute_within_cluster_uniqueness(valid_questions, cluster_ids)
        within_weight = weights.get("within_cluster_uniqueness", 0.05)
        for idx, valid_idx in enumerate(valid_indices):
            diversity_rewards[valid_idx] += within_weight * within_cluster_rewards[idx]
    
    # Update cluster counts for valid questions
    assigner.update_counts(cluster_ids)
    
    return diversity_rewards.tolist()

# ============================================================================
# Main Reward Function
# ============================================================================

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1, file_path: str = "") -> List[Dict[str, float]]:
    """
    Compute rewards with Rentropy diversity bonus.
    
    Returns dict with:
        - overall: final score (ZPD-style base + diversity bonus)
        - format: 1 if valid format, 0 otherwise
        - accuracy: diversity reward (for logging compatibility)
        - diversity: diversity reward bonus
        - base_score: original majority voting score
    """
    results = []
    
    # Extract questions and answers from predictions
    for i in range(len(predicts)):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})
    
    # Get majority voting scores from vLLM servers
    final_results = generate_results(results)
    
    # Extract questions and base scores
    questions = [r['question'] for r in final_results]
    base_scores = [r['score'] if r['question'] else -1 for r in final_results]
    
    # Compute diversity rewards
    diversity_rewards = compute_diversity_rewards(questions, base_scores)
    
    # Compute final scores
    scores = []
    for i in range(len(final_results)):
        base_score = final_results[i]["score"]
        has_valid_question = bool(final_results[i]['question'])
        
        if has_valid_question and base_score >= 0:
            # ZPD-style score: min(score, 1-score) peaks at 0.5
            zpd_score = min(base_score, 1 - base_score)
            # Add diversity bonus
            final_score = zpd_score + diversity_rewards[i]
        else:
            final_score = -1
        
        scores.append({
            "overall": final_score,
            "format": 1 if has_valid_question else 0,
            "accuracy": diversity_rewards[i],  # For logging compatibility
            "diversity": diversity_rewards[i],
            "base_score": base_score,
        })
    
    return scores
