"""
Reward Function for Knowledge-Point-Based Challenger Training

This module implements the reward function used to train the challenger model.
The reward is based on:
1. Uncertainty reward: How uncertain is the solver about the answer (target: 50%)
2. Format check: Is the output properly formatted
3. Repetition penalty: Penalize similar questions in the same batch
"""

import regex as re
from typing import Dict, List
import json
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Try to import mathruler, provide fallback if not available
try:
    from mathruler.grader import extract_boxed_content, grade_answer
except ImportError:
    print("Warning: mathruler not found, using basic extraction")
    
    def extract_boxed_content(text: str) -> str:
        """Extract content from \\boxed{...}"""
        results = []
        prefix = r'\boxed{'
        plen = len(prefix)
        i = 0
        
        while True:
            start = text.find(prefix, i)
            if start == -1:
                break
            
            j = start + plen
            depth = 1
            while j < len(text) and depth:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            
            results.append(text[start + plen : j - 1])
            i = j
        
        return results[-1] if results else None
    
    def grade_answer(pred: str, truth: str) -> bool:
        """Simple answer grading."""
        if pred is None or truth is None:
            return False
        return pred.strip().lower() == truth.strip().lower()


STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/rzero_storage")


def _bleu_distance_matrix(sentences: List[str]) -> np.ndarray:
    """Compute BLEU-based distance matrix for sentences."""
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    
    return dist


def cluster_share_per_problem(
    problems: List[str],
    distance_threshold: float = 0.5,
    linkage: str = "average"
) -> List[float]:
    """
    Compute repetition penalty based on BLEU clustering.
    
    Returns proportion of cluster size for each problem.
    """
    if not problems:
        return []
    
    if len(problems) == 1:
        return [0.0]
    
    print('Starting BLEU clustering...')
    start_time = time.time()
    
    dist_mat = _bleu_distance_matrix(problems)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    
    print(f'Clustering completed in {time.time() - start_time:.2f}s')
    
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}
    
    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


def generate_temp_filename(prefix: str = "temp", suffix: str = ".json") -> str:
    """Generate a temporary filename."""
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"


def split_list(lst: List, n: int = 4) -> List[List]:
    """Split a list into n roughly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"


def fetch(index: int, filepath: str) -> bool:
    """Fetch results from vLLM server."""
    try:
        response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={filepath}", timeout=600)
        return True
    except Exception as e:
        print(f"Error fetching from server {index}: {e}")
        return False


def generate_results(data: List[Dict], num_servers: int = 4) -> List[Dict]:
    """
    Generate results by distributing work across vLLM servers.
    
    This function sends questions to solver servers and collects
    the uncertainty scores (based on answer consistency).
    """
    if not data:
        return []
    
    datas = split_list(data, num_servers)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(num_servers)]
    
    # Ensure temp directory exists
    os.makedirs(f"{STORAGE_PATH}/temp_results", exist_ok=True)
    
    # Write data to temp files
    for i in range(num_servers):
        with open(random_names[i], 'w', encoding='utf-8') as f:
            json.dump(datas[i], f, indent=4, ensure_ascii=False)
    
    # Fetch results in parallel
    final_results = []
    with ThreadPoolExecutor(max_workers=num_servers) as executor:
        futures = [executor.submit(fetch, i, random_names[i]) for i in range(num_servers)]
        
        for future in as_completed(futures):
            print(f"Server completed: {future.result()}")
    
    # Collect results
    for i in range(num_servers):
        result_file = random_names[i].replace('.json', '_results.json')
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                final_results.extend(json.load(f))
            os.remove(result_file)
        except FileNotFoundError:
            print(f"Warning: Result file {result_file} not found")
    
    # Cleanup temp files
    for name in random_names:
        try:
            os.remove(name)
        except:
            pass
    
    return final_results


def format_reward(predict: str) -> float:
    """Check if prediction has correct format."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.1,
    file_path: str = ""
) -> List[Dict[str, float]]:
    """
    Compute scores for challenger outputs.
    
    This is the main reward function used during challenger GRPO training.
    
    Args:
        predicts: List of challenger outputs (questions + answers)
        ground_truths: List of ground truths (not used for challenger)
        format_weight: Weight for format reward (unused)
        file_path: Optional file path for debugging
    
    Returns:
        List of score dicts with 'overall', 'format', 'accuracy' keys
    """
    results = []
    
    # Debug output
    with open('challenger_debug.json', 'w', encoding='utf-8') as f:
        json.dump(predicts, f, indent=4, ensure_ascii=False)
    
    # Parse questions and answers from predictions
    for i in range(len(predicts)):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers if isinstance(answers, str) else answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})
    
    # Get uncertainty scores from solver
    final_results = generate_results(results)
    
    # Calculate repetition penalty
    valid_questions = [r['question'] for r in final_results if r.get('question')]
    
    if valid_questions:
        penalty = cluster_share_per_problem(
            valid_questions,
            distance_threshold=0.5
        )
    else:
        penalty = [0.0] * len(final_results)
    
    # Ensure penalty list matches results
    penalty_idx = 0
    full_penalty = []
    for r in final_results:
        if r.get('question'):
            full_penalty.append(penalty[penalty_idx] if penalty_idx < len(penalty) else 0.0)
            penalty_idx += 1
        else:
            full_penalty.append(1.0)  # Maximum penalty for invalid
    
    assert len(full_penalty) == len(final_results)
    
    # Compute final scores
    scores = []
    for i in range(len(final_results)):
        # Uncertainty reward: maximize when score is around 0.5
        # r_uncertainty = 1 - 2 * |p - 0.5| where p is solver accuracy
        if final_results[i].get('question'):
            solver_score = final_results[i].get("score", 0.5)
            uncertainty_reward = min(solver_score, 1 - solver_score)
            
            # Subtract repetition penalty
            final_score = uncertainty_reward - full_penalty[i]
            final_score = max(0, final_score)  # Clamp to non-negative
        else:
            final_score = -1  # Invalid format
        
        scores.append({
            "overall": final_score,
            "format": 1 if final_results[i].get('question') else 0,
            "accuracy": full_penalty[i],  # Store penalty in accuracy field
            "knowledge_point": final_results[i].get("knowledge_point", ""),
            "difficulty": final_results[i].get("difficulty", 1),
        })
    
    return scores


def compute_score_with_knowledge_tracking(
    predicts: List[str],
    ground_truths: List[str],
    knowledge_points: List[str],
    difficulties: List[int],
    format_weight: float = 0.1,
) -> tuple:
    """
    Compute scores with knowledge point tracking.
    
    This version also returns per-knowledge-point statistics.
    
    Args:
        predicts: List of challenger outputs
        ground_truths: List of ground truths (unused)
        knowledge_points: List of knowledge points for each prediction
        difficulties: List of difficulty levels for each prediction
        format_weight: Weight for format reward
    
    Returns:
        Tuple of (scores, kp_stats)
    """
    # Compute base scores
    scores = compute_score(predicts, ground_truths, format_weight)
    
    # Aggregate by knowledge point
    kp_stats = {}
    for i, (kp, diff) in enumerate(zip(knowledge_points, difficulties)):
        if kp not in kp_stats:
            kp_stats[kp] = {
                "difficulty": diff,
                "rewards": [],
                "valid_count": 0,
                "total_count": 0
            }
        
        kp_stats[kp]["total_count"] += 1
        if scores[i]["overall"] >= 0:
            kp_stats[kp]["rewards"].append(scores[i]["overall"])
            kp_stats[kp]["valid_count"] += 1
    
    # Calculate average reward per knowledge point
    for kp in kp_stats:
        rewards = kp_stats[kp]["rewards"]
        kp_stats[kp]["avg_reward"] = sum(rewards) / len(rewards) if rewards else 0.0
    
    return scores, kp_stats


# Test function
if __name__ == "__main__":
    # Simple test
    test_predicts = [
        "<question>What is 2+2?</question>\n\\boxed{4}",
        "Invalid output without proper format",
        "<question>Solve x^2 = 4</question>\n\\boxed{\\pm 2}",
    ]
    test_truths = ["", "", ""]
    
    print("Testing reward computation...")
    # Note: This won't work without the vLLM servers running
    # scores = compute_score(test_predicts, test_truths)
    # print(f"Scores: {scores}")
