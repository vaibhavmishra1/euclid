"""
Self-consistency reward computation.
"""
from typing import List, Dict
from collections import Counter


def compute_self_consistency_reward(rollouts: List[Dict]) -> float:
    """
    Compute self-consistency reward based on majority voting.
    
    Args:
        rollouts: List of rollouts, each with 'answer' field
        
    Returns:
        Reward score (0.0 to 1.0) based on agreement with majority answer
    """
    if not rollouts:
        return 0.0
    
    # Extract answers (may be empty if the model didn't produce a boxed answer)
    answers = [r["answer"] for r in rollouts if r.get("answer")]
    
    if not answers:
        return 0.0
    
    # Find majority answer (simple string matching)
    answer_counts = Counter(answers)
    majority_answer, majority_count = answer_counts.most_common(1)[0]
    
    # Reward is the fraction of *all rollouts* that agree with majority.
    # This penalizes rollouts that fail to produce a valid boxed answer.
    reward = majority_count / max(1, len(rollouts))
    
    return reward


def compute_rewards_for_batch(questions: List[str], rollouts_per_question: List[List[Dict]]) -> List[float]:
    """
    Compute rewards for a batch of questions.
    
    Args:
        questions: List of question strings
        rollouts_per_question: List of lists, each inner list contains rollouts for one question
        
    Returns:
        List of rewards (one per question)
    """
    rewards = []
    for rollouts in rollouts_per_question:
        reward = compute_self_consistency_reward(rollouts)
        rewards.append(reward)
    
    return rewards
