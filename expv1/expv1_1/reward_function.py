"""
Reward function for ExpV1_1 GRPO training.
Uses SELF-CONSISTENCY across rollouts to determine correctness.

For synthetic datasets with no ground truth, we:
1. Generate multiple rollouts per prompt
2. Extract \boxed{} answer from each rollout
3. Find the modal (majority) answer
4. Reward rollouts that agree with the modal answer
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} content from a string.
    Handles nested braces.
    
    Reused from ExpV1_0 text_parse.py
    """
    prefix = r"\boxed{"
    i = 0
    last: Optional[str] = None
    while True:
        start = text.find(prefix, i)
        if start == -1:
            break
        j = start + len(prefix)
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[start + len(prefix) : j - 1].strip()
        i = j
    return last


def normalize_answer_str(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("$", "")
    return s.lower()


def answers_equivalent(pred: str, gold: str) -> bool:
    """
    Check if predicted answer is equivalent to gold answer.
    
    Reused from ExpV1_0 grading.py
    """
    pred = pred.strip()
    gold = gold.strip()
    if not pred or not gold:
        return False

    # Exact match after normalization
    if normalize_answer_str(pred) == normalize_answer_str(gold):
        return True

    # Try math-verify if available
    try:
        from math_verify import parse, verify
        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        pass
    
    # Try numeric comparison
    try:
        pred_val = float(eval(pred.replace("^", "**")))
        gold_val = float(eval(gold.replace("^", "**")))
        if abs(pred_val - gold_val) < 1e-6:
            return True
    except Exception:
        pass

    return False


def find_modal_answer(answers: List[Optional[str]]) -> Tuple[Optional[str], int]:
    """
    Find the modal (most common) answer from a list.
    
    Args:
        answers: List of extracted answers (may contain None)
    
    Returns:
        Tuple of (modal_answer, count)
    """
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return None, 0
    
    # Normalize answers for counting
    normalized_counts: Counter = Counter()
    normalized_to_original: dict = {}
    
    for ans in valid_answers:
        norm = normalize_answer_str(ans)
        normalized_counts[norm] += 1
        if norm not in normalized_to_original:
            normalized_to_original[norm] = ans
    
    most_common_norm, count = normalized_counts.most_common(1)[0]
    return normalized_to_original[most_common_norm], count


def compute_self_consistency_rewards(
    completions: List[str],
    num_generations: int,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    format_penalty: float = 0.1,
    min_agreement: int = 2,
) -> List[float]:
    """
    Compute rewards based on self-consistency across rollout groups.
    
    For each group of num_generations rollouts (for the same prompt):
    1. Extract answer from each completion
    2. Find the modal (majority) answer
    3. Reward completions that match the modal answer
    
    Args:
        completions: List of all completions (grouped by prompt)
        num_generations: Number of rollouts per prompt
        correct_reward: Reward for matching modal answer
        incorrect_reward: Reward for not matching
        format_penalty: Additional penalty if no \boxed{} found
        min_agreement: Minimum count for modal answer to be trusted
    
    Returns:
        List of rewards (same length as completions)
    """
    rewards = []
    num_prompts = len(completions) // num_generations
    
    for prompt_idx in range(num_prompts):
        start_idx = prompt_idx * num_generations
        end_idx = start_idx + num_generations
        
        group_completions = completions[start_idx:end_idx]
        
        # Extract answers from all rollouts in this group
        extracted_answers = [extract_boxed_answer(c) for c in group_completions]
        
        # Find modal answer
        modal_answer, modal_count = find_modal_answer(extracted_answers)
        
        # Determine if we have sufficient consensus
        has_consensus = modal_answer is not None and modal_count >= min_agreement
        
        # Assign rewards for this group
        for answer in extracted_answers:
            if answer is None:
                # No valid answer format - penalty
                rewards.append(incorrect_reward - format_penalty)
            elif not has_consensus:
                # No clear consensus - neutral reward (just having format is slightly better)
                rewards.append(incorrect_reward)
            elif answers_equivalent(answer, modal_answer):
                # Matches consensus answer - reward!
                rewards.append(correct_reward)
            else:
                # Disagrees with consensus
                rewards.append(incorrect_reward)
    
    return rewards


def create_self_consistency_reward_fn(
    num_generations: int,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    format_penalty: float = 0.1,
    min_agreement: int = 2,
):
    """
    Create a self-consistency reward function for TRL's GRPOTrainer.
    
    TRL's GRPOTrainer passes completions grouped by prompt:
    - For num_generations=4 and batch_size=2:
      completions[0:4] are 4 rollouts for prompt[0]
      completions[4:8] are 4 rollouts for prompt[1]
    
    Args:
        num_generations: Number of rollouts per prompt (group size)
        correct_reward: Reward for matching modal answer
        incorrect_reward: Reward for not matching
        format_penalty: Penalty if no \\boxed{} found
        min_agreement: Minimum agreement for modal answer to be trusted
    
    Returns:
        Reward function compatible with GRPOTrainer
    """
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        """
        Compute self-consistency rewards for all completions.
        
        Args:
            completions: List of model completions (grouped by prompt)
            **kwargs: Additional arguments from TRL (prompts, etc.)
        
        Returns:
            List of reward values
        """
        return compute_self_consistency_rewards(
            completions=completions,
            num_generations=num_generations,
            correct_reward=correct_reward,
            incorrect_reward=incorrect_reward,
            format_penalty=format_penalty,
            min_agreement=min_agreement,
        )
    
    return reward_fn


# Legacy function for backward compatibility (not used in self-consistency mode)
def compute_reward(
    generated_text: str,
    ground_truth: str,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    format_penalty: float = 0.0,
) -> float:
    """
    Compute reward for a generated solution (requires ground truth).
    
    NOTE: This is NOT used in self-consistency mode.
    Kept for reference/testing only.
    """
    predicted = extract_boxed_answer(generated_text)
    
    if predicted is None:
        return incorrect_reward - format_penalty
    
    if answers_equivalent(predicted, ground_truth):
        return correct_reward
    else:
        return incorrect_reward


if __name__ == "__main__":
    # Test self-consistency reward function
    print("Testing self-consistency reward function:")
    print("=" * 60)
    
    # Simulate 2 prompts with 4 rollouts each
    num_generations = 4
    
    completions = [
        # Group 1: 3 agree on 42, 1 disagrees
        "The answer is \\boxed{42}",
        "So we get \\boxed{42}",
        "Therefore \\boxed{42}",
        "I think \\boxed{43}",  # Wrong
        # Group 2: 2 agree on 7, 1 says 8, 1 has no boxed
        "The result is \\boxed{7}",
        "Answer: \\boxed{7}",
        "\\boxed{8}",  # Disagrees
        "The answer is 7",  # No boxed - format penalty
    ]
    
    rewards = compute_self_consistency_rewards(
        completions=completions,
        num_generations=num_generations,
        correct_reward=1.0,
        incorrect_reward=0.0,
        format_penalty=0.1,
        min_agreement=2,
    )
    
    print(f"\nNum prompts: {len(completions) // num_generations}")
    print(f"Num generations per prompt: {num_generations}")
    print(f"\nCompletions and Rewards:")
    
    for i, (comp, rew) in enumerate(zip(completions, rewards)):
        prompt_idx = i // num_generations
        rollout_idx = i % num_generations
        answer = extract_boxed_answer(comp)
        print(f"  Prompt {prompt_idx}, Rollout {rollout_idx}: answer={answer!r:10} reward={rew:.2f}")
    
    print(f"\nExpected behavior:")
    print(f"  Group 1: modal=42 (count=3) -> 3 correct (1.0), 1 incorrect (0.0)")
    print(f"  Group 2: modal=7 (count=2) -> 2 correct (1.0), 1 incorrect (0.0), 1 no-format (-0.1)")
