"""
Reward function for ExpV1_1 GRPO training.
Computes binary reward based on answer correctness.
"""

from __future__ import annotations

import re
from typing import List, Optional


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


def compute_reward(
    generated_text: str,
    ground_truth: str,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    format_penalty: float = 0.0,
) -> float:
    """
    Compute reward for a generated solution.
    
    Args:
        generated_text: The model's generated solution
        ground_truth: The correct answer
        correct_reward: Reward for correct answer (default 1.0)
        incorrect_reward: Reward for incorrect answer (default 0.0)
        format_penalty: Penalty if no \\boxed{} found (default 0.0)
    
    Returns:
        Reward value (float)
    """
    predicted = extract_boxed_answer(generated_text)
    
    if predicted is None:
        # No boxed answer found - apply format penalty
        return incorrect_reward - format_penalty
    
    if answers_equivalent(predicted, ground_truth):
        return correct_reward
    else:
        return incorrect_reward


def compute_batch_rewards(
    generated_texts: List[str],
    ground_truths: List[str],
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    format_penalty: float = 0.0,
) -> List[float]:
    """
    Compute rewards for a batch of generated solutions.
    
    Args:
        generated_texts: List of model generations
        ground_truths: List of correct answers (same length)
        correct_reward: Reward for correct answer
        incorrect_reward: Reward for incorrect answer
        format_penalty: Penalty if no \\boxed{} found
    
    Returns:
        List of reward values
    """
    assert len(generated_texts) == len(ground_truths), \
        f"Length mismatch: {len(generated_texts)} vs {len(ground_truths)}"
    
    return [
        compute_reward(gen, gt, correct_reward, incorrect_reward, format_penalty)
        for gen, gt in zip(generated_texts, ground_truths)
    ]


# For TRL GRPOTrainer - reward function signature
def reward_fn(completions: List[str], prompts: List[str], answers: List[str], **kwargs) -> List[float]:
    """
    Reward function compatible with TRL GRPOTrainer.
    
    Args:
        completions: List of model completions (generated text after prompt)
        prompts: List of prompts (not used for reward, but passed by TRL)
        answers: List of ground truth answers
        **kwargs: Additional arguments from TRL
    
    Returns:
        List of reward values
    """
    return compute_batch_rewards(completions, answers)


if __name__ == "__main__":
    # Test the reward function
    test_cases = [
        # (generated, ground_truth, expected_correct)
        ("Let me solve this... The answer is \\boxed{42}", "42", True),
        ("The answer is \\boxed{42}", "42", True),
        ("\\boxed{42}", "42", True),
        ("The answer is \\boxed{43}", "42", False),
        ("The answer is 42", "42", False),  # No boxed
        ("\\boxed{2+2}", "4", True),  # Math evaluation
        ("\\boxed{\\frac{1}{2}}", "0.5", False),  # Fraction (may not match without math_verify)
    ]
    
    print("Testing reward function:")
    for gen, gt, expected in test_cases:
        reward = compute_reward(gen, gt)
        correct = reward > 0.5
        status = "✓" if correct == expected else "✗"
        print(f"  {status} gen='{gen[:30]}...' gt='{gt}' → reward={reward} (expected {'correct' if expected else 'incorrect'})")
