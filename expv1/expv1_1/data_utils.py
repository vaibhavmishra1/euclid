"""
Data utilities for ExpV1_1 GRPO training.
Loads and converts the ExpV1_0 dataset for use with TRL's GRPOTrainer.

NOTE: For self-consistency based GRPO, we only need the 'problem' field.
The 'answer' field from the dataset is NOT used for reward computation -
rewards are computed via self-consistency across rollouts during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset


def load_expv1_0_dataset(
    jsonl_path: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load the accepted dataset from ExpV1_0.
    
    Args:
        jsonl_path: Path to accepted.jsonl or accepted_verified_retried.jsonl
        limit: Optional limit on number of examples
    
    Returns:
        List of dicts with 'problem' field (answer is optional, not used for training)
    """
    data = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            row = json.loads(line.strip())
            
            # Handle both formats: original and verified
            if "original_data" in row:
                # Verified format
                original = row.get("original_data", {})
                candidate = original.get("candidate", {})
                problem = candidate.get("problem", "")
            else:
                # Original format
                candidate = row.get("candidate", {})
                problem = candidate.get("problem", "")
            
            if problem:
                data.append({
                    "problem": problem,
                })
    
    print(f"[data_utils] Loaded {len(data)} problems from {jsonl_path}")
    return data


def create_grpo_dataset(
    data: List[Dict[str, Any]],
    prompt_template: str,
) -> Dataset:
    """
    Create a HuggingFace Dataset for GRPO training.
    
    For self-consistency GRPO, we only need the 'prompt' column.
    The reward is computed during training from rollout agreement,
    NOT from a ground truth answer.
    
    Args:
        data: List of dicts with 'problem' field
        prompt_template: Jinja-style template for the prompt
    
    Returns:
        HuggingFace Dataset with 'prompt' column
    """
    from jinja2 import Template
    
    template = Template(prompt_template)
    
    processed = []
    for item in data:
        prompt = template.render(problem=item["problem"])
        processed.append({
            "prompt": prompt,
        })
    
    return Dataset.from_list(processed)


def get_default_prompt_template() -> str:
    """
    Get the default prompt template for math problem solving.
    This matches the format used in ExpV1_0.
    """
    return """You are a careful mathematical problem solver.
Please reason step by step and put your final answer inside \\boxed{}.

Problem:
{{ problem }}

Solution:"""


def load_dataset_for_grpo(
    jsonl_path: str,
    prompt_template: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dataset:
    """
    Convenience function to load and prepare dataset for GRPO.
    
    For self-consistency based GRPO:
    - Only the 'prompt' column is needed
    - Rewards are computed from agreement across rollouts during training
    - No ground truth answer is used
    
    Args:
        jsonl_path: Path to ExpV1_0 accepted.jsonl
        prompt_template: Optional custom prompt template
        limit: Optional limit on number of examples
    
    Returns:
        HuggingFace Dataset ready for GRPOTrainer (with 'prompt' column)
    """
    if prompt_template is None:
        prompt_template = get_default_prompt_template()
    
    data = load_expv1_0_dataset(jsonl_path, limit=limit)
    dataset = create_grpo_dataset(data, prompt_template)
    
    return dataset


if __name__ == "__main__":
    # Test loading
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to accepted.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Number of examples to show")
    args = parser.parse_args()
    
    dataset = load_dataset_for_grpo(args.input, limit=args.limit)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"\nFirst example:")
    print(f"Prompt: {dataset[0]['prompt'][:200]}...")
    print(f"\nNOTE: No 'answer' column - rewards computed via self-consistency during training")
