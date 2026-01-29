"""
Evaluate math models on the MATH dataset using vLLM.

Usage:
    python -m tree.euclid.evaluate_math.evaluate \
        --model Qwen/Qwen2.5-3B \
        --dataset-config number_theory \
        --split test \
        --limit 100 \
        --output results_number_theory.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

# Lazy import vLLM to allow checking args first
LLM = None
SamplingParams = None


def load_vllm():
    """Lazy load vLLM modules."""
    global LLM, SamplingParams
    if LLM is None:
        from vllm import LLM as _LLM, SamplingParams as _SamplingParams
        LLM = _LLM
        SamplingParams = _SamplingParams


# --------------------------------------------------------------------------- #
# Answer extraction and comparison
# --------------------------------------------------------------------------- #

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the *last* \\boxed{...} content from a string (handles nested braces).

    Notes:
    - MATH solutions sometimes contain multiple boxed expressions; the final answer is typically the last one.
    - Model outputs can also include intermediate boxed values; taking the last is more robust.
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


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    - Remove whitespace
    - Convert fractions to standard form
    - Handle common LaTeX
    """
    if answer is None:
        return ""
    
    # Remove whitespace
    answer = answer.strip()
    answer = re.sub(r'\s+', '', answer)
    
    # Remove \text{}, \mathrm{}, etc.
    answer = re.sub(r'\\(text|mathrm|mathbf|mathit)\{([^}]*)\}', r'\2', answer)
    
    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', answer)
    
    # Remove \left and \right
    answer = re.sub(r'\\(left|right)', '', answer)
    
    # Remove $ signs
    answer = answer.replace('$', '')
    
    return answer.lower()


def answers_match(predicted: str, ground_truth: str) -> bool:
    """
    Check if the predicted answer matches the ground truth.
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm == gt_norm:
        return True
    
    # Try numeric comparison
    try:
        pred_val = eval(pred_norm.replace('^', '**'))
        gt_val = eval(gt_norm.replace('^', '**'))
        if abs(pred_val - gt_val) < 1e-6:
            return True
    except:
        pass
    
    return False


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #

def build_prompt(problem: str, few_shot: bool = False) -> str:
    """
    Build the prompt for the model.
    """
    
    # IMPORTANT: require \\boxed{} so answer extraction is comparable across models.
    prompt = f"""You are a careful mathematical problem solver.
Please reason step by step and put your final answer inside \\boxed{{}}.

Problem:
{problem}
"""
    return prompt


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #

def evaluate(
    model_name: str,
    dataset_config: str,
    split: str,
    output_path: Path,
    limit: Optional[int],
    few_shot: bool,
    tensor_parallel_size: int,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """
    Run evaluation on the MATH dataset.
    """
    load_vllm()
    
    # Load dataset
    print(f"Loading dataset: EleutherAI/hendrycks_math/{dataset_config} ({split})")
    ds = load_dataset("EleutherAI/hendrycks_math", dataset_config, split=split)
    
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    print(f"Loaded {len(ds)} examples")
    
    # Initialize vLLM
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["Problem:", "---", "\n\nProblem"],
    )
    
    # Build prompts
    prompts = []
    examples_data = []
    
    for example in ds:
        problem = example["problem"]
        solution = example["solution"]
        level = example.get("level", "")
        
        # Extract ground truth answer
        gt_answer = extract_boxed_answer(solution)
        
        prompt = build_prompt(problem, few_shot=few_shot)
        prompts.append(prompt)
        examples_data.append({
            "problem": problem,
            "solution": solution,
            "level": level,
            "ground_truth": gt_answer,
        })
    
    # Generate responses
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Evaluate results
    results = []
    correct = 0
    total = 0
    level_stats: Dict[str, Dict[str, int]] = {}
    
    for i, output in enumerate(tqdm(outputs, desc="Evaluating")):
        generated_text = output.outputs[0].text
        predicted_answer = extract_boxed_answer(generated_text)
        
        example_data = examples_data[i]
        gt_answer = example_data["ground_truth"]
        level = example_data["level"]
        
        is_correct = answers_match(predicted_answer or "", gt_answer or "")
        
        if is_correct:
            correct += 1
        total += 1
        
        # Track by level
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        if is_correct:
            level_stats[level]["correct"] += 1
        
        result = {
            "problem": example_data["problem"],
            "level": level,
            "ground_truth": gt_answer,
            "predicted": predicted_answer,
            "generated_text": generated_text,
            "correct": is_correct,
        }
        results.append(result)
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"{'='*50}")
    
    print("\nAccuracy by Level:")
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        level_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {level}: {stats['correct']}/{stats['total']} = {level_acc:.2%}")
    
    # Save results
    print(f"\nSaving results to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Save summary
    summary = {
        "model": model_name,
        "dataset": f"EleutherAI/hendrycks_math/{dataset_config}",
        "split": split,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "level_stats": level_stats,
    }
    
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_path}")
    
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models on MATH dataset using vLLM")
    
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="Model name or path (default: Qwen/Qwen2.5-3B)",
    )
    parser.add_argument(
        "--dataset-config",
        default="number_theory",
        choices=[
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ],
        help="MATH dataset config (default: number_theory)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--output",
        default="math_eval_results.jsonl",
        help="Output JSONL path for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max examples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Use few-shot prompting",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    evaluate(
        model_name=args.model,
        dataset_config=args.dataset_config,
        split=args.split,
        output_path=Path(args.output),
        limit=args.limit or None,
        few_shot=args.few_shot,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
