"""
Evaluate math models on the MATH dataset using HuggingFace Transformers.
Works on CPU/MPS (Mac) without requiring CUDA.

Usage:
    python -m euclid.evaluate_math.evaluate_hf \
        --model Qwen/Qwen2.5-3B \
        --dataset-config number_theory \
        --split test \
        --limit 50 \
        --output results_number_theory.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Answer extraction and comparison
# --------------------------------------------------------------------------- #

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the answer from \\boxed{...} in the text.
    Handles nested braces.
    """
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end()
    depth = 1
    pos = start
    
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1
    
    if depth == 0:
        return text[start:pos-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if answer is None:
        return ""
    
    answer = answer.strip()
    answer = re.sub(r'\s+', '', answer)
    answer = re.sub(r'\\(text|mathrm|mathbf|mathit)\{([^}]*)\}', r'\2', answer)
    answer = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', answer)
    answer = re.sub(r'\\(left|right)', '', answer)
    answer = answer.replace('$', '')
    
    return answer.lower()


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if the predicted answer matches the ground truth."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm == gt_norm:
        return True
    
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
    """Build the prompt for the model."""
    if few_shot:
        examples = """Problem: What is the remainder when 2^100 is divided by 7?

Solution: Let me find the pattern of remainders when powers of 2 are divided by 7.
2^1 ≡ 2 (mod 7)
2^2 ≡ 4 (mod 7)
2^3 ≡ 8 ≡ 1 (mod 7)

The pattern repeats every 3 powers. Since 100 = 33 × 3 + 1, we have:
2^100 ≡ 2^1 ≡ 2 (mod 7)

The answer is \\boxed{2}.

---

Problem: Find the greatest common divisor of 1001 and 2431.

Solution: Using the Euclidean algorithm:
2431 = 2 × 1001 + 429
1001 = 2 × 429 + 143
429 = 3 × 143 + 0

The GCD is \\boxed{143}.

---

"""
    else:
        examples = ""
    
    prompt = f"""{examples}Problem: {problem}

Solution: Let me solve this step by step.
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
    max_tokens: int,
    temperature: float,
    device: str,
) -> Dict[str, Any]:
    """Run evaluation on the MATH dataset."""
    
    # Load dataset
    print(f"Loading dataset: EleutherAI/hendrycks_math/{dataset_config} ({split})")
    ds = load_dataset("EleutherAI/hendrycks_math", dataset_config, split=split)
    
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    print(f"Loaded {len(ds)} examples")
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Move to device
    if device == "cuda":
        model = model.half().to(device)
    elif device == "mps":
        model = model.half().to(device)
    else:
        model = model.to(device)
    
    model.eval()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    level_stats: Dict[str, Dict[str, int]] = {}
    
    for example in tqdm(ds, desc="Evaluating"):
        problem = example["problem"]
        solution = example["solution"]
        level = example.get("level", "")
        
        # Extract ground truth answer
        gt_answer = extract_boxed_answer(solution)
        
        # Build prompt and generate
        prompt = build_prompt(problem, few_shot=few_shot)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            if temperature > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted_answer = extract_boxed_answer(generated_text)
        
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
            "problem": problem,
            "level": level,
            "ground_truth": gt_answer,
            "predicted": predicted_answer,
            "generated_text": generated_text,
            "correct": is_correct,
        }
        results.append(result)
        
        # Print running accuracy
        tqdm.write(f"Running accuracy: {correct}/{total} = {correct/total:.2%}")
    
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
    parser = argparse.ArgumentParser(description="Evaluate models on MATH dataset using HuggingFace")
    
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
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
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
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
    )


if __name__ == "__main__":
    main()
