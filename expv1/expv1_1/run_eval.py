"""
ExpV1_1: Evaluation Script

Wrapper around evaluate_math.evaluate_hf for convenience.
Evaluates the GRPO-trained model on MATH number_theory.

Usage:
    python -m tree.euclid.expv1.expv1_1.run_eval \
        --model ./output_grpo \
        --output results_grpo.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="ExpV1_1: Evaluate GRPO model")
    
    parser.add_argument(
        "--model",
        type=str,
        default="./output_expv1_1_grpo",
        help="Path to trained model or HuggingFace model ID",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
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
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_grpo.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max examples (0 = all)",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Use few-shot prompting",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Build command for evaluate_hf
    cmd = [
        sys.executable,
        "-m", "tree.euclid.evaluate_math.evaluate_hf",
        "--model", args.model,
        "--dataset-config", args.dataset_config,
        "--split", args.split,
        "--output", args.output,
        "--device", args.device,
    ]
    
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    
    if args.few_shot:
        cmd.append("--few-shot")
    
    print(f"Running evaluation: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run evaluation
    result = subprocess.run(cmd)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
