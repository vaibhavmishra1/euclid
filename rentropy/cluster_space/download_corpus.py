#!/usr/bin/env python3
"""
Download math question corpus for cluster building.
Uses HuggingFace datasets to get MATH and GSM8K questions.

Usage:
    python download_corpus.py --output_dir ./corpus
"""
import argparse
import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    raise


def download_math_dataset(output_dir: str):
    """Download MATH dataset questions."""
    print("Downloading MATH dataset...")
    try:
        ds = load_dataset("lighteval/MATH", "all", split="train")
        questions = [item["problem"] for item in ds if item.get("problem")]
        
        output_path = os.path.join(output_dir, "math_questions.json")
        with open(output_path, 'w') as f:
            json.dump(questions, f, indent=2)
        print(f"Saved {len(questions)} MATH questions to {output_path}")
        return questions
    except Exception as e:
        print(f"Failed to download MATH dataset: {e}")
        return []


def download_gsm8k_dataset(output_dir: str):
    """Download GSM8K dataset questions."""
    print("Downloading GSM8K dataset...")
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        questions = [item["question"] for item in ds if item.get("question")]
        
        output_path = os.path.join(output_dir, "gsm8k_questions.json")
        with open(output_path, 'w') as f:
            json.dump(questions, f, indent=2)
        print(f"Saved {len(questions)} GSM8K questions to {output_path}")
        return questions
    except Exception as e:
        print(f"Failed to download GSM8K dataset: {e}")
        return []


def download_math12k_dataset(output_dir: str):
    """Download math12k dataset (used by R-Zero)."""
    print("Downloading math12k dataset...")
    try:
        ds = load_dataset("hiyouga/math12k", split="train")
        questions = [item["problem"] for item in ds if item.get("problem")]
        
        output_path = os.path.join(output_dir, "math12k_questions.json")
        with open(output_path, 'w') as f:
            json.dump(questions, f, indent=2)
        print(f"Saved {len(questions)} math12k questions to {output_path}")
        return questions
    except Exception as e:
        print(f"Failed to download math12k dataset: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Download math corpus for cluster building")
    parser.add_argument("--output_dir", type=str, default="./corpus")
    parser.add_argument("--datasets", nargs="+", default=["math12k"], 
                        choices=["math", "gsm8k", "math12k", "all"],
                        help="Which datasets to download")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_questions = []
    
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["math", "gsm8k", "math12k"]
    
    if "math" in datasets_to_download:
        all_questions.extend(download_math_dataset(args.output_dir))
    
    if "gsm8k" in datasets_to_download:
        all_questions.extend(download_gsm8k_dataset(args.output_dir))
    
    if "math12k" in datasets_to_download:
        all_questions.extend(download_math12k_dataset(args.output_dir))
    
    # Save combined corpus
    combined_path = os.path.join(args.output_dir, "all_questions.json")
    with open(combined_path, 'w') as f:
        json.dump(all_questions, f, indent=2)
    print(f"\nTotal: {len(all_questions)} questions saved to {combined_path}")
    print("\nNext step: Run build_clusters.py to create cluster centroids")
    print(f"  python build_clusters.py --corpus_file {combined_path} --output_dir ./cluster_data")


if __name__ == "__main__":
    main()
