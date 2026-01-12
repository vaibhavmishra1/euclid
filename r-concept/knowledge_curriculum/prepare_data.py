"""
Data Preparation Utility for Knowledge-Point-Based Curriculum Learning

This script prepares training data from knowledge points JSONL file:
1. Loads knowledge points
2. Creates training dataset for challenger
3. Exports to HuggingFace format
"""

import json
import os
import argparse
from typing import List, Dict
from datasets import Dataset, DatasetDict
from huggingface_hub import login

from .knowledge_manager import KnowledgePointManager


def prepare_challenger_data(
    knowledge_points_path: str,
    output_path: str,
    questions_per_kp: int = 5,
    state_path: str = None,
) -> str:
    """
    Prepare training data for challenger from knowledge points.
    
    Args:
        knowledge_points_path: Path to JSONL file with knowledge points
        output_path: Path to save the prepared dataset
        questions_per_kp: Number of training samples per knowledge point
        state_path: Path to load existing state (for difficulty levels)
    
    Returns:
        Path to the prepared dataset
    """
    # Load knowledge point manager
    manager = KnowledgePointManager(
        knowledge_points_path=knowledge_points_path,
        questions_per_kp=questions_per_kp,
        state_save_path=state_path,
    )
    
    # Create training samples
    samples = []
    for kp, state in manager.knowledge_points.items():
        for _ in range(questions_per_kp):
            samples.append({
                "problem": f"KNOWLEDGE_POINT:{kp}|DIFFICULTY:{state.difficulty}",
                "answer": "",
                "knowledge_point": kp,
                "difficulty": state.difficulty,
            })
    
    # Shuffle samples
    import random
    random.shuffle(samples)
    
    # Save as JSONL
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Prepared {len(samples)} training samples")
    print(f"Saved to {output_path}")
    
    return output_path


def upload_to_huggingface(
    data_path: str,
    repo_name: str,
    token_path: str = "tokens.json",
    private: bool = True,
) -> str:
    """
    Upload prepared data to HuggingFace Hub.
    
    Args:
        data_path: Path to JSONL file
        repo_name: Repository name on HuggingFace
        token_path: Path to JSON file with HuggingFace token
        private: Whether to make the repo private
    
    Returns:
        URL to the uploaded dataset
    """
    # Login to HuggingFace
    try:
        with open(token_path, 'r') as f:
            token = json.load(f)['huggingface']
        login(token=token)
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")
        return None
    
    # Load data
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    # Create dataset
    dataset = Dataset.from_list(samples)
    dataset_dict = DatasetDict({"train": dataset})
    
    # Upload
    try:
        hf_username = os.getenv("HUGGINGFACENAME", "default_user")
        full_repo = f"{hf_username}/{repo_name}"
        dataset_dict.push_to_hub(full_repo, private=private)
        print(f"Uploaded to https://huggingface.co/datasets/{full_repo}")
        return full_repo
    except Exception as e:
        print(f"Error uploading: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for knowledge-point curriculum")
    
    parser.add_argument("--knowledge_points_path", type=str, required=True,
                        help="Path to knowledge points JSONL file")
    parser.add_argument("--output_path", type=str, default="kp_training_data.jsonl",
                        help="Path to save prepared data")
    parser.add_argument("--questions_per_kp", type=int, default=5,
                        help="Training samples per knowledge point")
    parser.add_argument("--state_path", type=str, default=None,
                        help="Path to existing state file")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace")
    parser.add_argument("--repo_name", type=str, default="kp_curriculum_data",
                        help="HuggingFace repo name for upload")
    
    args = parser.parse_args()
    
    # Prepare data
    output_path = prepare_challenger_data(
        knowledge_points_path=args.knowledge_points_path,
        output_path=args.output_path,
        questions_per_kp=args.questions_per_kp,
        state_path=args.state_path,
    )
    
    # Upload if requested
    if args.upload:
        upload_to_huggingface(output_path, args.repo_name)


if __name__ == "__main__":
    main()
