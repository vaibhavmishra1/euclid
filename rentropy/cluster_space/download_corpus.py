#!/usr/bin/env python3
"""
Download math question corpus for cluster building.
Downloads OpenDataArena/MathLake dataset and filters for "Problem Solving" format.

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


def download_mathlake_dataset(output_dir: str):
    """
    Download OpenDataArena/MathLake dataset questions.
    Filters for format == "Problem Solving" and preserves all metadata fields.
    
    Fields preserved: source, id, question, subject, format, difficulty
    """
    print("Downloading OpenDataArena/MathLake dataset...")
    try:
        ds = load_dataset("OpenDataArena/MathLake", split="train")
        print(f"Loaded {len(ds)} total examples from MathLake")
        
        # Filter for "Problem Solving" format and extract all fields
        questions = []
        for item in ds:
            # Only keep items with format == "Problem Solving"
            if item.get("format") == "Problem Solving":
                question_data = {
                    "source": item.get("source", "mathlake"),
                    "id": item.get("id", ""),
                    "question": item.get("question", ""),
                    "subject": item.get("subject", ""),
                    "format": item.get("format", ""),
                    "difficulty": item.get("difficulty", ""),
                }
                
                # Only add if question text exists and is non-empty
                if question_data["question"] and len(question_data["question"].strip()) > 10:
                    questions.append(question_data)
        
        output_path = os.path.join(output_dir, "mathlake_questions.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(questions)} MathLake questions (format='Problem Solving') to {output_path}")
        
        # Print statistics
        if questions:
            from collections import Counter
            subjects = [q.get("subject", "unknown") for q in questions]
            difficulties = [q.get("difficulty", "unknown") for q in questions]
            sources = [q.get("source", "unknown") for q in questions]
            
            print(f"\nFiltered statistics:")
            print(f"  Questions by subject:")
            for subject, count in Counter(subjects).most_common(10):
                print(f"    {subject}: {count}")
            print(f"  Questions by difficulty:")
            for difficulty, count in Counter(difficulties).most_common():
                print(f"    {difficulty}: {count}")
            print(f"  Questions by source:")
            for source, count in Counter(sources).most_common(10):
                print(f"    {source}: {count}")
        
        return questions
    except Exception as e:
        print(f"Failed to download MathLake dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description="Download MathLake corpus for cluster building")
    parser.add_argument("--output_dir", type=str, default="./corpus")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download MathLake dataset
    all_questions = download_mathlake_dataset(args.output_dir)
    
    # Save combined corpus (all questions from MathLake)
    combined_path = os.path.join(args.output_dir, "all_questions.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    print(f"\nTotal: {len(all_questions)} questions saved to {combined_path}")
    
    print("\nNext step: Run build_clusters.py to create cluster centroids")
    print(f"  python build_clusters.py --corpus_file {combined_path} --output_dir ./cluster_data")


if __name__ == "__main__":
    main()
