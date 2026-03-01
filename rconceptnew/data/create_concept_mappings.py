#!/usr/bin/env python3
"""
Create two mapping files from seed_concepts_canonical.jsonl:
1. concept_to_questions.json: Maps each unique concept to list of questions
2. concept_set_to_questions.json: Maps each unique concept set to list of questions
"""

import json
import os
from collections import defaultdict
from datasets import load_dataset
from pathlib import Path

def parse_problem_id(problem_id: str):
    """Parse problem_id like 'hendrycks_math/number_theory/train/0'"""
    parts = problem_id.split("/")
    if len(parts) == 4 and parts[0] == "hendrycks_math":
        return parts[1], parts[2], int(parts[3])  # domain, split, index
    return None, None, None

def load_math_questions():
    """Load all MATH questions into a dictionary keyed by (domain, split, index)"""
    print("Loading MATH dataset...")
    questions_dict = {}
    
    # Load number_theory train split (based on the JSONL file)
    try:
        ds = load_dataset("EleutherAI/hendrycks_math", "number_theory", split="train")
        for i, ex in enumerate(ds):
            questions_dict[("number_theory", "train", i)] = ex["problem"]
        print(f"Loaded {len(questions_dict)} questions from number_theory/train")
    except Exception as e:
        print(f"Warning: Could not load MATH dataset: {e}")
        print("Will use problem_id as question identifier")
    
    return questions_dict

def main():
    # Paths
    input_file = Path("/Users/vaibhav/Desktop/brahma/tree/euclid/expv1/expv1_0/output/concepts/cleaned/seed_concepts_canonical.jsonl")
    output_dir = Path("/Users/vaibhav/Desktop/brahma/tree/euclid/R-Zero/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file1 = output_dir / "concept_to_questions.json"
    output_file2 = output_dir / "concept_set_to_questions.json"
    
    # Load MATH questions
    questions_dict = load_math_questions()
    use_actual_questions = len(questions_dict) > 0
    
    # Data structures
    concept_to_questions = defaultdict(list)
    concept_set_to_questions = defaultdict(list)
    
    # Parse JSONL file
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                problem_id = data["problem_id"]
                canonical_concepts = data["canonical_concepts"]
                
                # Extract concept keys
                concepts = [c["canonical_key"] for c in canonical_concepts]
                concept_set = tuple(sorted(concepts))  # Use sorted tuple as key for concept set
                
                # Get question text
                if use_actual_questions:
                    domain, split, index = parse_problem_id(problem_id)
                    if domain and split is not None and index is not None:
                        question = questions_dict.get((domain, split, index))
                        if question:
                            question_text = question
                        else:
                            question_text = problem_id  # Fallback
                    else:
                        question_text = problem_id
                else:
                    question_text = problem_id
                
                # Add to concept mapping
                for concept in concepts:
                    if question_text not in concept_to_questions[concept]:
                        concept_to_questions[concept].append(question_text)
                
                # Add to concept set mapping
                if question_text not in concept_set_to_questions[concept_set]:
                    concept_set_to_questions[concept_set].append(question_text)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    # Convert defaultdicts to regular dicts and sort by value list size (descending)
    concept_to_questions = dict(concept_to_questions)
    concept_set_to_questions = {str(k): v for k, v in concept_set_to_questions.items()}
    
    # Sort by length of value list (descending)
    concept_to_questions_sorted = dict(
        sorted(concept_to_questions.items(), key=lambda x: len(x[1]), reverse=True)
    )
    concept_set_to_questions_sorted = dict(
        sorted(concept_set_to_questions.items(), key=lambda x: len(x[1]), reverse=True)
    )
    
    # Save files
    print(f"\nWriting {output_file1}...")
    with open(output_file1, 'w', encoding='utf-8') as f:
        json.dump(concept_to_questions_sorted, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(concept_to_questions_sorted)} unique concepts (sorted by question count)")
    
    print(f"\nWriting {output_file2}...")
    with open(output_file2, 'w', encoding='utf-8') as f:
        json.dump(concept_set_to_questions_sorted, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(concept_set_to_questions_sorted)} unique concept sets (sorted by question count)")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total unique concepts: {len(concept_to_questions_sorted)}")
    print(f"Total unique concept sets: {len(concept_set_to_questions_sorted)}")
    
    # Show top concepts by question count
    print("\n=== Top 5 Concepts by Question Count ===")
    for i, (concept, questions) in enumerate(list(concept_to_questions_sorted.items())[:5]):
        print(f"{i+1}. {concept}: {len(questions)} questions")
    
    print("\n=== Top 5 Concept Sets by Question Count ===")
    for i, (concept_set, questions) in enumerate(list(concept_set_to_questions_sorted.items())[:5]):
        print(f"{i+1}. {concept_set}: {len(questions)} questions")

if __name__ == "__main__":
    main()
