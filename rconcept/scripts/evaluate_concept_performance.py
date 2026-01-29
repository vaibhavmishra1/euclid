#!/usr/bin/env python3
"""
Evaluate model performance on held-out questions for each concept.
This is called after each concept iteration to track improvement.
"""

import argparse
import json
import os
import vllm
from transformers import AutoTokenizer
from pathlib import Path
from typing import Dict, List
from mathruler.grader import extract_boxed_content, grade_answer
import stopit

STORAGE_PATH = os.getenv("STORAGE_PATH", "")


@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """Wrapper for grade_answer with timeout."""
    return grade_answer(res1, res2)


def evaluate_concept_questions(
    model_path: str,
    concept_name: str,
    questions: List[str],
    num_rollouts: int = 9,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate model on questions for a specific concept using majority voting.
    
    Returns:
        {
            "concept": concept_name,
            "num_questions": len(questions),
            "avg_score": float,
            "scores": List[float],  # per-question scores
            "questions_evaluated": int
        }
    """
    print(f"\n[EvaluateConcept] Evaluating '{concept_name}' on {len(questions)} questions...")
    
    if not questions:
        return {
            "concept": concept_name,
            "num_questions": 0,
            "avg_score": 0.0,
            "scores": [],
            "questions_evaluated": 0,
        }
    
    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = vllm.LLM(
        model=model_path,
        tokenizer=model_path,
        gpu_memory_utilization=0.85,
    )
    
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        stop_token_ids=[tokenizer.eos_token_id],
        n=num_rollouts,
    )
    
    # Build prompts
    chats = [
        [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q}
        ]
        for q in questions
    ]
    
    if tokenizer.chat_template:
        prompts = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True)
            for chat in chats
        ]
    else:
        prompts = [
            "system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"]
            for chat in chats
        ]
    
    # Generate responses
    print(f"[EvaluateConcept] Generating {num_rollouts} rollouts for each question...")
    responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
    
    # Process and score
    scores = []
    questions_evaluated = 0
    
    for response, question in zip(responses, questions):
        try:
            # Extract boxed answers from all rollouts
            results = [extract_boxed_content(output.text) for output in response.outputs]
            results = [res for res in results if res]  # Filter empty
            
            if not results:
                continue
            
            # Group equivalent answers (majority voting)
            answer_counts = {}
            for result in results:
                matched = False
                for existing_answer in list(answer_counts.keys()):
                    # Cheap string comparison
                    if result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                        answer_counts[existing_answer] += 1
                        matched = True
                        break
                    
                    # Expensive grader check
                    match_1 = grade_answer_with_timeout(result, existing_answer, timeout=10)
                    if match_1 == 'TIMED_OUT':
                        continue
                    if match_1:
                        answer_counts[existing_answer] += 1
                        matched = True
                        break
                    
                    match_2 = grade_answer_with_timeout(existing_answer, result, timeout=10)
                    if match_2 == 'TIMED_OUT':
                        continue
                    if match_2:
                        answer_counts[existing_answer] += 1
                        matched = True
                        break
                
                if not matched:
                    answer_counts[result] = 1
            
            if not answer_counts:
                continue
            
            # Compute majority score
            majority_answer = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_answer]
            score = max_count / len(results)
            
            scores.append(score)
            questions_evaluated += 1
            
        except Exception as e:
            print(f"[EvaluateConcept] Error processing question: {e}")
            continue
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    result = {
        "concept": concept_name,
        "num_questions": len(questions),
        "avg_score": avg_score,
        "scores": scores,
        "questions_evaluated": questions_evaluated,
    }
    
    print(f"[EvaluateConcept] '{concept_name}': avg_score={avg_score:.3f} ({questions_evaluated}/{len(questions)} questions)")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model to evaluate")
    parser.add_argument("--concept_mapping", required=True, help="Path to concept_to_questions.json")
    parser.add_argument("--concept_sets", required=True, help="Path to concept_sets.json")
    parser.add_argument("--iteration", type=int, required=True, help="Current iteration number (0-indexed)")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--num_rollouts", type=int, default=9, help="Number of rollouts per question")
    args = parser.parse_args()
    
    # Load concept mapping
    with open(args.concept_mapping, 'r') as f:
        concept_to_questions = json.load(f)
    
    # Load concept sets
    with open(args.concept_sets, 'r') as f:
        concept_sets = json.load(f)
    
    # Extract concept names from concept_sets (they might not have "concept:" prefix)
    concept_names_map = {}
    for cset in concept_sets:
        for concept in cset["concepts"]:
            # Try multiple variations to find the matching key
            possible_keys = [
                f"concept:{concept}",  # Add prefix
                concept,  # Exact match
                concept.lower(),  # Lowercase
                f"concept:{concept.lower()}",  # Lowercase with prefix
            ]
            
            found_key = None
            for key in possible_keys:
                if key in concept_to_questions:
                    found_key = key
                    break
            
            if found_key:
                concept_names_map[concept] = found_key
            else:
                print(f"[EvaluateConcept] Warning: Could not find concept '{concept}' in mapping. Available keys (first 10): {list(concept_to_questions.keys())[:10]}")
    
    # Evaluate each concept
    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cset in concept_sets:
        for concept in cset["concepts"]:
            concept_key = concept_names_map.get(concept, f"concept:{concept}")
            
            if concept_key not in concept_to_questions:
                print(f"[EvaluateConcept] Warning: Concept '{concept}' not found in mapping, skipping")
                continue
            
            questions = concept_to_questions[concept_key]
            
            # Evaluate
            result = evaluate_concept_questions(
                model_path=args.model_path,
                concept_name=concept,
                questions=questions,
                num_rollouts=args.num_rollouts,
            )
            result["iteration"] = args.iteration
            results.append(result)
    
    # Save results
    output_file = output_dir / f"iteration_{args.iteration}_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[EvaluateConcept] Saved results to {output_file}")
    
    # Also append to cumulative results file
    cumulative_file = output_dir / "cumulative_evaluation.json"
    if cumulative_file.exists():
        with open(cumulative_file, 'r') as f:
            cumulative = json.load(f)
    else:
        cumulative = []
    
    cumulative.extend(results)
    with open(cumulative_file, 'w') as f:
        json.dump(cumulative, f, indent=2)
    
    print(f"[EvaluateConcept] Updated cumulative results in {cumulative_file}")


if __name__ == "__main__":
    main()
