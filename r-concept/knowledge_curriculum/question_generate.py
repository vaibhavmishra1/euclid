"""
Question Generation for Knowledge-Set-Based Curriculum Learning

This script generates questions using the Challenger model based on:
- Knowledge SETS (groups of KPs from original questions)
- Difficulty levels 1-5 (initialized from original problem level)

Output includes set_id for tracking rewards back to knowledge sets.
"""

import argparse
import json
import os
import re
from typing import List, Dict, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .knowledge_manager import KnowledgeSetManager
from .prompts import build_knowledge_challenger_messages, extract_boxed_answer


def extract_question(text: str) -> str:
    """
    Extract question from <question>...</question> tags.
    
    Args:
        text: Generated text
    
    Returns:
        Extracted question or empty string
    """
    match = re.search(r'<question>(.*?)</question>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_question_and_answer(text: str) -> Tuple[str, str]:
    """
    Extract question and answer from challenger output.
    
    Expected format:
    <question>...</question>
    \\boxed{answer}
    
    Returns:
        Tuple of (question, answer)
    """
    question = extract_question(text)
    answer = extract_boxed_answer(text)
    
    return question, answer


def generate_questions_for_batch(
    model: LLM,
    tokenizer,
    batch: List[Tuple[int, List[str], int]],  # (set_id, kp_list, difficulty)
    sampling_params: SamplingParams,
) -> List[Dict]:
    """
    Generate questions for a batch of knowledge sets.
    
    Args:
        model: vLLM model
        tokenizer: Tokenizer
        batch: List of (set_id, knowledge_points_list, difficulty)
        sampling_params: Sampling parameters
    
    Returns:
        List of results with question, answer, set_id, etc.
    """
    # Build prompts for all items
    prompts = []
    metadata = []
    
    for set_id, kp_list, difficulty in batch:
        messages = build_knowledge_challenger_messages(kp_list, difficulty)
        
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = f"system: {messages[0]['content']}\nuser: {messages[1]['content']}"
        
        prompts.append(prompt)
        metadata.append({
            "set_id": set_id,
            "knowledge_points": kp_list,
            "difficulty": difficulty,
        })
    
    # Generate
    outputs = model.generate(prompts, sampling_params)
    
    # Process results
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        question, answer = extract_question_and_answer(generated_text)
        
        results.append({
            "set_id": metadata[i]["set_id"],
            "knowledge_points": metadata[i]["knowledge_points"],
            "difficulty": metadata[i]["difficulty"],
            "question": question,
            "answer": answer,
            "raw_output": generated_text,
            "valid": bool(question and answer),
            "score": 0,  # Required for evaluate.py to process this item
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate questions for knowledge-set curriculum")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to challenger model")
    parser.add_argument("--save_name", type=str, required=True,
                        help="Name for saving results")
    parser.add_argument("--suffix", type=str, default="0",
                        help="Suffix for output file (e.g., GPU ID)")
    parser.add_argument("--knowledge_points_path", type=str, required=True,
                        help="Path to knowledge points JSONL file")
    parser.add_argument("--state_path", type=str, default=None,
                        help="Path to load/save state")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Upper threshold for difficulty increase")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Lower threshold for difficulty decrease")
    parser.add_argument("--questions_per_set", type=int, default=5,
                        help="Questions to generate per knowledge set")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of items (0 = all)")
    
    args = parser.parse_args()
    
    # Storage path
    storage_path = os.getenv("STORAGE_PATH", "/tmp/rzero_storage")
    output_dir = f"{storage_path}/generated_question"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize knowledge set manager
    ks_manager = KnowledgeSetManager(
        knowledge_points_path=args.knowledge_points_path,
        alpha=args.alpha,
        beta=args.beta,
        questions_per_set=args.questions_per_set,
        state_save_path=args.state_path,
    )
    
    # Get training batch
    batch = ks_manager.get_training_batch()
    
    if args.limit > 0:
        batch = batch[:args.limit]
    
    print(f"Generating questions for {len(batch)} items")
    print(f"Knowledge sets: {len(ks_manager.knowledge_sets)}")
    print(f"Questions per set: {args.questions_per_set}")
    print(f"Difficulty range: 1-5")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.99,
    )
    
    # Generate in batches
    all_results = []
    for i in range(0, len(batch), args.batch_size):
        sub_batch = batch[i:i + args.batch_size]
        print(f"Processing batch {i//args.batch_size + 1}/{(len(batch) + args.batch_size - 1)//args.batch_size}")
        
        results = generate_questions_for_batch(model, tokenizer, sub_batch, sampling_params)
        all_results.extend(results)
    
    # Save results
    output_path = f"{output_dir}/{args.save_name}_{args.suffix}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Statistics
    valid_count = sum(1 for r in all_results if r["valid"])
    difficulty_counts = {}
    for r in all_results:
        d = r["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
    
    print(f"\nGeneration complete!")
    print(f"Total: {len(all_results)}")
    print(f"Valid: {valid_count} ({100*valid_count/len(all_results):.1f}%)")
    print(f"By difficulty: {dict(sorted(difficulty_counts.items()))}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
