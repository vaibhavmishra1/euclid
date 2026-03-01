#!/usr/bin/env python3
"""
Generate new questions using few-shot prompting based on cluster membership.

Algorithm:
1. Sample a cluster based on its occurrence probability
2. Randomly select 3 questions from that cluster
3. Prompt the model to generate a new question with similar concept but different details
4. Repeat for desired number of samples
5. Save generated questions with cluster information
"""
import os
import sys
import json
import argparse
import numpy as np
import regex as re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "rentropy" / "R-Zero-main"))

# Lazy import vLLM to allow checking args first
LLM = None
SamplingParams = None
AutoTokenizer = None


def load_vllm():
    """Lazy load vLLM modules."""
    global LLM, SamplingParams, AutoTokenizer
    if LLM is None:
        from vllm import LLM as _LLM, SamplingParams as _SamplingParams
        from transformers import AutoTokenizer as _AutoTokenizer
        LLM = _LLM
        SamplingParams = _SamplingParams
        AutoTokenizer = _AutoTokenizer


def extract_boxed(text):
    """Extract content from \\boxed{...} with proper brace matching."""
    results, i = [], 0
    prefix = r'\boxed{'
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break   # no more \boxed{…}

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        results.append(text[start + plen : j - 1])
        i = j

    return results


def load_questions_with_clusters(json_path: str) -> List[Dict]:
    """
    Load questions with cluster IDs from JSON file.
    
    Args:
        json_path: Path to JSON file with questions and cluster IDs
    
    Returns:
        List of question dictionaries with 'question', 'answer', 'cluster_id' fields
    """
    print(f"Loading questions from {json_path}...")
    with open(json_path, 'r') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Verify all questions have cluster_id
    questions_with_cluster = [q for q in questions if 'cluster_id' in q and 'question' in q]
    if len(questions_with_cluster) < len(questions):
        print(f"Warning: {len(questions) - len(questions_with_cluster)} questions missing cluster_id or question field")
    
    return questions_with_cluster


def filter_long_questions(questions: List[Dict], max_chars: int = 800) -> List[Dict]:
    """
    Filter out questions that are too long to avoid prompt length issues.
    
    Args:
        questions: List of question dictionaries
        max_chars: Maximum number of characters allowed (default: 800)
    
    Returns:
        List of questions that are not too long
    """
    filtered = [q for q in questions if len(q.get('question', '')) <= max_chars]
    if len(filtered) < len(questions):
        print(f"Filtered out {len(questions) - len(filtered)} long questions (>{max_chars} chars)")
    return filtered


def organize_by_cluster(questions: List[Dict], max_question_chars: int = 800) -> Dict[int, List[Dict]]:
    """
    Organize questions by cluster ID and filter out very long questions.
    
    Args:
        questions: List of question dictionaries with cluster_id field
        max_question_chars: Maximum characters per question to include
    
    Returns:
        Dictionary mapping cluster_id -> list of questions (excluding very long ones)
    """
    # First filter out very long questions
    filtered_questions = filter_long_questions(questions, max_question_chars)
    print(f"Using {len(filtered_questions)}/{len(questions)} questions after length filtering")
    
    clusters = defaultdict(list)
    for q in filtered_questions:
        cluster_id = q['cluster_id']
        clusters[cluster_id].append(q)
    
    print(f"Organized into {len(clusters)} clusters")
    
    # Print cluster size statistics
    cluster_sizes = [len(qs) for qs in clusters.values()]
    if cluster_sizes:
        print(f"Cluster size stats - Min: {min(cluster_sizes)}, Max: {max(cluster_sizes)}, "
              f"Mean: {np.mean(cluster_sizes):.2f}, Median: {np.median(cluster_sizes):.2f}")
    
    return dict(clusters)


def compute_cluster_probabilities(clusters: Dict[int, List[Dict]]) -> Tuple[List[int], List[float]]:
    """
    Compute sampling probabilities for each cluster based on occurrence frequency.
    
    Args:
        clusters: Dictionary mapping cluster_id -> list of questions
    
    Returns:
        Tuple of (cluster_ids, probabilities)
    """
    cluster_ids = list(clusters.keys())
    cluster_sizes = np.array([len(clusters[cid]) for cid in cluster_ids])
    
    # Normalize to get probabilities
    probabilities = cluster_sizes / cluster_sizes.sum()
    
    return cluster_ids, probabilities.tolist()


def build_few_shot_prompt(example_questions: List[Dict], tokenizer) -> List[Dict]:
    """
    Build a few-shot chat prompt with example questions from the same cluster.
    Uses the same format as question_generate.py but adds few-shot examples.
    
    Args:
        example_questions: List of 2-3 example questions from the same cluster
        tokenizer: Tokenizer for chat template formatting
    
    Returns:
        Chat messages list for apply_chat_template
    """
    # Build few-shot examples string
    examples_text = "Here are some example problems on related mathematical concepts:\n\n"
    for i, q in enumerate(example_questions, 1):
        question_text = q['question'].strip()
        examples_text += f"Example {i}:\n{question_text}\n\n"
    
    # System message (same as question_generate.py)
    system_content = (
        "You are an expert competition-math problem setter.\n"
        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
        "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
        "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
        "Avoid re-using textbook clichés or famous contest problems.\n"
        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
        "<question>\n"
        "{The full problem statement on one or more lines}\n"
        "</question>\n\n"
        r"\boxed{final_answer}"
        "\n\n"
        "Do NOT output anything else—no explanations, no extra markup."
    )
    
    # User message with few-shot examples
    user_content = (
        f"{examples_text}"
        "Based on the mathematical concepts demonstrated in the examples above, generate ONE NEW, more challenging "
        "problem that:\n"
        "1. Explores the same underlying mathematical concept or theory\n"
        "2. Is distinctly different from the examples (not just changing numbers)\n\n"
        "Remember to format the output exactly as instructed: <question> tags and \\boxed{} answer only."
    )
    
    chat = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    return chat


def generate_few_shot_questions(
    model_path: str,
    clusters: Dict[int, List[Dict]],
    num_samples: int,
    output_path: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    num_examples: int = 3,
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
    max_model_len: int = 32768,
) -> List[Dict]:
    """
    Generate new questions using few-shot prompting based on cluster membership.
    
    Args:
        model_path: Path to the question generation model
        clusters: Dictionary mapping cluster_id -> list of questions
        num_samples: Number of new questions to generate
        output_path: Path to save generated questions
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        num_examples: Number of example questions to include in prompt (default: 3)
        batch_size: Batch size for generation
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum model context length (default: 32768 for Qwen3)
    
    Returns:
        List of generated questions with cluster IDs
    """
    load_vllm()
    
    # Compute cluster probabilities
    cluster_ids, probabilities = compute_cluster_probabilities(clusters)
    
    print(f"\nGenerating {num_samples} questions using few-shot prompting...")
    print(f"Model: {model_path}")
    print(f"Temperature: {temperature}")
    print(f"Examples per prompt: {num_examples}")
    print(f"Batch size: {batch_size}")
    print(f"Max model length: {max_model_len}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    
    # Generate samples
    generated_questions = []
    prompts_batch = []
    cluster_ids_batch = []
    example_questions_batch = []
    
    # Filter out clusters with fewer than num_examples questions
    valid_cluster_ids = [cid for cid in cluster_ids if len(clusters[cid]) >= num_examples]
    valid_probabilities = [probabilities[cluster_ids.index(cid)] for cid in valid_cluster_ids]
    # Renormalize probabilities
    valid_probabilities = np.array(valid_probabilities)
    valid_probabilities = valid_probabilities / valid_probabilities.sum()
    
    if len(valid_cluster_ids) == 0:
        raise ValueError(f"No clusters have at least {num_examples} questions!")
    
    print(f"Using {len(valid_cluster_ids)} clusters (with >= {num_examples} questions)")
    
    # Statistics tracking
    parse_failures = 0
    
    with tqdm(total=num_samples, desc="Generating questions") as pbar:
        for i in range(num_samples):
            # Sample a cluster based on probabilities
            sampled_cluster_id = np.random.choice(valid_cluster_ids, p=valid_probabilities)
            
            # Get random sample of num_examples questions from this cluster
            cluster_questions = clusters[sampled_cluster_id]
            example_questions = np.random.choice(
                cluster_questions, 
                size=min(num_examples, len(cluster_questions)), 
                replace=False
            ).tolist()
            
            # Build chat prompt
            chat = build_few_shot_prompt(example_questions, tokenizer)
            
            # Apply chat template
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    chat, 
                    tokenize=False,
                    add_generation_prompt=True, 
                    add_special_tokens=True
                )
            else:
                # Fallback for models without chat template
                prompt = "system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"]
            
            prompts_batch.append(prompt)
            cluster_ids_batch.append(sampled_cluster_id)
            example_questions_batch.append(example_questions)
            
            # Generate when batch is full or at the end
            if len(prompts_batch) >= batch_size or i == num_samples - 1:
                # Generate questions
                outputs = llm.generate(prompts_batch, sampling_params)
                
                # Extract generated questions
                for j, output in enumerate(outputs):
                    response = output.outputs[0].text
                    
                    try:
                        # Extract question from <question> tags
                        questions_found = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
                        # Extract answer from \boxed{}
                        answers_found = extract_boxed(response)
                        
                        if questions_found and answers_found:
                            question_text = questions_found[-1].strip()
                            answer_text = answers_found[-1].strip()
                            score = 0
                        else:
                            # Parsing failed, use raw response
                            question_text = response.strip()
                            answer_text = ""
                            score = -1
                            parse_failures += 1
                    except Exception as e:
                        # Parsing error, use raw response
                        question_text = response.strip()
                        answer_text = ""
                        score = -1
                        parse_failures += 1
                    
                    generated_questions.append({
                        "question": question_text,
                        "answer": answer_text,
                        "score": score,
                        "cluster_id": int(cluster_ids_batch[j]),
                        "source_cluster": int(cluster_ids_batch[j]),
                        "example_questions": [eq["question"] for eq in example_questions_batch[j]],
                        "generation_method": "few_shot_cluster_based"
                    })
                
                pbar.update(len(prompts_batch))
                
                # Clear batch
                prompts_batch = []
                cluster_ids_batch = []
                example_questions_batch = []
    
    # Save generated questions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(generated_questions, f, indent=2)
    
    print(f"\nSaved {len(generated_questions)} generated questions to {output_path}")
    
    # Print statistics
    cluster_distribution = Counter([q["cluster_id"] for q in generated_questions])
    successful_parses = len([q for q in generated_questions if q["score"] != -1])
    print(f"\nGenerated questions span {len(cluster_distribution)} clusters")
    print(f"Top 5 clusters: {cluster_distribution.most_common(5)}")
    print(f"Successfully parsed: {successful_parses}/{len(generated_questions)} ({100*successful_parses/len(generated_questions):.1f}%)")
    if parse_failures > 0:
        print(f"Parse failures: {parse_failures} (questions saved with score=-1)")
    
    return generated_questions


def main():
    parser = argparse.ArgumentParser(
        description="Generate new questions using few-shot prompting based on cluster membership"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to question generation model"
    )
    parser.add_argument(
        "--questions_with_clusters",
        type=str,
        required=True,
        help="Path to JSON file with questions and cluster IDs"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of new questions to generate (default: 10000)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for generated questions (default: auto-generated based on input)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2,
        help="Number of example questions to include in each prompt (default: 2)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per question (default: 1024)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation (default: 32)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Maximum model context length (default: 32768 for Qwen3, can be 4096, 8192, 16384, or 32768)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load questions with clusters
    questions = load_questions_with_clusters(args.questions_with_clusters)
    
    # Organize by cluster (filters out long questions automatically)
    clusters = organize_by_cluster(questions, max_question_chars=800)
    
    # Determine output path
    if args.output_path is None:
        input_path = Path(args.questions_with_clusters)
        model_name = Path(args.model).name
        output_dir = input_path.parent.parent / "generated_question_few_shot"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_few_shot_{model_name}.json"
    else:
        output_path = args.output_path
    
    print(f"Output will be saved to: {output_path}")
    
    # Generate questions
    generated_questions = generate_few_shot_questions(
        model_path=args.model,
        clusters=clusters,
        num_samples=args.num_samples,
        output_path=output_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Total questions generated: {len(generated_questions)}")
    print(f"Output saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
