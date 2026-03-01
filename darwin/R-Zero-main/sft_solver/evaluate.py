#!/usr/bin/env python3
"""
Evaluate a trained SFT solver model on the validation set.

Usage:
    python evaluate.py \
        --model_path ./outputs/sft_solver_darwin_iter2/final_model \
        --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --data_path /path/to/dataset.json \
        --num_samples 100
"""

import argparse
import json
import re
import torch
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def extract_answer(text: str) -> str:
    """
    Extract the final answer from the model's response.
    Looks for \boxed{...} format commonly used in math problems.
    """
    # Try to find \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Try to find "The answer is ..."
    answer_match = re.search(r'[Tt]he answer is[:\s]+([^\n\.]+)', text)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Try to find "Final answer: ..."
    final_match = re.search(r'[Ff]inal answer[:\s]+([^\n\.]+)', text)
    if final_match:
        return final_match.group(1).strip()
    
    # Return last line as fallback
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove extra whitespace
    answer = answer.strip()
    # Remove dollar signs
    answer = answer.replace('$', '')
    # Remove backslashes (LaTeX commands)
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    # Remove extra spaces
    answer = ' '.join(answer.split())
    return answer.lower()


def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return pred_norm == gold_norm


def format_prompt(question: str) -> str:
    """Format the prompt for inference."""
    return f"""<|im_start|>system
You are a helpful assistant that solves mathematical problems step by step.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""


def evaluate_model(
    model,
    tokenizer,
    data: List[Dict],
    num_samples: int = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate the model on a dataset.
    
    Returns:
        Dictionary containing evaluation metrics
    """
    if num_samples:
        data = data[:num_samples]
    
    print(f"\nEvaluating on {len(data)} examples...")
    
    results = []
    correct = 0
    total = 0
    
    for item in tqdm(data, desc="Evaluating"):
        question = item.get('question', '')
        gold_answer = item.get('openai_answer', '')
        
        if not question or not gold_answer:
            continue
        
        # Format prompt
        prompt = format_prompt(question)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (remove prompt)
        response = generated_text[len(prompt):].strip()
        
        # Extract answer
        pred_answer = extract_answer(response)
        
        # Check correctness
        is_correct = answers_match(pred_answer, gold_answer)
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        results.append({
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': pred_answer,
            'response': response,
            'correct': is_correct,
            'cluster_id': item.get('cluster_id', -1),
            'score': item.get('score', 0.0),
        })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate per-cluster metrics
    cluster_metrics = {}
    for result in results:
        cluster_id = result['cluster_id']
        if cluster_id not in cluster_metrics:
            cluster_metrics[cluster_id] = {'correct': 0, 'total': 0}
        cluster_metrics[cluster_id]['total'] += 1
        if result['correct']:
            cluster_metrics[cluster_id]['correct'] += 1
    
    # Calculate per-cluster accuracy
    for cluster_id in cluster_metrics:
        cluster_data = cluster_metrics[cluster_id]
        cluster_data['accuracy'] = cluster_data['correct'] / cluster_data['total']
    
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'cluster_metrics': cluster_metrics,
        'results': results,
    }
    
    return metrics


def print_metrics(metrics: Dict):
    """Print evaluation metrics."""
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Overall Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    
    # Cluster-wise accuracy
    print(f"\nCluster-wise Performance:")
    cluster_metrics = metrics['cluster_metrics']
    
    if cluster_metrics:
        # Sort by cluster ID
        sorted_clusters = sorted(cluster_metrics.items())
        
        # Print top 10 clusters by count
        print("\nTop 10 clusters by sample count:")
        sorted_by_count = sorted(
            sorted_clusters,
            key=lambda x: x[1]['total'],
            reverse=True
        )[:10]
        
        for cluster_id, data in sorted_by_count:
            print(f"  Cluster {cluster_id}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
        
        # Calculate variance
        accuracies = [data['accuracy'] for data in cluster_metrics.values()]
        mean_acc = sum(accuracies) / len(accuracies)
        variance = sum((acc - mean_acc) ** 2 for acc in accuracies) / len(accuracies)
        std_dev = variance ** 0.5
        
        print(f"\nCluster Statistics:")
        print(f"  Number of clusters: {len(cluster_metrics)}")
        print(f"  Mean cluster accuracy: {mean_acc:.2%}")
        print(f"  Std dev: {std_dev:.2%}")
    
    # Show some example predictions
    print("\n" + "=" * 80)
    print("Example Predictions (first 3)")
    print("=" * 80)
    
    for i, result in enumerate(metrics['results'][:3], 1):
        print(f"\nExample {i}:")
        print(f"Question: {result['question'][:100]}...")
        print(f"Gold Answer: {result['gold_answer']}")
        print(f"Predicted Answer: {result['predicted_answer']}")
        print(f"Correct: {'✓' if result['correct'] else '✗'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT Solver Model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (LoRA adapter)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or name of the base model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the JSON dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Where to save detailed results"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Num samples: {args.num_samples or 'all'}")
    print("=" * 80)
    
    # Load data
    print("\n[1/4] Loading dataset...")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    # Filter for openai_match == 1
    data = [item for item in data if item.get('openai_match', 0) == 1]
    print(f"✓ Loaded {len(data)} examples")
    
    # Load model and tokenizer
    print("\n[2/4] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    print(f"✓ Model loaded on {device}")
    
    # Evaluate
    print("\n[3/4] Running evaluation...")
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        data=data,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    
    # Print results
    print_metrics(metrics)
    
    # Save results
    print(f"\n[4/4] Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        # Remove full responses to save space
        metrics_to_save = metrics.copy()
        metrics_to_save['results'] = [
            {k: v for k, v in r.items() if k != 'response'}
            for r in metrics['results']
        ]
        json.dump(metrics_to_save, f, indent=2)
    print(f"✓ Results saved")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
