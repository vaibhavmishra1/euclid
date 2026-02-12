#!/usr/bin/env python3
"""
Balanced cluster-aware question generation from multiple models.

This script:
1. Loads n questioner models
2. Iteratively generates questions from each model (100 per model per iteration)
3. Classifies questions into clusters
4. Maintains cluster frequency balance: max 20 questions per cluster
5. Stops when all clusters have 20 questions OR max 50 iterations reached

Usage:
    python balanced_cluster_generation.py \
        --models vibhuiitj/qwen3-4b-questioner-v1 vibhuiitj/qwen3-4b-questioner-v2 \
        --centroids_dataset vibhuiitj/math_clusters_new \
        --output_file balanced_questions.json \
        --max_per_cluster 20 \
        --questions_per_model 100 \
        --max_iterations 50
"""

import argparse
import json
import numpy as np
import os
import sys
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../R-Zero-main'))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("ERROR: transformers and torch required. Install with: pip install transformers torch")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers required. Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets required. Install with: pip install datasets")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


# ===========================================================================
# CLUSTER ASSIGNMENT
# ===========================================================================

def load_centroids(centroids_dataset: str, centroids_file: str = "centroids.npy") -> np.ndarray:
    """Load pre-computed cluster centroids from HuggingFace."""
    print(f"[Centroids] Loading from {centroids_dataset}/{centroids_file}")
    
    try:
        if hf_hub_download is None:
            raise ImportError("huggingface_hub not available")
        
        centroids_path = hf_hub_download(
            repo_id=centroids_dataset,
            filename=centroids_file,
            repo_type="dataset"
        )
        centroids = np.load(centroids_path)
        print(f"[Centroids] Loaded {centroids.shape[0]} centroids with dimension {centroids.shape[1]}")
        return centroids
    except Exception as e:
        print(f"[Centroids] ERROR: {e}")
        sys.exit(1)


def assign_to_cluster(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign embeddings to nearest cluster centroids."""
    # Normalize centroids
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normalized = centroids / (centroid_norms + 1e-12)
    
    # Compute cosine similarity
    similarities = embeddings @ centroids_normalized.T
    labels = np.argmax(similarities, axis=1)
    return labels


# ===========================================================================
# QUESTION GENERATION
# ===========================================================================

def load_models(model_names: List[str], device: str = "cuda:0") -> List[Tuple]:
    """Load multiple questioner models."""
    models = []
    print(f"\n[Models] Loading {len(model_names)} models on {device}")
    
    for model_name in model_names:
        print(f"  Loading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device
            )
            model.eval()
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            models.append((model, tokenizer, model_name))
            print(f"    ✓ Loaded")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            continue
    
    if not models:
        print("[ERROR] No models loaded successfully!")
        sys.exit(1)
    
    print(f"[Models] {len(models)} models ready\n")
    return models


def generate_questions_from_model(
    model, 
    tokenizer, 
    model_name: str,
    num_questions: int = 100,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> List[Dict]:
    """Generate questions from a single model."""
    
    # System and user prompts for questioner
    system_prompt = (
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
    
    user_prompt = (
        "Generate one new, challenging reasoning question now. "
        "Remember to format the output exactly as instructed."
    )
    
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True
        )
    else:
        prompt = "system: " + system_prompt + '\n' + "user: " + user_prompt
    
    # Generate in batches
    batch_size = 10
    all_questions = []
    
    print(f"    Generating {num_questions} questions from {os.path.basename(model_name)}...")
    
    for batch_start in range(0, num_questions, batch_size):
        batch_end = min(batch_start + batch_size, num_questions)
        batch_count = batch_end - batch_start
        
        inputs = tokenizer([prompt] * batch_count, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and parse
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract question and answer
            import re
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            
            # Extract boxed content
            def extract_boxed(text):
                results, i = [], 0
                prefix = r'\boxed{'
                plen = len(prefix)
                while True:
                    start = text.find(prefix, i)
                    if start == -1:
                        break
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
            
            answers = extract_boxed(response)
            
            if questions and answers:
                all_questions.append({
                    "problem": questions[-1].strip(),
                    "answer": answers[-1].strip(),
                    "source_model": model_name,
                    "raw_response": response
                })
    
    print(f"    ✓ Generated {len(all_questions)}/{num_questions} valid questions")
    return all_questions


# ===========================================================================
# MAIN LOOP
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Balanced cluster-aware question generation")
    
    parser.add_argument("--models", nargs="+", required=True,
                       help="List of questioner model names")
    parser.add_argument("--centroids_dataset", type=str, default="vibhuiitj/math_clusters_new",
                       help="HF dataset with cluster centroids")
    parser.add_argument("--centroids_file", type=str, default="centroids.npy",
                       help="Centroids file name")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                       help="Embedding model for cluster assignment")
    parser.add_argument("--output_file", type=str, default="balanced_questions.json",
                       help="Output JSON file for selected questions")
    parser.add_argument("--max_per_cluster", type=int, default=20,
                       help="Maximum questions per cluster")
    parser.add_argument("--questions_per_model", type=int, default=100,
                       help="Questions to generate per model per iteration")
    parser.add_argument("--max_iterations", type=int, default=50,
                       help="Maximum iterations")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for generation models")
    parser.add_argument("--embedding_device", type=str, default="cuda:0",
                       help="Device for embedding model")
    
    args = parser.parse_args()
    
    # Load centroids
    centroids = load_centroids(args.centroids_dataset, args.centroids_file)
    num_clusters = len(centroids)
    
    # Initialize frequency array
    cluster_freq = np.zeros(num_clusters, dtype=int)
    
    # Load embedding model
    print(f"[Embedding] Loading {args.embedding_model} on {args.embedding_device}")
    embed_model = SentenceTransformer(args.embedding_model, trust_remote_code=True, device=args.embedding_device)
    
    # Load generation models
    models = load_models(args.models, device=args.device)
    
    # Storage for selected questions
    selected_questions = []
    
    # Main loop
    print(f"\n{'='*70}")
    print(f"STARTING BALANCED GENERATION")
    print(f"  Target: {args.max_per_cluster} questions × {num_clusters} clusters = {args.max_per_cluster * num_clusters} total")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"{'='*70}\n")
    
    for iteration in range(args.max_iterations):
        print(f"\n[Iteration {iteration + 1}/{args.max_iterations}]")
        
        # Check stopping condition
        min_freq = cluster_freq.min()
        if min_freq >= args.max_per_cluster:
            print(f"\n✓ All clusters have {args.max_per_cluster}+ questions. Stopping.")
            break
        
        # Status
        filled_clusters = (cluster_freq >= args.max_per_cluster).sum()
        print(f"  Progress: {filled_clusters}/{num_clusters} clusters filled")
        print(f"  Current distribution - min: {cluster_freq.min()}, max: {cluster_freq.max()}, mean: {cluster_freq.mean():.1f}")
        
        # Generate from all models
        iteration_questions = []
        for model, tokenizer, model_name in models:
            questions = generate_questions_from_model(
                model, tokenizer, model_name,
                num_questions=args.questions_per_model,
            )
            iteration_questions.extend(questions)
        
        print(f"  Total generated: {len(iteration_questions)} questions")
        
        if not iteration_questions:
            print("  ⚠ No questions generated, skipping iteration")
            continue
        
        # Embed and classify
        print(f"  Embedding and classifying...")
        problem_texts = [q["problem"] for q in iteration_questions]
        embeddings = embed_model.encode(
            problem_texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        labels = assign_to_cluster(embeddings, centroids)
        
        # Assign cluster IDs
        for q, label in zip(iteration_questions, labels):
            q["cluster_id"] = int(label)
        
        # Select questions to add (respecting cluster limits)
        added_count = 0
        cluster_added = defaultdict(int)
        
        for question in iteration_questions:
            cid = question["cluster_id"]
            
            # Check if cluster still needs questions
            if cluster_freq[cid] < args.max_per_cluster:
                selected_questions.append(question)
                cluster_freq[cid] += 1
                cluster_added[cid] += 1
                added_count += 1
        
        print(f"  Added: {added_count} questions ({len(cluster_added)} clusters)")
        
        # Save checkpoint
        with open(args.output_file, 'w') as f:
            json.dump(selected_questions, f, indent=2)
        
        # Save frequency array
        freq_file = args.output_file.replace('.json', '_frequencies.npy')
        np.save(freq_file, cluster_freq)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total questions selected: {len(selected_questions)}")
    print(f"Cluster distribution:")
    print(f"  Min: {cluster_freq.min()}")
    print(f"  Max: {cluster_freq.max()}")
    print(f"  Mean: {cluster_freq.mean():.2f}")
    print(f"  Std: {cluster_freq.std():.2f}")
    print(f"  Clusters with {args.max_per_cluster}+ questions: {(cluster_freq >= args.max_per_cluster).sum()}/{num_clusters}")
    print(f"\nOutput files:")
    print(f"  Questions: {args.output_file}")
    print(f"  Frequencies: {freq_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
