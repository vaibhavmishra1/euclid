#!/usr/bin/env python3
"""
Test script to verify that questions and embeddings are correctly aligned.

Usage:
    python test_embedding_alignment.py --questions corpus_cleaned/all_questions_cleaned.json \
                                        --embeddings corpus_cleaned/embeddings_cleaned.npy \
                                        --model "Qwen/Qwen3-Embedding-0.6B" \
                                        --num_samples 10000 \
                                        --use_vllm
"""
import argparse
import json
import numpy as np
import random
from tqdm import tqdm
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from vllm import LLM
except ImportError:
    LLM = None


def load_questions(filepath: str) -> List[dict]:
    """Load questions from JSON file."""
    print(f"Loading questions from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict) and 'questions' in data:
        questions = data['questions']
    else:
        raise ValueError(f"Unsupported JSON format")
    
    print(f"Loaded {len(questions):,} questions")
    return questions


def load_stored_embeddings(filepath: str) -> np.ndarray:
    """Load stored embeddings."""
    print(f"Loading stored embeddings from: {filepath}")
    embeddings = np.load(filepath)
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings


def extract_question_text(question_dict) -> str:
    """Extract question text from question dict (handles multiple formats)."""
    if isinstance(question_dict, str):
        return question_dict
    elif isinstance(question_dict, dict):
        # Try different possible keys
        for key in ['question', 'problem', 'text', 'input']:
            if key in question_dict and question_dict[key]:
                return question_dict[key]
    return str(question_dict)


def embed_questions_vllm(questions: List[str], model_name: str, max_tokens: int = 32768) -> np.ndarray:
    """Embed questions using vLLM."""
    print(f"Loading vLLM model: {model_name}")
    
    model = LLM(
        model=model_name,
        runner="pooling",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=max_tokens,
    )
    
    tokenizer = model.get_tokenizer()
    
    # Truncate questions if needed
    print("Checking and truncating questions if needed...")
    truncated_questions = []
    for q in questions:
        if len(q) > max_tokens * 2:
            tokens = tokenizer.encode(q)
            if len(tokens) > max_tokens:
                q = tokenizer.decode(tokens[:max_tokens - 1])
        truncated_questions.append(q)
    
    print(f"Embedding {len(truncated_questions)} questions...")
    outputs = model.embed(truncated_questions)
    
    # Extract embeddings
    embeddings = []
    for output in outputs:
        embedding = output.outputs.embedding
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings, dtype='float32')
    
    # Normalize
    print("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / (norms + 1e-12)
    
    return embeddings


def embed_questions_sentence_transformer(questions: List[str], model_name: str) -> np.ndarray:
    """Embed questions using sentence-transformers."""
    print(f"Loading sentence-transformers model: {model_name}")
    
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    print(f"Embedding {len(questions)} questions...")
    embeddings = model.encode(
        questions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    return embeddings


def compare_embeddings(fresh_embeddings: np.ndarray, stored_embeddings: np.ndarray, 
                      indices: List[int], tolerance: float = 1e-4) -> dict:
    """
    Compare fresh embeddings with stored embeddings.
    
    Returns dict with comparison statistics.
    """
    print(f"\nComparing embeddings (tolerance: {tolerance})...")
    
    assert len(fresh_embeddings) == len(stored_embeddings) == len(indices), \
        f"Size mismatch: {len(fresh_embeddings)} vs {len(stored_embeddings)} vs {len(indices)}"
    
    results = {
        "num_samples": len(indices),
        "indices_tested": indices,
        "matches": 0,
        "mismatches": 0,
        "cosine_similarities": [],
        "l2_distances": [],
        "max_differences": [],
        "mismatch_details": []
    }
    
    print("\nTesting alignment...")
    for i, idx in enumerate(tqdm(indices, desc="Comparing")):
        fresh = fresh_embeddings[i]
        stored = stored_embeddings[i]
        
        # Compute cosine similarity
        cosine_sim = np.dot(fresh, stored) / (np.linalg.norm(fresh) * np.linalg.norm(stored) + 1e-12)
        results["cosine_similarities"].append(float(cosine_sim))
        
        # Compute L2 distance
        l2_dist = np.linalg.norm(fresh - stored)
        results["l2_distances"].append(float(l2_dist))
        
        # Compute max absolute difference
        max_diff = np.max(np.abs(fresh - stored))
        results["max_differences"].append(float(max_diff))
        
        # Check if they match within tolerance
        if np.allclose(fresh, stored, atol=tolerance, rtol=0):
            results["matches"] += 1
        else:
            results["mismatches"] += 1
            if len(results["mismatch_details"]) < 10:  # Keep first 10 mismatches
                results["mismatch_details"].append({
                    "index": int(idx),
                    "cosine_similarity": float(cosine_sim),
                    "l2_distance": float(l2_dist),
                    "max_difference": float(max_diff)
                })
    
    # Compute statistics
    results["avg_cosine_similarity"] = float(np.mean(results["cosine_similarities"]))
    results["min_cosine_similarity"] = float(np.min(results["cosine_similarities"]))
    results["avg_l2_distance"] = float(np.mean(results["l2_distances"]))
    results["max_l2_distance"] = float(np.max(results["l2_distances"]))
    results["avg_max_difference"] = float(np.mean(results["max_differences"]))
    
    return results


def print_results(results: dict):
    """Print comparison results in a nice format."""
    print("\n" + "="*70)
    print("ALIGNMENT TEST RESULTS")
    print("="*70)
    
    print(f"\nSamples tested: {results['num_samples']}")
    print(f"Tolerance: {1e-4}")
    
    print(f"\n{'='*70}")
    print("EXACT MATCH RESULTS:")
    print(f"{'='*70}")
    print(f"Exact matches (within tolerance): {results['matches']}/{results['num_samples']}")
    print(f"Mismatches: {results['mismatches']}/{results['num_samples']}")
    
    if results['matches'] == results['num_samples']:
        print("\n✅ SUCCESS: All embeddings match exactly!")
    else:
        print(f"\n⚠️  WARNING: {results['mismatches']} embeddings don't match within tolerance")
    
    print(f"\n{'='*70}")
    print("SIMILARITY METRICS:")
    print(f"{'='*70}")
    print(f"Average cosine similarity: {results['avg_cosine_similarity']:.10f}")
    print(f"Minimum cosine similarity: {results['min_cosine_similarity']:.10f}")
    print(f"Average L2 distance: {results['avg_l2_distance']:.10f}")
    print(f"Maximum L2 distance: {results['max_l2_distance']:.10f}")
    print(f"Average max element difference: {results['avg_max_difference']:.10f}")
    
    # Interpret results
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    if results['avg_cosine_similarity'] > 0.9999:
        print("✅ Cosine similarity is excellent (>0.9999)")
        print("   Embeddings are very highly aligned!")
    elif results['avg_cosine_similarity'] > 0.999:
        print("✅ Cosine similarity is very good (>0.999)")
        print("   Embeddings are well aligned with minor numerical differences")
    elif results['avg_cosine_similarity'] > 0.99:
        print("⚠️  Cosine similarity is good (>0.99) but not excellent")
        print("   There may be some minor differences")
    else:
        print("❌ Cosine similarity is low (<0.99)")
        print("   Embeddings may be misaligned!")
    
    if results['avg_l2_distance'] < 0.01:
        print(f"✅ Average L2 distance is very small (<0.01)")
    elif results['avg_l2_distance'] < 0.1:
        print(f"⚠️  Average L2 distance is moderate (<0.1)")
    else:
        print(f"❌ Average L2 distance is large (>0.1)")
    
    # Show mismatch details if any
    if results['mismatch_details']:
        print(f"\n{'='*70}")
        print(f"MISMATCH DETAILS (first {len(results['mismatch_details'])}):")
        print(f"{'='*70}")
        for detail in results['mismatch_details']:
            print(f"\nIndex: {detail['index']}")
            print(f"  Cosine similarity: {detail['cosine_similarity']:.10f}")
            print(f"  L2 distance: {detail['l2_distance']:.10f}")
            print(f"  Max element difference: {detail['max_difference']:.10f}")
    
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Test embedding alignment")
    parser.add_argument("--questions", type=str, required=True,
                       help="Path to questions JSON file")
    parser.add_argument("--embeddings", type=str, required=True,
                       help="Path to stored embeddings .npy file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                       help="Embedding model name")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of random samples to test")
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM for embedding")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                       help="Tolerance for exact matching")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*70)
    print("EMBEDDING ALIGNMENT TEST")
    print("="*70)
    
    # Load data
    questions = load_questions(args.questions)
    stored_embeddings = load_stored_embeddings(args.embeddings)
    
    # Verify counts match
    if len(questions) != len(stored_embeddings):
        print(f"\n❌ ERROR: Count mismatch!")
        print(f"   Questions: {len(questions)}")
        print(f"   Embeddings: {len(stored_embeddings)}")
        return
    
    # Sample random indices
    total_count = len(questions)
    num_samples = min(args.num_samples, total_count)
    print(f"\nSampling {num_samples} random indices from {total_count} total...")
    sampled_indices = sorted(random.sample(range(total_count), num_samples))
    print(f"Sampled indices range: [{sampled_indices[0]}, {sampled_indices[-1]}]")
    
    # Extract sampled questions
    print("\nExtracting sampled questions...")
    sampled_questions = []
    for idx in tqdm(sampled_indices, desc="Extracting"):
        q = questions[idx]
        q_text = extract_question_text(q)
        sampled_questions.append(q_text)
    
    # Extract corresponding stored embeddings
    print("\nExtracting stored embeddings for sampled indices...")
    sampled_stored_embeddings = stored_embeddings[sampled_indices]
    
    # Generate fresh embeddings for sampled questions
    print("\nGenerating fresh embeddings...")
    if args.use_vllm:
        if LLM is None:
            raise ImportError("vllm is not installed. Please install it with: pip install vllm")
        fresh_embeddings = embed_questions_vllm(sampled_questions, args.model)
    else:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please install it with: pip install sentence-transformers")
        fresh_embeddings = embed_questions_sentence_transformer(sampled_questions, args.model)
    
    # Compare embeddings
    results = compare_embeddings(
        fresh_embeddings,
        sampled_stored_embeddings,
        sampled_indices,
        tolerance=args.tolerance
    )
    
    # Print results
    print_results(results)
    
    # Save results to JSON
    results_file = "alignment_test_results.json"
    print(f"\nSaving results to: {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
