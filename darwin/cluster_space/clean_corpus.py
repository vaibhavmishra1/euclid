#!/usr/bin/env python3
"""
Clean corpus by removing questions with zero embeddings.

Usage:
    python clean_corpus.py --questions all_questions.json --embeddings embeddings_201027.npy --output_dir corpus_cleaned
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_embeddings(filepath):
    """Load embeddings file (handles memmap format)."""
    print(f"Loading embeddings from: {filepath}")
    
    # Get file size
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size / 1e9:.2f} GB")
    
    # Detect shape based on file size
    embedding_dim = 1024  # From build_clusters.py
    num_samples = file_size // (embedding_dim * 4)  # 4 bytes for float32
    
    print(f"Detected shape: ({num_samples}, {embedding_dim})")
    
    # Load as memmap (memory-efficient)
    embeddings = np.memmap(
        filepath,
        dtype='float32',
        mode='r',
        shape=(num_samples, embedding_dim)
    )
    
    return embeddings


def find_non_zero_indices(embeddings):
    """Find indices of embeddings that are not all zeros."""
    print("\nFinding non-zero embeddings...")
    
    # Check which embeddings are completely zero
    zero_mask = np.all(embeddings == 0, axis=1)
    non_zero_mask = ~zero_mask
    non_zero_indices = np.where(non_zero_mask)[0]
    
    num_total = len(embeddings)
    num_valid = len(non_zero_indices)
    num_invalid = num_total - num_valid
    
    print(f"Total embeddings: {num_total:,}")
    print(f"Valid (non-zero) embeddings: {num_valid:,} ({100 * num_valid / num_total:.2f}%)")
    print(f"Invalid (zero) embeddings: {num_invalid:,} ({100 * num_invalid / num_total:.2f}%)")
    
    return non_zero_indices


def load_questions(filepath):
    """Load questions from JSON file."""
    print(f"\nLoading questions from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict) and 'questions' in data:
        questions = data['questions']
    else:
        raise ValueError(f"Unsupported JSON format in {filepath}")
    
    print(f"Loaded {len(questions):,} questions")
    return questions


def filter_questions(questions, valid_indices):
    """Filter questions to keep only those with valid embeddings."""
    print("\nFiltering questions...")
    
    valid_indices_set = set(valid_indices)
    filtered_questions = []
    
    for idx in tqdm(valid_indices, desc="Selecting questions"):
        if idx < len(questions):
            filtered_questions.append(questions[idx])
        else:
            print(f"Warning: Index {idx} out of range (max: {len(questions)-1})")
    
    print(f"Filtered questions: {len(filtered_questions):,}")
    return filtered_questions


def save_cleaned_data(output_dir, questions, embeddings, valid_indices):
    """Save cleaned questions and embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cleaned questions
    questions_path = os.path.join(output_dir, "all_questions_cleaned.json")
    print(f"\nSaving cleaned questions to: {questions_path}")
    with open(questions_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(questions):,} questions")
    
    # Save cleaned embeddings
    embeddings_path = os.path.join(output_dir, "embeddings_cleaned.npy")
    print(f"\nSaving cleaned embeddings to: {embeddings_path}")
    
    # Extract only valid embeddings
    print("Extracting valid embeddings...")
    cleaned_embeddings = np.zeros((len(valid_indices), embeddings.shape[1]), dtype='float32')
    
    for new_idx, orig_idx in enumerate(tqdm(valid_indices, desc="Copying embeddings")):
        cleaned_embeddings[new_idx] = embeddings[orig_idx]
    
    # Save as standard numpy array
    np.save(embeddings_path, cleaned_embeddings)
    
    file_size = os.path.getsize(embeddings_path)
    print(f"Saved embeddings: {cleaned_embeddings.shape}")
    print(f"File size: {file_size / 1e9:.2f} GB")
    
    # Save metadata about the cleaning
    metadata = {
        "original_count": int(len(embeddings)),
        "cleaned_count": int(len(valid_indices)),
        "removed_count": int(len(embeddings) - len(valid_indices)),
        "embedding_dim": int(embeddings.shape[1]),
        "valid_indices_range": {
            "first": int(valid_indices[0]),
            "last": int(valid_indices[-1])
        }
    }
    
    metadata_path = os.path.join(output_dir, "cleaning_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved cleaning metadata to: {metadata_path}")
    
    return cleaned_embeddings


def verify_cleaned_data(embeddings):
    """Verify that cleaned embeddings have no zeros."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    zero_mask = np.all(embeddings == 0, axis=1)
    num_zeros = np.sum(zero_mask)
    
    if num_zeros == 0:
        print("✓ All embeddings are non-zero!")
    else:
        print(f"✗ Warning: Found {num_zeros} zero embeddings in cleaned data")
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nNormalization check:")
    print(f"  Min norm: {np.min(norms):.6f}")
    print(f"  Max norm: {np.max(norms):.6f}")
    print(f"  Mean norm: {np.mean(norms):.6f}")
    
    if np.allclose(norms, 1.0, atol=0.01):
        print("  ✓ Embeddings are normalized")
    else:
        print("  ✗ Embeddings may not be normalized")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Clean corpus by removing zero embeddings")
    parser.add_argument("--questions", type=str, required=True,
                       help="Path to questions JSON file")
    parser.add_argument("--embeddings", type=str, required=True,
                       help="Path to embeddings .npy file")
    parser.add_argument("--output_dir", type=str, default="corpus_cleaned",
                       help="Output directory for cleaned data")
    args = parser.parse_args()
    
    print("="*60)
    print("CLEANING CORPUS")
    print("="*60)
    
    # Load embeddings
    embeddings = load_embeddings(args.embeddings)
    
    # Find valid (non-zero) embeddings
    valid_indices = find_non_zero_indices(embeddings)
    
    # Load questions
    questions = load_questions(args.questions)
    
    # Verify counts match
    if len(questions) != len(embeddings):
        print(f"\n⚠️  WARNING: Question count ({len(questions):,}) doesn't match embedding count ({len(embeddings):,})")
        print(f"   Will only use questions up to the number of embeddings")
    
    # Filter questions
    filtered_questions = filter_questions(questions, valid_indices)
    
    # Save cleaned data
    cleaned_embeddings = save_cleaned_data(
        args.output_dir,
        filtered_questions,
        embeddings,
        valid_indices
    )
    
    # Verify
    verify_cleaned_data(cleaned_embeddings)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Files created:")
    print(f"  - all_questions_cleaned.json ({len(filtered_questions):,} questions)")
    print(f"  - embeddings_cleaned.npy ({cleaned_embeddings.shape})")
    print(f"  - cleaning_metadata.json")
    print("="*60)


if __name__ == "__main__":
    main()
