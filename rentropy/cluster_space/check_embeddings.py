#!/usr/bin/env python3
"""
Check embeddings file for shape and zero entries.
"""
import numpy as np
import sys

def check_embeddings(filepath):
    print(f"Loading embeddings from: {filepath}")
    print("This may take a moment for large files...")
    
    # Try to determine the shape from file size
    # Assuming float32 (4 bytes) and common embedding dimensions
    import os
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size / 1e9:.2f} GB")
    
    # Try common embedding dimensions
    for dim in [1024, 768, 512, 384, 256, 128]:
        num_embeddings = file_size / (dim * 4)  # 4 bytes for float32
        if num_embeddings == int(num_embeddings):
            print(f"Possible shape: ({int(num_embeddings)}, {dim})")
    
    # Load as memmap first to check shape without loading into memory
    try:
        # Try loading as regular numpy array first
        embeddings = np.load(filepath, mmap_mode='r')
        print("Loaded using numpy mmap mode")
    except Exception as e:
        print(f"Could not load with np.load: {e}")
        # Try as memmap with guessed dimensions
        # Most likely 1024 dimensions based on build_clusters.py line 189
        embedding_dim = 1024
        num_samples = file_size // (embedding_dim * 4)
        print(f"\nTrying to load as memmap with shape ({num_samples}, {embedding_dim})...")
        embeddings = np.memmap(
            filepath,
            dtype='float32',
            mode='r',
            shape=(num_samples, embedding_dim)
        )
    
    print(f"\n{'='*60}")
    print("SHAPE INFORMATION:")
    print(f"{'='*60}")
    print(f"Shape: {embeddings.shape}")
    print(f"Number of embeddings: {embeddings.shape[0]:,}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Dtype: {embeddings.dtype}")
    print(f"Total elements: {embeddings.size:,}")
    print(f"Memory size: {embeddings.nbytes / 1e9:.2f} GB")
    
    print(f"\n{'='*60}")
    print("ZERO ENTRY ANALYSIS:")
    print(f"{'='*60}")
    
    # Check for completely zero embeddings (all dimensions are zero)
    print("Checking for completely zero embeddings...")
    zero_rows = np.all(embeddings == 0, axis=1)
    num_zero_embeddings = np.sum(zero_rows)
    
    print(f"Completely zero embeddings: {num_zero_embeddings:,} / {embeddings.shape[0]:,}")
    print(f"Percentage: {100 * num_zero_embeddings / embeddings.shape[0]:.4f}%")
    
    if num_zero_embeddings > 0:
        zero_indices = np.where(zero_rows)[0]
        print(f"\nIndices of zero embeddings (first 20):")
        print(zero_indices[:20])
        
        # Show some statistics about where zeros are
        if len(zero_indices) > 1:
            print(f"\nFirst zero at index: {zero_indices[0]}")
            print(f"Last zero at index: {zero_indices[-1]}")
            
            # Check if zeros are contiguous
            diffs = np.diff(zero_indices)
            if np.all(diffs == 1):
                print("Zero embeddings are contiguous (all in a row)")
            else:
                print("Zero embeddings are scattered")
    
    # Check for any zero values (not necessarily entire embeddings)
    print(f"\n{'='*60}")
    print("ZERO VALUE STATISTICS:")
    print(f"{'='*60}")
    total_zeros = np.sum(embeddings == 0)
    print(f"Total zero values: {total_zeros:,} / {embeddings.size:,}")
    print(f"Percentage of zero values: {100 * total_zeros / embeddings.size:.4f}%")
    
    # Statistical summary
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY:")
    print(f"{'='*60}")
    print(f"Min value: {np.min(embeddings):.6f}")
    print(f"Max value: {np.max(embeddings):.6f}")
    print(f"Mean value: {np.mean(embeddings):.6f}")
    print(f"Std deviation: {np.std(embeddings):.6f}")
    
    # Check if embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1)
    non_zero_norms = norms[norms > 0]
    
    print(f"\n{'='*60}")
    print("NORMALIZATION CHECK:")
    print(f"{'='*60}")
    if len(non_zero_norms) > 0:
        print(f"Norm statistics (excluding zero embeddings):")
        print(f"  Min norm: {np.min(non_zero_norms):.6f}")
        print(f"  Max norm: {np.max(non_zero_norms):.6f}")
        print(f"  Mean norm: {np.mean(non_zero_norms):.6f}")
        print(f"  Std norm: {np.std(non_zero_norms):.6f}")
        
        # Check if approximately normalized (norm ~1)
        if np.allclose(non_zero_norms, 1.0, atol=0.01):
            print("  ✓ Embeddings appear to be normalized (L2 norm ≈ 1)")
        else:
            print("  ✗ Embeddings do not appear to be normalized")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "/root/euclid/rentropy/cluster_space/corpus/embeddings_201027.npy"
    
    check_embeddings(filepath)
