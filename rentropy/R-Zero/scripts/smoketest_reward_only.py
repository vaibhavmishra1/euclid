#!/usr/bin/env python3
"""
Smoke test for Rentropy reward computation only.
Tests all 4 diversity modes without running the full training loop.

Usage:
    python scripts/smoketest_reward_only.py

This tests:
1. Cluster space loading
2. Cluster assignment
3. All 4 reward computation modes
4. Count tracking / EMA updates
"""
import os
import sys
import yaml
import numpy as np
from typing import List, Dict

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RZERO_DIR = os.path.dirname(SCRIPT_DIR)
RENTROPY_DIR = os.path.dirname(RZERO_DIR)
sys.path.insert(0, RENTROPY_DIR)

# Mock questions for testing
MOCK_QUESTIONS = [
    "Find all integers n such that n^2 + 1 is divisible by n + 1.",
    "Prove that the sum of the first n odd numbers equals n^2.",
    "Calculate the integral of sin(x)cos(x) from 0 to pi.",
    "How many ways can you arrange 5 distinct books on a shelf?",
    "Find the derivative of f(x) = x^3 * e^x.",
    "Solve the equation 2x + 3 = 7.",
    "What is the probability of rolling a sum of 7 with two dice?",
    "Prove that sqrt(2) is irrational.",
]

# Mock base scores (majority vote scores)
MOCK_BASE_SCORES = [0.6, 0.8, 0.4, 0.7, 0.5, 0.9, 0.3, 0.6]


def test_cluster_assigner():
    """Test cluster assignment without pre-built centroids."""
    print("\n" + "="*60)
    print("TEST 1: Cluster Assigner (mock centroids)")
    print("="*60)
    
    from cluster_space.cluster_assigner import ClusterAssigner
    
    # Create mock centroids (random for testing)
    np.random.seed(42)
    mock_centroids = np.random.randn(16, 384)  # 16 clusters, 384-dim
    mock_centroids = mock_centroids / np.linalg.norm(mock_centroids, axis=1, keepdims=True)
    
    # Save mock centroids
    mock_centroids_path = os.path.join(RENTROPY_DIR, "cluster_space", "cluster_data")
    os.makedirs(mock_centroids_path, exist_ok=True)
    centroids_file = os.path.join(mock_centroids_path, "centroids.npy")
    np.save(centroids_file, mock_centroids)
    print(f"Created mock centroids at {centroids_file}")
    
    # Test assigner
    try:
        assigner = ClusterAssigner(
            centroids_path=centroids_file,
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            ema_decay=0.99,
            smoothing_alpha=1.0,
        )
        print("✓ ClusterAssigner initialized successfully")
        
        # Test cluster assignment
        cluster_ids = assigner.assign_clusters(MOCK_QUESTIONS[:3])
        print(f"✓ Assigned clusters for 3 questions: {cluster_ids}")
        
        # Test rarity reward
        rarity = assigner.compute_rarity_reward(cluster_ids)
        print(f"✓ Rarity rewards: {rarity}")
        
        # Test batch uniqueness
        batch_unique = assigner.compute_batch_uniqueness_reward(cluster_ids)
        print(f"✓ Batch uniqueness: {batch_unique}")
        
        # Test within-cluster uniqueness
        within_unique = assigner.compute_within_cluster_uniqueness(MOCK_QUESTIONS[:3], cluster_ids)
        print(f"✓ Within-cluster uniqueness: {within_unique}")
        
        # Test count update
        assigner.update_counts(cluster_ids)
        print(f"✓ Updated counts, total: {assigner.total_count:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_diversity_modes():
    """Test all 4 diversity reward modes."""
    print("\n" + "="*60)
    print("TEST 2: All Diversity Modes")
    print("="*60)
    
    # Load config
    config_path = os.path.join(RENTROPY_DIR, "rentropy_config.yaml")
    
    for mode in [1, 2, 3, 4]:
        print(f"\n--- Mode {mode} ---")
        
        # Update config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['diversity_mode'] = mode
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # Reload the reward module to pick up new config
        if 'examples.reward_function.caller_rentropy' in sys.modules:
            del sys.modules['examples.reward_function.caller_rentropy']
        
        # Import and test
        sys.path.insert(0, RZERO_DIR)
        try:
            from cluster_space.cluster_assigner import ClusterAssigner, reset_assigner
            reset_assigner()  # Reset global assigner
            
            # Simulate diversity reward computation
            centroids_path = os.path.join(RENTROPY_DIR, "cluster_space", "cluster_data", "centroids.npy")
            if not os.path.exists(centroids_path):
                print(f"  Skipping (no centroids)")
                continue
            
            assigner = ClusterAssigner(centroids_path)
            
            # Filter questions by threshold
            threshold = 0.3
            valid_indices = [i for i, s in enumerate(MOCK_BASE_SCORES) 
                          if s >= threshold and MOCK_QUESTIONS[i]]
            valid_questions = [MOCK_QUESTIONS[i] for i in valid_indices]
            
            if not valid_questions:
                print(f"  No valid questions")
                continue
            
            cluster_ids = assigner.assign_clusters(valid_questions)
            
            # Compute rewards based on mode
            rewards = np.zeros(len(valid_questions))
            
            if mode >= 2:
                rarity = assigner.compute_rarity_reward(cluster_ids)
                rewards += 0.1 * rarity
                print(f"  Rarity contribution: {0.1 * rarity}")
            
            if mode >= 3:
                batch_uniq = assigner.compute_batch_uniqueness_reward(cluster_ids)
                rewards += 0.05 * batch_uniq
                print(f"  Batch uniqueness contribution: {0.05 * batch_uniq}")
            
            if mode >= 4:
                within_uniq = assigner.compute_within_cluster_uniqueness(valid_questions, cluster_ids)
                rewards += 0.05 * within_uniq
                print(f"  Within-cluster contribution: {0.05 * within_uniq}")
            
            print(f"  Total diversity rewards: {rewards}")
            print(f"  ✓ Mode {mode} passed")
            
        except Exception as e:
            print(f"  ✗ Mode {mode} failed: {e}")
            import traceback
            traceback.print_exc()


def test_reward_magnitude():
    """Test that rewards are in expected ranges."""
    print("\n" + "="*60)
    print("TEST 3: Reward Magnitude Sanity Check")
    print("="*60)
    
    centroids_path = os.path.join(RENTROPY_DIR, "cluster_space", "cluster_data", "centroids.npy")
    if not os.path.exists(centroids_path):
        print("Skipping (no centroids)")
        return
    
    from cluster_space.cluster_assigner import ClusterAssigner, reset_assigner
    reset_assigner()
    
    assigner = ClusterAssigner(centroids_path)
    cluster_ids = assigner.assign_clusters(MOCK_QUESTIONS)
    
    # All rewards should be in [0, 1] after normalization
    rarity = assigner.compute_rarity_reward(cluster_ids)
    batch_uniq = assigner.compute_batch_uniqueness_reward(cluster_ids)
    within_uniq = assigner.compute_within_cluster_uniqueness(MOCK_QUESTIONS, cluster_ids)
    
    def check_range(name, values, min_val=0, max_val=1.5):
        in_range = all(min_val <= v <= max_val for v in values)
        status = "✓" if in_range else "✗"
        print(f"{status} {name}: min={min(values):.3f}, max={max(values):.3f}, mean={np.mean(values):.3f}")
        return in_range
    
    check_range("Rarity rewards", rarity)
    check_range("Batch uniqueness", batch_uniq)
    check_range("Within-cluster uniqueness", within_uniq)


def main():
    print("="*60)
    print("RENTROPY REWARD SMOKE TEST")
    print("="*60)
    print(f"RENTROPY_DIR: {RENTROPY_DIR}")
    print(f"RZERO_DIR: {RZERO_DIR}")
    
    # Run tests
    test1_passed = test_cluster_assigner()
    test_all_diversity_modes()
    test_reward_magnitude()
    
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    
    if test1_passed:
        print("✓ Core functionality working")
        print("\nTo run full training smoke test:")
        print("  cd R-Zero")
        print("  bash scripts/smoketest_all_variants.sh Qwen/Qwen3-0.6B-Base")
    else:
        print("✗ Some tests failed - check errors above")


if __name__ == "__main__":
    main()
