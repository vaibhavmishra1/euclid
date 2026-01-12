"""
Test Script for Knowledge-Point-Based Curriculum Learning

This script verifies that all components work correctly:
1. KnowledgePointManager loads knowledge points
2. Prompts are generated correctly
3. Difficulty adjustment works
4. Dataset classes function properly
"""

import json
import os
import sys
import tempfile

def test_knowledge_manager():
    """Test KnowledgePointManager functionality."""
    print("\n" + "="*60)
    print("Testing KnowledgePointManager")
    print("="*60)
    
    from knowledge_curriculum.knowledge_manager import KnowledgePointManager
    
    # Create test data
    test_data = [
        {"question": "Test Q1", "knowledge_points": ["KP1", "KP2"], "level": "Level 1"},
        {"question": "Test Q2", "knowledge_points": ["KP2", "KP3"], "level": "Level 2"},
        {"question": "Test Q3", "knowledge_points": ["KP1"], "level": "Level 3"},
    ]
    
    # Write test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        test_file = f.name
    
    try:
        # Initialize manager
        manager = KnowledgePointManager(
            knowledge_points_path=test_file,
            alpha=0.7,
            beta=0.3,
            questions_per_kp=2,
        )
        
        # Test loading
        assert len(manager.knowledge_points) == 3, "Should have 3 unique KPs"
        print(f"✓ Loaded {len(manager.knowledge_points)} knowledge points")
        
        # Test initial difficulty
        for kp, state in manager.knowledge_points.items():
            assert state.difficulty == 1, f"Initial difficulty should be 1, got {state.difficulty}"
        print("✓ Initial difficulties are all 1")
        
        # Test training batch
        batch = manager.get_training_batch()
        assert len(batch) == 6, f"Should have 6 samples (3 KPs x 2), got {len(batch)}"
        print(f"✓ Training batch has {len(batch)} samples")
        
        # Test recording rewards
        manager.record_reward("KP1", 0.8)
        manager.record_reward("KP1", 0.9)
        manager.record_reward("KP2", 0.2)
        manager.record_reward("KP2", 0.1)
        manager.record_reward("KP3", 0.5)
        manager.record_reward("KP3", 0.6)
        print("✓ Rewards recorded successfully")
        
        # Test difficulty adjustment
        adjustments = manager.update_difficulties()
        
        # KP1: avg=0.85 > alpha(0.7) → increase
        assert adjustments["KP1"]["action"] == "increased"
        assert manager.knowledge_points["KP1"].difficulty == 2
        print("✓ KP1 difficulty increased (high reward)")
        
        # KP2: avg=0.15 < beta(0.3) → decrease, but can't go below 1
        assert adjustments["KP2"]["action"] == "decreased"
        assert manager.knowledge_points["KP2"].difficulty == 1  # Can't go below 1
        print("✓ KP2 difficulty maintained at min (low reward)")
        
        # KP3: avg=0.55, between beta and alpha → maintain
        assert adjustments["KP3"]["action"] == "maintained"
        assert manager.knowledge_points["KP3"].difficulty == 1
        print("✓ KP3 difficulty maintained (medium reward)")
        
        # Test statistics
        stats = manager.get_statistics()
        assert stats["total_knowledge_points"] == 3
        print(f"✓ Statistics: {stats}")
        
        # Test save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        
        manager.state_save_path = state_file
        manager.save_state()
        
        # Create new manager and load state
        manager2 = KnowledgePointManager(
            knowledge_points_path=test_file,
            state_save_path=state_file,
        )
        
        assert manager2.knowledge_points["KP1"].difficulty == 2
        print("✓ State save/load works correctly")
        
        os.unlink(state_file)
        
    finally:
        os.unlink(test_file)
    
    print("\n✓ KnowledgePointManager tests passed!")
    return True


def test_prompts():
    """Test prompt generation."""
    print("\n" + "="*60)
    print("Testing Prompt Generation")
    print("="*60)
    
    from knowledge_curriculum.prompts import (
        build_knowledge_challenger_system_prompt,
        build_knowledge_challenger_user_prompt,
        build_knowledge_challenger_messages,
    )
    
    kp = "Modular arithmetic: If a ≡ b (mod m), then a^k ≡ b^k (mod m)"
    difficulty = 5
    
    # Test system prompt
    system_prompt = build_knowledge_challenger_system_prompt(kp, difficulty)
    assert kp in system_prompt, "Knowledge point should be in prompt"
    assert "5/10" in system_prompt, "Difficulty should be in prompt"
    assert "moderate" in system_prompt.lower(), "Difficulty description should be in prompt"
    print("✓ System prompt contains knowledge point and difficulty")
    
    # Test user prompt
    user_prompt = build_knowledge_challenger_user_prompt()
    assert "Generate" in user_prompt
    print("✓ User prompt generated correctly")
    
    # Test full messages
    messages = build_knowledge_challenger_messages(kp, difficulty)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    print("✓ Message format is correct")
    
    print("\n✓ Prompt tests passed!")
    return True


def test_reward_function():
    """Test reward computation."""
    print("\n" + "="*60)
    print("Testing Reward Function")
    print("="*60)
    
    from knowledge_curriculum.reward_function import compute_score_with_knowledge_tracking
    
    # Test data
    predicts = [
        "<question>What is 2+2?</question>\n\\boxed{4}",
        "<question>Solve x^2=4</question>\n\\boxed{2}",
        "Invalid format",
    ]
    ground_truths = ["4", "2", "3"]
    knowledge_points = ["KP1", "KP2", "KP1"]
    difficulties = [1, 2, 1]
    
    scores, kp_stats = compute_score_with_knowledge_tracking(
        predicts, ground_truths, knowledge_points, difficulties
    )
    
    assert len(scores) == 3
    print(f"✓ Computed {len(scores)} scores")
    
    assert "KP1" in kp_stats
    assert "KP2" in kp_stats
    print(f"✓ Knowledge point stats computed: {list(kp_stats.keys())}")
    
    # First two should have valid format, third should be penalized
    assert scores[0]["format"] == 1
    assert scores[1]["format"] == 1
    assert scores[2]["format"] == 0
    print("✓ Format detection works correctly")
    
    print("\n✓ Reward function tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Knowledge-Point Curriculum Learning Test Suite")
    print("#"*60)
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    tests = [
        ("KnowledgePointManager", test_knowledge_manager),
        ("Prompt Generation", test_prompts),
        ("Reward Function", test_reward_function),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
