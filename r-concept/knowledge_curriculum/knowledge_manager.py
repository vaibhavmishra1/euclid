"""
Knowledge Set Manager for Curriculum Learning

This module manages knowledge SETS (groups of knowledge points from original questions).
Each knowledge set corresponds to one original question's KP list.

Key features:
- 868 sets = 868 original questions
- Difficulty levels 1-5 (initialized from original problem level)
- Each set has its own difficulty level
"""

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


@dataclass
class KnowledgeSetState:
    """State for a single knowledge set."""
    knowledge_points: List[str]  # The list of KPs in this set
    difficulty: int = 1
    original_level: int = 1  # Original problem level from dataset
    reward_history: List[float] = field(default_factory=list)
    questions_generated: int = 0
    current_iteration_rewards: List[float] = field(default_factory=list)
    
    def get_avg_reward(self) -> float:
        """Get average reward for current iteration."""
        if not self.current_iteration_rewards:
            return 0.5  # Default if no rewards yet
        return sum(self.current_iteration_rewards) / len(self.current_iteration_rewards)
    
    def reset_iteration_rewards(self):
        """Reset rewards for new iteration, save to history."""
        if self.current_iteration_rewards:
            self.reward_history.append(self.get_avg_reward())
        self.current_iteration_rewards = []


class KnowledgeSetManager:
    """
    Manages knowledge sets for curriculum learning.
    
    Each knowledge set = one original question's KP list
    Total sets = number of questions in dataset (e.g., 868)
    Difficulty levels: 1-5 (initialized from original problem level)
    """
    
    MIN_DIFFICULTY = 1
    MAX_DIFFICULTY = 5
    
    def __init__(
        self,
        knowledge_points_path: str,
        alpha: float = 0.7,
        beta: float = 0.3,
        questions_per_set: int = 5,
        state_save_path: Optional[str] = None,
    ):
        """
        Initialize the knowledge set manager.
        
        Args:
            knowledge_points_path: Path to JSONL file with questions and their KP lists
            alpha: Upper threshold - if avg_reward > alpha, increase difficulty
            beta: Lower threshold - if avg_reward < beta, decrease difficulty
            questions_per_set: Number of questions to generate per knowledge set
            state_save_path: Path to save/load state for resuming
        """
        self.knowledge_points_path = knowledge_points_path
        self.alpha = alpha
        self.beta = beta
        self.questions_per_set = questions_per_set
        self.state_save_path = state_save_path
        
        # Load knowledge sets from file
        self.knowledge_sets: Dict[int, KnowledgeSetState] = {}
        self._load_knowledge_sets()
        
        # Try to load existing state
        if state_save_path and os.path.exists(state_save_path):
            self.load_state()
    
    def _parse_level(self, level_str: str) -> int:
        """
        Parse level string like 'Level 4' to integer 1-5.
        
        Args:
            level_str: String like 'Level 1', 'Level 5', etc.
        
        Returns:
            Integer difficulty 1-5
        """
        if not level_str:
            return 1
        
        # Extract number from string like "Level 4"
        match = re.search(r'(\d+)', level_str)
        if match:
            level = int(match.group(1))
            # Clamp to 1-5 range
            return max(self.MIN_DIFFICULTY, min(self.MAX_DIFFICULTY, level))
        
        return 1  # Default
    
    def _load_knowledge_sets(self):
        """Load knowledge sets from JSONL file."""
        with open(self.knowledge_points_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    kp_list = data.get('knowledge_points', [])
                    level_str = data.get('level', 'Level 1')
                    
                    # Parse original level and use as initial difficulty
                    original_level = self._parse_level(level_str)
                    
                    # Each line = one knowledge set
                    self.knowledge_sets[idx] = KnowledgeSetState(
                        knowledge_points=kp_list,
                        difficulty=original_level,  # Initialize from original level
                        original_level=original_level,
                    )
        
        # Print statistics
        levels = [s.difficulty for s in self.knowledge_sets.values()]
        level_dist = defaultdict(int)
        for l in levels:
            level_dist[l] += 1
        
        print(f"Loaded {len(self.knowledge_sets)} knowledge sets")
        print(f"Initial difficulty distribution (from problem levels): {dict(sorted(level_dist.items()))}")
    
    def get_training_batch(self) -> List[Tuple[int, List[str], int]]:
        """
        Get training batch: (set_id, kp_list, difficulty) for each set.
        
        Returns:
            List of (set_id, knowledge_points_list, difficulty) tuples
            Total items = num_sets × questions_per_set
        """
        batch = []
        
        for set_id, state in self.knowledge_sets.items():
            for _ in range(self.questions_per_set):
                batch.append((set_id, state.knowledge_points, state.difficulty))
        
        # Shuffle to mix different sets
        random.shuffle(batch)
        
        return batch
    
    def record_reward(self, set_id: int, reward: float):
        """
        Record a reward for a knowledge set.
        
        Args:
            set_id: The knowledge set ID (index)
            reward: The reward value (from R-Zero's reward function)
        """
        if set_id in self.knowledge_sets:
            self.knowledge_sets[set_id].current_iteration_rewards.append(reward)
            self.knowledge_sets[set_id].questions_generated += 1
    
    def update_difficulties(self) -> Dict[int, Dict]:
        """
        Update difficulties for all knowledge sets based on average rewards.
        
        Logic:
        - If avg_reward > alpha: Solver finds it too easy → increase difficulty
        - If avg_reward < beta: Solver finds it too hard → decrease difficulty
        - Otherwise: Keep same difficulty (in the sweet spot)
        
        Returns:
            Dictionary of adjustments made
        """
        adjustments = {}
        
        for set_id, state in self.knowledge_sets.items():
            avg_reward = state.get_avg_reward()
            old_difficulty = state.difficulty
            action = "no_change"
            
            if len(state.current_iteration_rewards) == 0:
                # No questions generated for this set, skip
                action = "skipped"
            elif avg_reward > self.alpha:
                # Too easy → increase difficulty
                if state.difficulty < self.MAX_DIFFICULTY:
                    state.difficulty += 1
                    action = "increased"
                else:
                    action = "at_max"
            elif avg_reward < self.beta:
                # Too hard → decrease difficulty
                if state.difficulty > self.MIN_DIFFICULTY:
                    state.difficulty -= 1
                    action = "decreased"
                else:
                    action = "at_min"
            else:
                # Sweet spot → maintain
                action = "maintained"
            
            adjustments[set_id] = {
                "action": action,
                "old_difficulty": old_difficulty,
                "new_difficulty": state.difficulty,
                "avg_reward": avg_reward,
                "num_questions": len(state.current_iteration_rewards),
                "kp_preview": state.knowledge_points[0][:50] if state.knowledge_points else "",
            }
            
            # Reset for next iteration
            state.reset_iteration_rewards()
        
        return adjustments
    
    def get_statistics(self) -> Dict:
        """Get current statistics about knowledge sets."""
        difficulties = [s.difficulty for s in self.knowledge_sets.values()]
        
        difficulty_distribution = defaultdict(int)
        for d in difficulties:
            difficulty_distribution[d] += 1
        
        return {
            "total_knowledge_sets": len(self.knowledge_sets),
            "avg_difficulty": sum(difficulties) / len(difficulties) if difficulties else 0,
            "min_difficulty": min(difficulties) if difficulties else 0,
            "max_difficulty": max(difficulties) if difficulties else 0,
            "difficulty_distribution": dict(sorted(difficulty_distribution.items())),
            "total_questions_generated": sum(s.questions_generated for s in self.knowledge_sets.values()),
        }
    
    def save_state(self):
        """Save current state to file."""
        if not self.state_save_path:
            return
        
        os.makedirs(os.path.dirname(self.state_save_path), exist_ok=True)
        
        state = {
            "alpha": self.alpha,
            "beta": self.beta,
            "questions_per_set": self.questions_per_set,
            "knowledge_sets": {
                str(set_id): {
                    "knowledge_points": s.knowledge_points,
                    "difficulty": s.difficulty,
                    "original_level": s.original_level,
                    "reward_history": s.reward_history,
                    "questions_generated": s.questions_generated,
                }
                for set_id, s in self.knowledge_sets.items()
            }
        }
        
        with open(self.state_save_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        print(f"State saved to {self.state_save_path}")
    
    def load_state(self):
        """Load state from file."""
        if not self.state_save_path or not os.path.exists(self.state_save_path):
            return
        
        with open(self.state_save_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.alpha = state.get("alpha", self.alpha)
        self.beta = state.get("beta", self.beta)
        self.questions_per_set = state.get("questions_per_set", self.questions_per_set)
        
        for set_id_str, data in state.get("knowledge_sets", {}).items():
            set_id = int(set_id_str)
            if set_id in self.knowledge_sets:
                self.knowledge_sets[set_id].difficulty = data.get("difficulty", 1)
                self.knowledge_sets[set_id].original_level = data.get("original_level", 1)
                self.knowledge_sets[set_id].reward_history = data.get("reward_history", [])
                self.knowledge_sets[set_id].questions_generated = data.get("questions_generated", 0)
        
        print(f"State loaded from {self.state_save_path}")
    
    def export_training_prompts(self, output_path: str):
        """Export training prompts to JSONL file."""
        batch = self.get_training_batch()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for set_id, kp_list, difficulty in batch:
                item = {
                    "set_id": set_id,
                    "knowledge_points": kp_list,
                    "difficulty": difficulty,
                    "problem": f"SET_ID:{set_id}|DIFFICULTY:{difficulty}",  # Placeholder
                    "answer": "",
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Exported {len(batch)} training prompts to {output_path}")


# Backwards compatibility alias
KnowledgePointManager = KnowledgeSetManager
