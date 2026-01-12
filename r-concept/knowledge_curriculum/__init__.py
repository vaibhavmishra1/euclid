"""
Knowledge-Set-Based Curriculum Learning for R-Zero

This module implements curriculum learning where:
1. Knowledge SETS are groups of KPs from original questions (868 sets = 868 questions)
2. Each set has a difficulty level 1-5 (initialized from original problem level)
3. Challenger generates 5 questions per set using ALL KPs in that set
4. Difficulty adjusts based on average R-Zero reward per set:
   - avg_reward > alpha → increase difficulty
   - avg_reward < beta → decrease difficulty
5. R-Zero's reward function is used as-is (not modified)

Key Components:
- KnowledgeSetManager: Manages knowledge sets, difficulties, rewards
- CurriculumTrainer: Main training orchestrator
- Prompt builders: Constructs prompts with ALL KPs from a set
"""

from .knowledge_manager import KnowledgeSetManager, KnowledgeSetState
# Backwards compatibility alias
from .knowledge_manager import KnowledgeSetManager as KnowledgePointManager

from .curriculum_trainer import CurriculumTrainer, CurriculumConfig

from .prompts import (
    build_knowledge_challenger_system_prompt,
    build_knowledge_challenger_user_prompt,
    build_knowledge_challenger_messages,
    build_solver_messages,
    extract_boxed_answer,
    format_knowledge_points,
    SOLVER_SYSTEM_PROMPT,
    DIFFICULTY_DESCRIPTIONS,
)

__all__ = [
    # Core classes
    'KnowledgeSetManager',
    'KnowledgeSetState',
    'KnowledgePointManager',  # Backwards compatibility
    'CurriculumTrainer',
    'CurriculumConfig',
    
    # Prompt builders
    'build_knowledge_challenger_system_prompt',
    'build_knowledge_challenger_user_prompt',
    'build_knowledge_challenger_messages',
    'build_solver_messages',
    'extract_boxed_answer',
    'format_knowledge_points',
    'SOLVER_SYSTEM_PROMPT',
    'DIFFICULTY_DESCRIPTIONS',
]
