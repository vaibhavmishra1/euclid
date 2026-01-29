"""
Simple configuration for R-Zero training pipeline.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Main configuration class."""
    
    # Model settings
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"  # Lightweight model for testing
    output_dir: str = "./output"
    
    # Concept settings
    concept_sets_file: str = "examples/concept_sets.json"
    num_concepts: int = 5
    
    # Training settings
    num_iterations: int = 3  # Train for 3 iterations per concept
    questions_per_concept: int = 20  # Generate 20 questions per concept
    num_rollouts: int = 8  # Number of rollouts for self-consistency
    
    # GRPO settings
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 50  # Steps per concept iteration
    
    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    
    # Evaluation settings
    eval_num_rollouts: int = 9  # More rollouts for evaluation
    
    # vLLM settings
    tensor_parallel_size: int = 1  # Number of GPUs for vLLM
    gpu_memory_utilization: float = 0.9
