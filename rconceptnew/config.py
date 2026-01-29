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
    num_rollouts: int = 4  # Number of rollouts for self-consistency
    
    # GRPO settings
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 10  # Steps per concept iteration
    
    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    
    # Evaluation settings
    eval_num_rollouts: int = 9  # More rollouts for evaluation
    eval_enabled: bool = True
    # Static held-out question bank used to track per-concept progress.
    eval_question_bank_file: str = "data/concept_to_questions.json"
    eval_questions_per_concept: int = 20
    eval_output_subdir: str = "evals"
    # vLLM memory for evaluation (keep low since it runs right after GRPO which may
    # not fully release memory immediately).
    eval_gpu_memory_utilization: float = 0.2
    
    # vLLM settings
    tensor_parallel_size: int = 1  # Number of GPUs for vLLM
    # Leave some headroom to avoid vLLM warmup OOMs between pipeline phases.
    gpu_memory_utilization: float = 0.60
    # vLLM defaults can warm up with 1024 dummy requests; cap concurrency to reduce
    # warmup memory spikes and improve stability when re-initializing vLLM.
    vllm_max_num_seqs: int = 64

    # TRL GRPO vLLM generation (used *inside* GRPOTrainer when use_vllm=True)
    # This can significantly speed up the generation portion of GRPO.
    grpo_use_vllm: bool = False
    # TRL defaults to "server" which expects `trl vllm-serve` to be running.
    # Use "colocate" to run vLLM in-process on the same GPU(s) as training.
    grpo_vllm_mode: str = "colocate"
    # Keep this low to avoid fighting with the HF training model for VRAM.
    grpo_vllm_gpu_memory_utilization: float = 0.10
    grpo_vllm_tensor_parallel_size: int = 1
    # Optional cap for vLLM's max model length during GRPO generation.
    grpo_vllm_max_model_length: int | None = None
    
    # HuggingFace Transformers model loading memory limit
    # Format: dict mapping device_id to memory string (e.g., "75GiB" for 75GB)
    # Set to None to use Transformers' default (90% of GPU memory)
    hf_max_memory: dict | None = None  # e.g., {0: "100GiB"} for H200-140GB
