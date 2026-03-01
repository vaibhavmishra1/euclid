"""
ExpV1_1: GRPO Training for Math Problem Solving with Self-Consistency Rewards

This script trains a model using Group Relative Policy Optimization (GRPO)
on the same dataset used in ExpV1_0 (SFT), allowing direct comparison of
training algorithms.

Key differences from SFT (ExpV1_0):
- SFT: Uses best rollout COT as supervised target
- GRPO: Uses SELF-CONSISTENCY across rollouts to determine rewards
  (no ground truth answer needed - rewards based on agreement)

Reward Mechanism:
- For each prompt, generate N rollouts
- Extract \boxed{} answer from each rollout
- Find the modal (majority) answer
- Reward rollouts that agree with the modal answer
- This allows training on synthetic datasets with no ground truth!

Usage:
    python -m tree.euclid.expv1.expv1_1.run_grpo \
        --config tree/euclid/expv1/expv1_1/config.yaml
    
    # Or with command-line overrides:
    python -m tree.euclid.expv1.expv1_1.run_grpo \
        --model Qwen/Qwen3-1.7B-Base \
        --dataset tree/euclid/expv1/expv1_0/output_Qwen3-1.7B-Base_accepted/accepted_verified_retried.jsonl \
        --output ./output_grpo \
        --num-generations 4 \
        --epochs 1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# TRL imports
try:
    from trl import GRPOConfig, GRPOTrainer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("Warning: TRL not installed. Install with: pip install trl>=0.9.0")

from .data_utils import load_dataset_for_grpo, get_default_prompt_template
from .reward_function import create_self_consistency_reward_fn


@dataclass
class GRPOExperimentConfig:
    """Configuration for GRPO experiment."""
    # Model
    model_name_or_path: str = "Qwen/Qwen3-1.7B-Base"
    
    # Data
    dataset_path: str = ""
    prompt_template: Optional[str] = None
    max_prompt_length: int = 512
    max_response_length: int = 1024
    
    # GRPO specific
    num_generations: int = 4  # Number of rollouts per prompt (group size)
    temperature: float = 0.8  # Sampling temperature for generations
    top_p: float = 0.95
    kl_coef: float = 0.05  # KL penalty coefficient
    
    # Self-consistency reward settings
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_penalty: float = 0.1  # Penalty for missing \boxed{}
    min_agreement: int = 2  # Minimum rollouts agreeing for consensus
    
    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2  # Prompts per device (×num_generations = total sequences)
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Output
    output_dir: str = "./output_grpo"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # vLLM Configuration (for fast rollout generation)
    use_vllm: bool = True  # Enable vLLM for generation
    vllm_mode: str = "colocate"  # "colocate" avoids needing an external server
    vllm_device: str = "cuda"
    vllm_gpu_memory_utilization: float = 0.7
    vllm_dtype: str = "bfloat16"
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: Optional[int] = None  # Auto-detect if None
    
    # Misc
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


def load_config(config_path: str) -> GRPOExperimentConfig:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    return GRPOExperimentConfig(**cfg_dict.get("grpo", {}))


def prepare_dataset(
    dataset_path: str,
    prompt_template: Optional[str],
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
) -> Dataset:
    """
    Prepare dataset for GRPO training.
    
    The dataset only needs the 'prompt' column.
    Rewards are computed via self-consistency during training.
    """
    if prompt_template is None:
        prompt_template = get_default_prompt_template()
    
    dataset = load_dataset_for_grpo(
        dataset_path,
        prompt_template=prompt_template,
    )
    
    # Tokenize to check lengths and filter
    def tokenize_and_filter(example):
        tokens = tokenizer(example["prompt"], truncation=False)
        return {"prompt_length": len(tokens["input_ids"])}
    
    dataset = dataset.map(tokenize_and_filter)
    
    # Filter out prompts that are too long
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: x["prompt_length"] <= max_prompt_length)
    filtered_len = len(dataset)
    
    if filtered_len < original_len:
        print(f"[GRPO] Filtered {original_len - filtered_len} examples due to prompt length > {max_prompt_length}")
    
    print(f"[GRPO] Dataset size: {len(dataset)}")
    
    # Remove the prompt_length column (not needed for training)
    dataset = dataset.remove_columns(["prompt_length"])
    
    return dataset


def train_grpo(config: GRPOExperimentConfig) -> None:
    """
    Main GRPO training function with self-consistency rewards.
    """
    if not HAS_TRL:
        raise ImportError("TRL is required for GRPO training. Install with: pip install trl>=0.9.0")
    
    print("=" * 60)
    print("ExpV1_1: GRPO Training with Self-Consistency Rewards")
    print("=" * 60)
    print(f"Model: {config.model_name_or_path}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"Num generations (group size): {config.num_generations}")
    print(f"Min agreement for consensus: {config.min_agreement}")
    print(f"KL coefficient: {config.kl_coef}")
    print(f"Use vLLM: {config.use_vllm}")
    if config.use_vllm:
        print(f"  vLLM GPU memory: {config.vllm_gpu_memory_utilization}")
        print(f"  vLLM tensor parallel: {config.vllm_tensor_parallel_size}")
    print("=" * 60)
    print("\nReward mechanism: SELF-CONSISTENCY")
    print("  - No ground truth answer required")
    print("  - Rewards based on agreement across rollouts")
    print("  - Modal (majority) answer is treated as 'correct'")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("\n[GRPO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("[GRPO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare dataset
    print("\n[GRPO] Preparing dataset...")
    dataset = prepare_dataset(
        config.dataset_path,
        config.prompt_template,
        tokenizer,
        config.max_prompt_length,
    )
    
    # Create self-consistency reward function
    print("\n[GRPO] Creating self-consistency reward function...")
    reward_fn = create_self_consistency_reward_fn(
        num_generations=config.num_generations,
        correct_reward=config.correct_reward,
        incorrect_reward=config.incorrect_reward,
        format_penalty=config.format_penalty,
        min_agreement=config.min_agreement,
    )
    
    # GRPO Config
    grpo_config_kwargs = dict(
        output_dir=config.output_dir,
        
        # GRPO specific
        num_generations=config.num_generations,
        temperature=config.temperature,
        max_completion_length=config.max_response_length,
        
        # KL penalty
        beta=config.kl_coef,
        
        # Training
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        
        # Precision
        bf16=config.bf16,
        
        # Logging
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        
        # Misc
        seed=config.seed,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        
        # Generation settings
        top_p=config.top_p,
    )
    
    # Add vLLM configuration if enabled
    if config.use_vllm:
        grpo_config_kwargs.update(
            use_vllm=True,
            vllm_mode=config.vllm_mode,
            vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
        )
        if config.vllm_max_model_len:
            grpo_config_kwargs["vllm_max_model_length"] = config.vllm_max_model_len
        print("\n[GRPO] vLLM enabled for fast rollout generation")
    
    grpo_config = GRPOConfig(**grpo_config_kwargs)
    
    # Create trainer
    print("\n[GRPO] Creating GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )
    
    # Train
    print("\n[GRPO] Starting training...")
    trainer.train()
    
    # Save final model
    print(f"\n[GRPO] Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save config
    config_save_path = os.path.join(config.output_dir, "grpo_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(vars(config), f, default_flow_style=False)
    
    print("\n[GRPO] Training complete!")
    print(f"Model saved to: {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ExpV1_1: GRPO Training with Self-Consistency Rewards")
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file",
    )
    
    # Allow command-line overrides
    parser.add_argument("--model", type=str, default=None, help="Model name or path")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset JSONL")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--num-generations", type=int, default=None, help="Rollouts per prompt")
    parser.add_argument("--min-agreement", type=int, default=None, help="Min rollouts for consensus")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=None, help="KL coefficient")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM (use HF generate)")
    parser.add_argument("--vllm-gpu-memory", type=float, default=None, help="vLLM GPU memory utilization")
    parser.add_argument("--vllm-tp", type=int, default=None, help="vLLM tensor parallel size")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = GRPOExperimentConfig()
    
    # Apply command-line overrides
    if args.model:
        config.model_name_or_path = args.model
    if args.dataset:
        config.dataset_path = args.dataset
    if args.output:
        config.output_dir = args.output
    if args.num_generations:
        config.num_generations = args.num_generations
    if args.min_agreement:
        config.min_agreement = args.min_agreement
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.kl_coef:
        config.kl_coef = args.kl_coef
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.temperature:
        config.temperature = args.temperature
    if args.no_vllm:
        config.use_vllm = False
    if args.vllm_gpu_memory:
        config.vllm_gpu_memory_utilization = args.vllm_gpu_memory
    if args.vllm_tp:
        config.vllm_tensor_parallel_size = args.vllm_tp
    
    # Validate
    if not config.dataset_path:
        raise ValueError("Dataset path is required. Use --dataset or set in config.yaml")
    
    if not Path(config.dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")
    
    # Run training
    train_grpo(config)


if __name__ == "__main__":
    main()
