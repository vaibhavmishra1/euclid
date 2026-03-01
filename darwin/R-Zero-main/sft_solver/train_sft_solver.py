#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for training a solver model on OpenAI validated questions.

This script trains a 4B solver model using the OpenAI validated dataset where each question
has been verified for correctness.

Usage:
    python train_sft_solver.py --config config.yaml
    
Or with command line arguments:
    python train_sft_solver.py \
        --model_name vibhuiitj/darwin_iter2_dataset_verified_matched \
        --data_path /path/to/balanced_questions__darwin_iter2_openai_validated_matched.json \
        --output_dir ./outputs/sft_solver \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # Base model to fine-tune
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    
    # Data configuration
    data_path: str = "/Users/vaibhav/Desktop/brahma/tree/euclid/darwin/question_generation_clustering/balanced_questions__darwin_iter2_openai_validated_matched.json"
    max_seq_length: int = 2048
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Training configuration
    output_dir: str = "./outputs/sft_solver_darwin_iter2"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 30  # ~10% of typical steps; replaces deprecated warmup_ratio
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 100
    eval_strategy: str = "steps"  # Renamed from evaluation_strategy in newer transformers
    
    # System configuration
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    
    # Data split
    test_size: float = 0.05
    seed: int = 42


def load_json_dataset(data_path: str) -> List[Dict]:
    """Load the JSON dataset."""
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} examples")
    return data


def format_prompt(question: str, response: str = None) -> str:
    """
    Format the prompt in the expected format for the model.
    Uses a simple instruction format suitable for math problem solving.
    """
    if response is None:
        # For inference/generation
        prompt = f"""<|im_start|>system
You are a helpful assistant that solves mathematical problems step by step.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    else:
        # For training
        prompt = f"""<|im_start|>system
You are a helpful assistant that solves mathematical problems step by step.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""
    
    return prompt


def preprocess_dataset(data: List[Dict], config: SFTConfig) -> Dataset:
    """
    Preprocess the JSON data into a HuggingFace Dataset.
    
    Each example has:
    - question: the math problem
    - openai_response: the solution from OpenAI (ground truth)
    - openai_match: 1 if verified correct
    """
    print("\n[Preprocess] Converting to HuggingFace dataset...")
    
    # Filter for only matched examples (openai_match == 1)
    matched_data = [item for item in data if item.get('openai_match', 0) == 1]
    print(f"✓ Filtered to {len(matched_data)} verified correct examples (from {len(data)} total)")
    
    # Create formatted examples
    formatted_examples = []
    for item in matched_data:
        question = item.get('question', '')
        openai_response = item.get('openai_response', '')
        
        if not question or not openai_response:
            continue
        
        # Create the full text with prompt template
        text = format_prompt(question, openai_response)
        
        formatted_examples.append({
            'text': text,
            'question': question,
            'answer': item.get('openai_answer', ''),
            'score': item.get('score', 0.0),
            'cluster_id': item.get('cluster_id', -1)
        })
    
    print(f"✓ Created {len(formatted_examples)} formatted examples")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_examples)
    
    # Split into train/validation
    dataset = dataset.train_test_split(test_size=config.test_size, seed=config.seed)
    print(f"✓ Split into {len(dataset['train'])} train and {len(dataset['test'])} validation examples")
    
    return dataset


def tokenize_function(examples, tokenizer, max_seq_length):
    """Tokenize the examples."""
    outputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )
    # DataCollatorForLanguageModeling will create labels from input_ids
    # and properly pad + mask them with -100
    return outputs


def setup_model_and_tokenizer(config: SFTConfig):
    """Setup the model and tokenizer with optional quantization and LoRA."""
    print(f"\n[Model] Loading model: {config.model_name}")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
        padding_side="right"  # Important for training
    )
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
        trust_remote_code=config.trust_remote_code,
        attn_implementation="flash_attention_2" if config.use_flash_attention else "sdpa",
        device_map="auto",
    )
    
    print(f"✓ Model loaded")
    
    # Setup LoRA if enabled
    if config.use_lora:
        print("[Model] Applying LoRA")
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing if specified
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train SFT Solver Model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--data_path", type=str, help="Path to JSON dataset")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    args = parser.parse_args()
    
    # Load config
    config = SFTConfig()
    
    # Override with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_train_epochs:
        config.num_train_epochs = args.num_train_epochs
    if args.per_device_train_batch_size:
        config.per_device_train_batch_size = args.per_device_train_batch_size
    if args.gradient_accumulation_steps:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.use_lora:
        config.use_lora = True
    print("=" * 80)
    print("SFT Solver Training Configuration")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Data: {config.data_path}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA: {config.use_lora}")
    print("=" * 80)
    
    # Load and preprocess dataset
    raw_data = load_json_dataset(config.data_path)
    dataset = preprocess_dataset(raw_data, config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Tokenize dataset
    print("\n[Tokenize] Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing",
    )
    print(f"✓ Tokenization complete")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_steps=config.eval_steps,
        eval_strategy=config.eval_strategy,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to="none",
        logging_first_step=True,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False if config.gradient_checkpointing else None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    trainer.train()
    
    # Save final model
    print("\n[Save] Saving final model...")
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final_model"))
    print(f"✓ Model saved to {os.path.join(config.output_dir, 'final_model')}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
