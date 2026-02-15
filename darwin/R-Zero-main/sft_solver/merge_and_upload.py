#!/usr/bin/env python3
"""
Merge LoRA weights with base model and optionally upload to HuggingFace Hub.

Usage:
    python merge_and_upload.py \
        --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --adapter_path ./outputs/sft_solver_darwin_iter2/final_model \
        --output_path ./merged_model \
        --push_to_hub \
        --repo_name vibhuiitj/darwin_iter2_solver_sft
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    repo_name: str = None,
):
    """
    Merge LoRA adapter with base model and save/upload.
    
    Args:
        base_model_path: Path or name of the base model
        adapter_path: Path to the LoRA adapter
        output_path: Where to save the merged model
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_name: HuggingFace repo name (required if push_to_hub=True)
    """
    print("=" * 80)
    print("Merging LoRA Adapter with Base Model")
    print("=" * 80)
    print(f"Base model: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")
    if push_to_hub:
        print(f"Will push to: {repo_name}")
    print("=" * 80)
    
    # Load base model
    print("\n[1/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✓ Base model loaded")
    
    # Load LoRA adapter
    print("\n[2/5] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✓ LoRA adapter loaded")
    
    # Merge weights
    print("\n[3/5] Merging weights...")
    merged_model = model.merge_and_unload()
    print("✓ Weights merged")
    
    # Load tokenizer
    print("\n[4/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    print("✓ Tokenizer loaded")
    
    # Save merged model
    print("\n[5/5] Saving merged model...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Model saved to {output_path}")
    
    # Upload to HuggingFace Hub if requested
    if push_to_hub:
        if not repo_name:
            raise ValueError("repo_name is required when push_to_hub=True")
        
        print(f"\n[Upload] Pushing to HuggingFace Hub: {repo_name}")
        
        # Push model
        merged_model.push_to_hub(
            repo_name,
            private=False,
            safe_serialization=True,
        )
        print("✓ Model pushed")
        
        # Push tokenizer
        tokenizer.push_to_hub(repo_name)
        print("✓ Tokenizer pushed")
        
        print(f"\n✅ Model available at: https://huggingface.co/{repo_name}")
    
    print("\n" + "=" * 80)
    print("Merge Complete!")
    print("=" * 80)
    
    # Print model info
    print("\nModel Information:")
    print(f"  Parameter count: {sum(p.numel() for p in merged_model.parameters()):,}")
    print(f"  Memory usage: {merged_model.get_memory_footprint() / 1e9:.2f} GB")
    
    return merged_model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or name of the base model"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./merged_model",
        help="Where to save the merged model"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        help="HuggingFace repository name (e.g., vibhuiitj/darwin_iter2_solver_sft)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.push_to_hub and not args.repo_name:
        parser.error("--repo_name is required when --push_to_hub is set")
    
    # Merge and save/upload
    merge_and_save(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        repo_name=args.repo_name,
    )


if __name__ == "__main__":
    main()
