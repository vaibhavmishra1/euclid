from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .utils import load_yaml, read_jsonl, write_jsonl, ensure_dir, read_text, render_template


def _extract_entry_data(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract candidate and verification data from either format:
    - Original format (accepted.jsonl): has 'candidate' at top level
    - Verified format (accepted_verified.jsonl): has 'original_data' containing the original entry
    
    Returns dict with: problem, answer, full_solution, verification_summary (if available)
    """
    # Check if this is the verified format (has 'original_data')
    if "original_data" in r:
        # Verified format: data is nested inside original_data
        original = r.get("original_data", {})
        cand = original.get("candidate", {})
        # In verified format, the COT solution is in original_data.verification.solution
        original_verification = original.get("verification", {})
        full_solution = original_verification.get("solution", "")
        
        # Get verification summary (OpenAI verification results)
        verification_summary = r.get("verification_summary", {})
        verification_status = r.get("verification", {}).get("status", "unknown")
        
        return {
            "problem": cand.get("problem", ""),
            "answer": cand.get("answer", ""),
            "full_solution": full_solution,
            "verification_summary": verification_summary,
            "verification_status": verification_status,
            "difficulty_bucket": r.get("difficulty_bucket", ""),
            "p_succ": original.get("zpd", {}).get("p_succ") or original.get("verification", {}).get("p_succ"),
        }
    else:
        # Original format: data is at top level
        cand = r.get("candidate", {}) if isinstance(r, dict) else {}
        verification = r.get("verification", {}) if isinstance(r, dict) else {}
        zpd = r.get("zpd", {}) if isinstance(r, dict) else {}
        
        return {
            "problem": cand.get("problem", ""),
            "answer": cand.get("answer", ""),
            "full_solution": verification.get("solution", ""),
            "verification_summary": None,  # Not available in original format
            "verification_status": None,
            "difficulty_bucket": "",
            "p_succ": zpd.get("p_succ") or verification.get("p_succ"),
        }


def _passes_filter(
    data: Dict[str, Any],
    require_valid: bool = False,
    require_correct: bool = False,
    min_cot_quality: Optional[str] = None,
    require_no_leakage: bool = False,
    min_overall_quality: Optional[str] = None,
) -> bool:
    """
    Check if an entry passes the verification filters.
    Returns True if filters are not applicable (no verification summary) or if all filters pass.
    """
    summary = data.get("verification_summary")
    
    # If no verification summary, only filter if we have strict requirements
    if not summary:
        # Can't filter without verification data - let it pass unless strict mode
        return True
    
    # Check verification status first
    if data.get("verification_status") != "success":
        return False
    
    # Quality levels for comparison
    cot_quality_levels = {"low": 0, "medium": 1, "high": 2}
    overall_quality_levels = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
    
    # Apply filters
    if require_valid and summary.get("question_validity") != "valid":
        return False
    
    if require_correct and summary.get("answer_correctness") != "correct":
        return False
    
    if require_no_leakage and summary.get("answer_leakage") != "no":
        return False
    
    if min_cot_quality:
        entry_quality = cot_quality_levels.get(summary.get("cot_quality"), -1)
        required_quality = cot_quality_levels.get(min_cot_quality, 0)
        if entry_quality < required_quality:
            return False
    
    if min_overall_quality:
        entry_quality = overall_quality_levels.get(summary.get("overall_quality"), -1)
        required_quality = overall_quality_levels.get(min_overall_quality, 0)
        if entry_quality < required_quality:
            return False
    
    return True


def build_sft_dataset(
    accepted_jsonl: str,
    solver_prompt_path: str,
    out_jsonl: str,
    require_valid: bool = False,
    require_correct: bool = False,
    min_cot_quality: Optional[str] = None,
    require_no_leakage: bool = False,
    min_overall_quality: Optional[str] = None,
) -> Dict[str, int]:
    """
    Build SFT dataset from accepted.jsonl or accepted_verified.jsonl.
    
    Args:
        accepted_jsonl: Path to input JSONL file
        solver_prompt_path: Path to prompt template
        out_jsonl: Path to output JSONL file
        require_valid: Only include questions marked as valid
        require_correct: Only include questions with correct answers
        min_cot_quality: Minimum COT quality (low/medium/high)
        require_no_leakage: Only include questions with no answer leakage
        min_overall_quality: Minimum overall quality (poor/fair/good/excellent)
    
    Returns:
        Dict with statistics: total, included, filtered, has_cot
    """
    prompt_template = read_text(solver_prompt_path)
    rows = read_jsonl(accepted_jsonl)
    out_rows: List[Dict[str, Any]] = []
    
    stats = {"total": 0, "included": 0, "filtered": 0, "has_cot": 0, "has_verification": 0}
    
    for r in rows:
        stats["total"] += 1
        
        # Extract data from either format
        data = _extract_entry_data(r)
        
        if data.get("verification_summary"):
            stats["has_verification"] += 1
        
        # Apply verification filters
        if not _passes_filter(
            data,
            require_valid=require_valid,
            require_correct=require_correct,
            min_cot_quality=min_cot_quality,
            require_no_leakage=require_no_leakage,
            min_overall_quality=min_overall_quality,
        ):
            stats["filtered"] += 1
            continue
        
        problem = data["problem"]
        answer = data["answer"]
        full_solution = data["full_solution"]
        
        if not problem:
            stats["filtered"] += 1
            continue
        
        prompt = render_template(prompt_template, {"problem": problem})
        
        # Use full COT if available, otherwise fall back to boxed answer only
        if full_solution and full_solution.strip():
            completion = full_solution
            stats["has_cot"] += 1
        else:
            completion = f"\\boxed{{{answer}}}"
        
        out_rows.append({
            "prompt": prompt, 
            "completion": completion, 
            "problem": problem, 
            "answer": answer,
            "has_cot": bool(full_solution and full_solution.strip()),
            "p_succ": data.get("p_succ"),
            "difficulty_bucket": data.get("difficulty_bucket"),
        })
        stats["included"] += 1
    
    write_jsonl(out_jsonl, out_rows)
    return stats


def run_sft(
    cfg: Dict[str, Any],
    accepted_path: str | None = None,
    require_valid: bool = False,
    require_correct: bool = False,
    min_cot_quality: Optional[str] = None,
    require_no_leakage: bool = False,
    min_overall_quality: Optional[str] = None,
    # Training hyperparameters (CLI overrides)
    batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_steps: Optional[int] = None,
    num_epochs: Optional[int] = None,
    max_seq_length: Optional[int] = None,
    # Performance options
    use_bf16: bool = True,
    use_flash_attention: bool = True,
    use_torch_compile: bool = False,
    use_gradient_checkpointing: bool = False,
    num_workers: int = 4,
) -> None:
    # Try to find accepted.jsonl in various locations
    if accepted_path is None:
        # First, try the new Pipeline B2 output location
        base_out_dir = cfg["io"]["output_dir"]
        solver_model = str(cfg["zpd"]["solver_model"])
        model_name_sanitized = solver_model.split("/")[-1].replace(" ", "_")
        # Try common ZPD threshold directories
        possible_paths = [
            f"{base_out_dir}_{model_name_sanitized}_accepted/accepted.jsonl",
            f"{base_out_dir}_{model_name_sanitized}_accepted/accepted_verified.jsonl",
            f"{base_out_dir}_{model_name_sanitized}_accepted/accepted_verified_retried.jsonl",
            f"{base_out_dir}_{model_name_sanitized}_accepted_zpd0.1-1.0/accepted.jsonl",
            f"{base_out_dir}_{model_name_sanitized}_accepted_zpd0.2-0.7/accepted.jsonl",
            str(Path(cfg["io"]["output_dir"]) / "accepted.jsonl"),  # Old location
        ]
        
        accepted_path = None
        for path in possible_paths:
            if Path(path).exists():
                accepted_path = path
                break
        
        if accepted_path is None:
            # List available directories to help user
            import os
            base = cfg["io"]["output_dir"]
            if os.path.exists(base):
                dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and "accepted" in d]
                if dirs:
                    raise FileNotFoundError(
                        f"Missing accepted.jsonl. Found these accepted directories: {dirs}\n"
                        f"Please specify --accepted-path or run Pipeline B2 first."
                    )
            raise FileNotFoundError(
                f"Missing accepted dataset. Run Pipeline B2 (run_filter_zpd) first, or specify --accepted-path."
            )
    
    if not Path(accepted_path).exists():
        raise FileNotFoundError(f"Accepted file not found: {accepted_path}")

    print(f"[expv1_0] Loading data from: {accepted_path}")
    
    # Determine output filename based on filters
    filter_suffix = ""
    if require_valid or require_correct or min_cot_quality or require_no_leakage or min_overall_quality:
        filter_parts = []
        if require_valid:
            filter_parts.append("valid")
        if require_correct:
            filter_parts.append("correct")
        if min_cot_quality:
            filter_parts.append(f"cot_{min_cot_quality}")
        if require_no_leakage:
            filter_parts.append("no_leak")
        if min_overall_quality:
            filter_parts.append(f"quality_{min_overall_quality}")
        filter_suffix = "_" + "_".join(filter_parts)
    
    # Save SFT dataset in the same directory as accepted.jsonl
    accepted_dir = Path(accepted_path).parent
    sft_dataset_path = str(accepted_dir / f"sft_dataset{filter_suffix}.jsonl")
    
    stats = build_sft_dataset(
        accepted_jsonl=accepted_path,
        solver_prompt_path=cfg["zpd"]["prompt_path"],
        out_jsonl=sft_dataset_path,
        require_valid=require_valid,
        require_correct=require_correct,
        min_cot_quality=min_cot_quality,
        require_no_leakage=require_no_leakage,
        min_overall_quality=min_overall_quality,
    )
    
    print(f"[expv1_0] SFT Dataset Statistics:")
    print(f"  - Total entries: {stats['total']}")
    print(f"  - With verification: {stats['has_verification']}")
    print(f"  - Filtered out: {stats['filtered']}")
    print(f"  - Included: {stats['included']}")
    print(f"  - With COT: {stats['has_cot']}")
    print(f"  - Output: {sft_dataset_path}")

    if not bool(cfg["sft"]["enabled"]):
        print("[expv1_0] SFT disabled in config; stopping after dataset prep.")
        return

    base_model = cfg["sft"]["base_model"] or cfg["zpd"]["solver_model"] or cfg["generation"]["generator_model"]
    if not base_model:
        raise ValueError("No base_model configured for SFT (set sft.base_model).")

    out_model_dir = ensure_dir(cfg["sft"]["output_dir"])
    
    # Get training hyperparameters (CLI overrides > config > defaults)
    sft_cfg = cfg.get("sft", {})
    _batch_size = batch_size or sft_cfg.get("batch_size", 32)
    _grad_accum = gradient_accumulation_steps or sft_cfg.get("gradient_accumulation_steps", 1)
    _lr = float(learning_rate or sft_cfg.get("learning_rate", 2e-5))
    _max_steps = max_steps or sft_cfg.get("max_steps", -1)
    _num_epochs = num_epochs or sft_cfg.get("num_epochs", 3)
    _max_seq_len = max_seq_length or sft_cfg.get("max_seq_length", 2048)

    # HF SFT trainer with optimizations for large GPUs
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    print(f"\n[expv1_0] Training Configuration:")
    print(f"  - Base model: {base_model}")
    print(f"  - Batch size: {_batch_size}")
    print(f"  - Gradient accumulation: {_grad_accum}")
    print(f"  - Effective batch size: {_batch_size * _grad_accum}")
    print(f"  - Learning rate: {_lr}")
    print(f"  - Max steps: {_max_steps if _max_steps > 0 else 'auto (epochs)'}")
    print(f"  - Num epochs: {_num_epochs}")
    print(f"  - Max sequence length: {_max_seq_len}")
    print(f"  - bf16: {use_bf16}")
    print(f"  - Flash Attention 2: {use_flash_attention}")
    print(f"  - torch.compile: {use_torch_compile}")
    print(f"  - Gradient checkpointing: {use_gradient_checkpointing}")
    print(f"  - Dataloader workers: {num_workers}")
    
    # Load dataset
    ds = load_dataset("json", data_files=sft_dataset_path, split="train")
    print(f"  - Dataset size: {len(ds)} samples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tok(ex):
        text = ex["prompt"] + "\n" + ex["completion"]
        return tokenizer(text, truncation=True, max_length=_max_seq_len, padding="max_length")

    print(f"\n[expv1_0] Tokenizing dataset...")
    ds_tok = ds.map(_tok, remove_columns=ds.column_names, num_proc=num_workers if num_workers > 0 else None)

    # Load model with optimizations
    print(f"\n[expv1_0] Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
    }
    
    # Use bf16 for faster training and lower memory
    if use_bf16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        print("  - Using bfloat16 precision")
    
    # Use Flash Attention 2 for faster attention computation
    if use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  - Using Flash Attention 2")
        except Exception as e:
            print(f"  - Flash Attention 2 not available: {e}")
    
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    
    # Optionally compile model for faster training (PyTorch 2.0+)
    if use_torch_compile and hasattr(torch, "compile"):
        print("  - Compiling model with torch.compile...")
        model = torch.compile(model)

    # Calculate logging/save steps based on dataset size
    total_steps = (len(ds_tok) // (_batch_size * _grad_accum)) * _num_epochs if _max_steps <= 0 else _max_steps
    log_steps = max(1, total_steps // 20)  # Log ~20 times per training
    save_steps = max(1, total_steps // 4)   # Save ~4 checkpoints
    
    print(f"\n[expv1_0] Training plan:")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Log every: {log_steps} steps")
    print(f"  - Save every: {save_steps} steps")

    # Training arguments optimized for large GPUs
    args = TrainingArguments(
        output_dir=out_model_dir,
        # Batch settings
        per_device_train_batch_size=_batch_size,
        gradient_accumulation_steps=_grad_accum,
        # Learning rate
        learning_rate=_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # Steps/epochs
        max_steps=_max_steps if _max_steps > 0 else -1,
        num_train_epochs=_num_epochs if _max_steps <= 0 else 1,
        # Logging & saving
        logging_steps=log_steps,
        save_steps=save_steps,
        save_total_limit=2,
        # Precision
        fp16=False,
        bf16=use_bf16 and torch.cuda.is_available(),
        tf32=True,  # Enable TF32 for faster matmuls on Ampere+ GPUs
        # Dataloader optimization
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2 if num_workers > 0 else None,
        # Memory optimization
        gradient_checkpointing=use_gradient_checkpointing,
        # Other
        report_to=[],
        remove_unused_columns=False,
        seed=cfg.get("experiment", {}).get("seed", 42),
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    print(f"\n[expv1_0] Starting training...")
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=ds_tok, 
        data_collator=collator,
    )
    trainer.train()
    
    print(f"\n[expv1_0] Saving model...")
    trainer.save_model(out_model_dir)
    tokenizer.save_pretrained(out_model_dir)

    print(f"\n[expv1_0] SFT complete!")
    print(f"  - Model saved to: {out_model_dir}")
    print(f"  - To load: AutoModelForCausalLM.from_pretrained('{out_model_dir}')")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="expv1_0: offline SFT on accepted synthetic dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all data from accepted.jsonl (no filtering)
  python -m tree.euclid.expv1.expv1_0.run_sft --config config.yaml
  
  # Use verified data with quality filters
  python -m tree.euclid.expv1.expv1_0.run_sft --config config.yaml \\
      --accepted-path output/accepted_verified.jsonl \\
      --require-valid --require-correct
  
  # High quality only (valid, correct, no leakage, good+ quality)
  python -m tree.euclid.expv1.expv1_0.run_sft --config config.yaml \\
      --accepted-path output/accepted_verified_retried.jsonl \\
      --require-valid --require-correct --require-no-leakage --min-overall-quality good
  
  # Fast training on large GPU (H200/A100)
  python -m tree.euclid.expv1.expv1_0.run_sft --config config.yaml \\
      --batch-size 64 --num-epochs 3 --use-bf16 --use-flash-attention
  
  # Maximum speed on H200 (140GB)
  python -m tree.euclid.expv1.expv1_0.run_sft --config config.yaml \\
      --batch-size 128 --num-epochs 3 --use-bf16 --use-flash-attention --use-torch-compile
        """
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--accepted-path", type=str, default=None, 
                        help="Path to accepted.jsonl or accepted_verified.jsonl (auto-detected if not provided)")
    
    # Verification filters (only apply to verified data)
    filter_group = parser.add_argument_group("Verification Filters (for accepted_verified.jsonl)")
    filter_group.add_argument("--require-valid", action="store_true",
                              help="Only include questions marked as valid")
    filter_group.add_argument("--require-correct", action="store_true",
                              help="Only include questions with correct answers")
    filter_group.add_argument("--min-cot-quality", type=str, choices=["low", "medium", "high"],
                              help="Minimum COT quality level")
    filter_group.add_argument("--require-no-leakage", action="store_true",
                              help="Only include questions with no answer leakage")
    filter_group.add_argument("--min-overall-quality", type=str, choices=["poor", "fair", "good", "excellent"],
                              help="Minimum overall quality level")
    
    # Training hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--batch-size", type=int, default=None,
                             help="Per-device batch size (default: 32, use 64-128 for large GPUs)")
    train_group.add_argument("--gradient-accumulation-steps", type=int, default=None,
                             help="Gradient accumulation steps (default: 1)")
    train_group.add_argument("--learning-rate", type=float, default=None,
                             help="Learning rate (default: 2e-5)")
    train_group.add_argument("--max-steps", type=int, default=None,
                             help="Max training steps (overrides epochs if set)")
    train_group.add_argument("--num-epochs", type=int, default=None,
                             help="Number of training epochs (default: 3)")
    train_group.add_argument("--max-seq-length", type=int, default=None,
                             help="Maximum sequence length (default: 2048)")
    
    # Performance options
    perf_group = parser.add_argument_group("Performance Options (for large GPUs)")
    perf_group.add_argument("--use-bf16", action="store_true", default=True,
                            help="Use bfloat16 precision (default: True, ~2x faster)")
    perf_group.add_argument("--no-bf16", action="store_false", dest="use_bf16",
                            help="Disable bfloat16 (use fp32)")
    perf_group.add_argument("--use-flash-attention", action="store_true", default=True,
                            help="Use Flash Attention 2 (default: True, ~1.5-2x faster)")
    perf_group.add_argument("--no-flash-attention", action="store_false", dest="use_flash_attention",
                            help="Disable Flash Attention 2")
    perf_group.add_argument("--use-torch-compile", action="store_true", default=False,
                            help="Use torch.compile for extra speed (default: False)")
    perf_group.add_argument("--use-gradient-checkpointing", action="store_true", default=False,
                            help="Use gradient checkpointing to reduce memory usage (default: False)")
    perf_group.add_argument("--num-workers", type=int, default=4,
                            help="Number of dataloader workers (default: 4)")
    
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    
    run_sft(
        cfg,
        accepted_path=args.accepted_path,
        require_valid=args.require_valid,
        require_correct=args.require_correct,
        min_cot_quality=args.min_cot_quality,
        require_no_leakage=args.require_no_leakage,
        min_overall_quality=args.min_overall_quality,
        # Training hyperparameters
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        # Performance options
        use_bf16=args.use_bf16,
        use_flash_attention=args.use_flash_attention,
        use_torch_compile=args.use_torch_compile,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

