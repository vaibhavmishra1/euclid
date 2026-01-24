from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .utils import load_yaml, read_jsonl, write_jsonl, ensure_dir, read_text, render_template


def build_sft_dataset(accepted_jsonl: str, solver_prompt_path: str, out_jsonl: str) -> int:
    prompt_template = read_text(solver_prompt_path)
    rows = read_jsonl(accepted_jsonl)
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        cand = r.get("candidate", {}) if isinstance(r, dict) else {}
        problem = cand.get("problem", "")
        answer = cand.get("answer", "")
        
        # Try to get full COT solution from verification field (Pipeline B2 format)
        verification = r.get("verification", {}) if isinstance(r, dict) else {}
        full_solution = verification.get("solution", "")
        
        prompt = render_template(prompt_template, {"problem": problem})
        
        # Use full COT if available, otherwise fall back to boxed answer only
        if full_solution and full_solution.strip():
            # Full solution already contains the reasoning + boxed answer
            completion = full_solution
        else:
            # Fallback: just boxed answer (for backward compatibility)
            completion = f"\\boxed{{{answer}}}"
        
        out_rows.append({
            "prompt": prompt, 
            "completion": completion, 
            "problem": problem, 
            "answer": answer,
            "has_cot": bool(full_solution and full_solution.strip())
        })
    write_jsonl(out_jsonl, out_rows)
    return len(out_rows)


def run_sft(cfg: Dict[str, Any], accepted_path: str | None = None) -> None:
    # Try to find accepted.jsonl in various locations
    if accepted_path is None:
        # First, try the new Pipeline B2 output location
        base_out_dir = cfg["io"]["output_dir"]
        solver_model = str(cfg["zpd"]["solver_model"])
        model_name_sanitized = solver_model.split("/")[-1].replace(" ", "_")
        # Try common ZPD threshold directories
        possible_paths = [
            f"{base_out_dir}_{model_name_sanitized}_accepted/accepted.jsonl",
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

    # Save SFT dataset in the same directory as accepted.jsonl
    accepted_dir = Path(accepted_path).parent
    sft_dataset_path = str(accepted_dir / "sft_dataset.jsonl")
    n = build_sft_dataset(
        accepted_jsonl=accepted_path,
        solver_prompt_path=cfg["zpd"]["prompt_path"],
        out_jsonl=sft_dataset_path,
    )
    print(f"[expv1_0] Prepared SFT dataset with {n} samples at {sft_dataset_path}")

    if not bool(cfg["sft"]["enabled"]):
        print("[expv1_0] SFT disabled in config; stopping after dataset prep.")
        return

    base_model = cfg["sft"]["base_model"] or cfg["zpd"]["solver_model"] or cfg["generation"]["generator_model"]
    if not base_model:
        raise ValueError("No base_model configured for SFT (set sft.base_model).")

    out_model_dir = ensure_dir(cfg["sft"]["output_dir"])
    max_steps = int(cfg["sft"]["max_steps"])

    # Minimal HF SFT trainer (answer-only). For serious training, switch to PEFT/LoRA + proper formatting.
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    ds = load_dataset("json", data_files=sft_dataset_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tok(ex):
        text = ex["prompt"] + "\n" + ex["completion"]
        return tokenizer(text, truncation=True, max_length=2048)

    ds_tok = ds.map(_tok, remove_columns=ds.column_names)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)

    args = TrainingArguments(
        output_dir=out_model_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        max_steps=max_steps,
        logging_steps=10,
        save_steps=max_steps,
        save_total_limit=1,
        fp16=False,
        bf16=False,
        report_to=[],
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds_tok, data_collator=collator)
    trainer.train()
    trainer.save_model(out_model_dir)
    tokenizer.save_pretrained(out_model_dir)

    print(f"[expv1_0] SFT complete. Model saved to: {out_model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="expv1_0: offline SFT on accepted synthetic dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--accepted-path", type=str, default=None, help="Path to accepted.jsonl (auto-detected if not provided)")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    run_sft(cfg, accepted_path=args.accepted_path)


if __name__ == "__main__":
    main()

