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
        prompt = render_template(prompt_template, {"problem": problem})
        completion = f"\\boxed{{{answer}}}"
        out_rows.append({"prompt": prompt, "completion": completion, "problem": problem, "answer": answer})
    write_jsonl(out_jsonl, out_rows)
    return len(out_rows)


def run_sft(cfg: Dict[str, Any]) -> None:
    out_dir = ensure_dir(cfg["io"]["output_dir"])
    accepted_path = str(Path(out_dir) / "accepted.jsonl")
    if not Path(accepted_path).exists():
        raise FileNotFoundError(f"Missing accepted dataset at {accepted_path}. Run run_build_dataset first.")

    sft_dataset_path = str(Path(out_dir) / "sft_dataset.jsonl")
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
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    run_sft(cfg)


if __name__ == "__main__":
    main()

