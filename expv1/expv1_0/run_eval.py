from __future__ import annotations

import argparse
from pathlib import Path

from .utils import load_yaml, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="expv1_0: evaluate solver before/after SFT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default="", help="Model path/name to evaluate (overrides config)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = ensure_dir(cfg["io"]["output_dir"])

    model_name = args.model or cfg["sft"]["output_dir"] or cfg["zpd"]["solver_model"]
    if not model_name:
        raise ValueError("No model specified for eval. Use --model or set sft.output_dir / zpd.solver_model.")

    if not bool(cfg["eval"]["enabled"]):
        print("[expv1_0] Eval disabled in config; set eval.enabled=true to run.")
        return

    from euclid.evaluate_math.evaluate_hf import evaluate

    dataset_config = cfg["eval"]["dataset_config"]
    split = cfg["eval"]["split"]
    limit = int(cfg["eval"]["limit"]) or None

    output_path = Path(out_dir) / "eval_results.jsonl"
    evaluate(
        model_name=model_name,
        dataset_config=dataset_config,
        split=split,
        output_path=output_path,
        limit=limit,
        few_shot=False,
        max_tokens=1024,
        temperature=0.0,
        device="auto",
    )


if __name__ == "__main__":
    main()

