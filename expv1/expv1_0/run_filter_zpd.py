from __future__ import annotations

import argparse

from .pipeline_filter_zpd import filter_questions_with_zpd


def main() -> None:
    parser = argparse.ArgumentParser(description="expv1_0: filter questions with ZPD and generate COT (Pipeline B2)")
    parser.add_argument("--config", type=str, required=True, help="Path to expv1_0 config.yaml")
    parser.add_argument("--questions", type=str, default=None, help="Path to generated_questions.jsonl (auto-detected if not provided)")
    parser.add_argument("--p-min", type=float, default=None, help="Minimum p_succ threshold (overrides config)")
    parser.add_argument("--p-max", type=float, default=None, help="Maximum p_succ threshold (overrides config)")
    args = parser.parse_args()

    out_dir = filter_questions_with_zpd(args.config, args.questions, args.p_min, args.p_max)
    print(f"[expv1_0] ZPD filtering complete. Outputs in: {out_dir}")
    print(f"[expv1_0] Next step: Run Pipeline B3 (SFT):")
    print(f"  python -m tree.euclid.expv1.expv1_0.run_sft --config {args.config}")


if __name__ == "__main__":
    main()
