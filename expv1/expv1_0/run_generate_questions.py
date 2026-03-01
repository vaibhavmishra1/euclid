from __future__ import annotations

import argparse

from .pipeline_generate_questions import generate_questions_from_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="expv1_0: generate questions only (Pipeline B1)")
    parser.add_argument("--config", type=str, required=True, help="Path to expv1_0 config.yaml")
    args = parser.parse_args()

    out_dir = generate_questions_from_graph(args.config)
    print(f"[expv1_0] Question generation complete. Outputs in: {out_dir}")
    print(f"[expv1_0] Next step: Run Pipeline B2 to filter with ZPD:")
    print(f"  python -m tree.euclid.expv1.expv1_0.run_filter_zpd --config {args.config}")


if __name__ == "__main__":
    main()
