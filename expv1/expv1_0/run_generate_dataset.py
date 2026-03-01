from __future__ import annotations

import argparse

from .pipeline_generate import generate_dataset_from_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="expv1_0: generate dataset from KCRG (Pipeline B)")
    parser.add_argument("--config", type=str, required=True, help="Path to expv1_0 config.yaml")
    args = parser.parse_args()

    out_dir = generate_dataset_from_graph(args.config)
    print(f"[expv1_0] Generation pipeline complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()

