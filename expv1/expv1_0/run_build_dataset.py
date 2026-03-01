from __future__ import annotations

import argparse

from .pipeline_concepts import build_concepts_and_graph
from .pipeline_generate import generate_dataset_from_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="expv1_0: run Pipeline A + Pipeline B (full end-to-end)")
    parser.add_argument("--config", type=str, required=True, help="Path to expv1_0 config.yaml")
    args = parser.parse_args()

    concepts_dir = build_concepts_and_graph(args.config)
    out_dir = generate_dataset_from_graph(args.config)
    print(f"[expv1_0] Full pipeline complete.\n- Concepts: {concepts_dir}\n- Dataset:  {out_dir}")


if __name__ == "__main__":
    main()

