from __future__ import annotations

import argparse

from .pipeline_concepts import build_concepts_and_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="expv1_0: build concept KB + KCRG artifacts (Pipeline A)")
    parser.add_argument("--config", type=str, required=True, help="Path to expv1_0 config.yaml")
    args = parser.parse_args()

    concepts_dir = build_concepts_and_graph(args.config)
    print(f"[expv1_0] Concept pipeline complete. Artifacts in: {concepts_dir}")


if __name__ == "__main__":
    main()

