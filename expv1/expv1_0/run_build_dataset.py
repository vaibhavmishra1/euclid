from __future__ import annotations

import argparse

from .pipeline import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="expv1_0: build teacher-verified dataset (Method 1 offline)")
    parser.add_argument("--config", type=str, required=True, help="Path to expv1_0 config.yaml")
    args = parser.parse_args()

    out_dir = build_dataset(args.config)
    print(f"[expv1_0] Dataset build complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()

