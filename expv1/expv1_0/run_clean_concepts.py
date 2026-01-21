#!/usr/bin/env python3
"""
Standalone script to clean/canonicalize extracted concepts using pairwise LLM comparison.

Usage:
    python -m tree.euclid.expv1.expv1_0.run_clean_concepts \
        --config tree/euclid/expv1/expv1_0/config.yaml

    # With custom batch size:
    python -m tree.euclid.expv1.expv1_0.run_clean_concepts \
        --config tree/euclid/expv1/expv1_0/config.yaml \
        --batch-size 128

    # Rebuild graph after cleaning:
    python -m tree.euclid.expv1.expv1_0.run_clean_concepts \
        --config tree/euclid/expv1/expv1_0/config.yaml \
        --rebuild-graph
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Clean and canonicalize extracted concepts using pairwise LLM comparison")
    parser.add_argument(
        "--config",
        type=str,
        default="tree/euclid/expv1/expv1_0/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Override path to concept_vocab.json (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for cleaned artifacts (default: concepts_dir/cleaned)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for LLM inference (default: 64)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model for pairwise comparison (default: uses concept_extractor_model from config)",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=None,
        help="Override GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="Also rebuild the concept graph using canonical keys",
    )

    args = parser.parse_args()

    # Import here to avoid slow imports if just checking --help
    from .concept_cleaner import (
        apply_mapping_to_seed_concepts,
        clean_concepts,
        rebuild_graph_with_canonical,
    )
    from .llm import build_llm_client
    from .utils import load_yaml

    # Load config
    cfg = load_yaml(args.config)

    # Determine paths
    concepts_dir = cfg.get("io", {}).get("concepts_dir", f"{cfg['io']['output_dir']}/concepts")
    vocab_path = args.vocab_path or f"{concepts_dir}/concept_vocab.json"
    output_dir = args.output_dir or f"{concepts_dir}/cleaned"

    if not Path(vocab_path).exists():
        print(f"ERROR: concept_vocab.json not found at {vocab_path}")
        print("Run Pipeline A (run_build_concepts.py) first to extract concepts.")
        sys.exit(1)

    print(f"Input vocab: {vocab_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Batch size:  {args.batch_size}")
    print()

    # Build LLM client for pairwise comparison
    print("Loading LLM for pairwise comparison...")
    
    # Use concept_extractor config or overrides
    ce_cfg = cfg.get("concept_extraction", {})
    vllm_cfg = ce_cfg.get("vllm", {}) or {}
    
    model_name = args.model or ce_cfg.get("concept_extractor_model", "Qwen/Qwen2.5-7B-Instruct")
    gpu_memory = args.gpu_memory or float(vllm_cfg.get("gpu_memory_utilization", 0.9))
    
    print(f"Model: {model_name}")
    print(f"GPU memory utilization: {gpu_memory}")
    print()
    
    llm_client = build_llm_client(
        backend=ce_cfg.get("concept_extractor_backend", "vllm"),
        model_name=model_name,
        seed=int(cfg["experiment"]["seed"]),
        vllm_gpu_memory_utilization=gpu_memory,
        vllm_tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
    )

    # Run cleaning
    result = clean_concepts(
        vocab_path=vocab_path,
        output_dir=output_dir,
        llm_client=llm_client,
        batch_size=args.batch_size,
        max_tokens=8,  # Just YES/NO
        temperature=0.0,
    )

    print(f"\nSaved artifacts to {output_dir}/")
    print("  - canonical_mapping.json  (original_key -> canonical_key)")
    print("  - canonical_vocab.json    (deduplicated vocab with merged counts)")
    print("  - pairwise_matches.json   (pairs that LLM said are the same)")
    print("  - clusters.json           (merged clusters)")

    # Optionally apply mapping to seed_concepts and rebuild graph
    if args.rebuild_graph:
        seed_concepts_path = f"{concepts_dir}/seed_concepts.jsonl"
        if Path(seed_concepts_path).exists():
            print("\nApplying mapping to seed_concepts.jsonl...")
            apply_mapping_to_seed_concepts(
                seed_concepts_path=seed_concepts_path,
                mapping=result.canonical_mapping,
                output_path=f"{output_dir}/seed_concepts_canonical.jsonl",
            )

            print("Rebuilding concept graph with canonical keys...")
            min_co = int(cfg.get("concept_graph", {}).get("min_cooccurrence", 1))
            rebuild_graph_with_canonical(
                seed_concepts_path=seed_concepts_path,
                mapping=result.canonical_mapping,
                output_path=f"{output_dir}/concept_graph_canonical.json",
                min_cooccurrence=min_co,
            )
        else:
            print(f"\nWARNING: seed_concepts.jsonl not found at {seed_concepts_path}")
            print("Cannot rebuild graph without seed concepts.")

    print("\nDone!")


if __name__ == "__main__":
    main()
