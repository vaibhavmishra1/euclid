#!/usr/bin/env python3
"""
Standalone script to clean/canonicalize extracted concepts.

Usage:
    # Pass 1 only (regex rules - fast, no dependencies):
    python -m tree.euclid.expv1.expv1_0.run_clean_concepts \
        --config tree/euclid/expv1/expv1_0/config.yaml \
        --pass1-only

    # Pass 1 + Pass 2 (requires sentence-transformers):
    python -m tree.euclid.expv1.expv1_0.run_clean_concepts \
        --config tree/euclid/expv1/expv1_0/config.yaml

    # Pass 1 + Pass 2 + Pass 3 (requires LLM):
    python -m tree.euclid.expv1.expv1_0.run_clean_concepts \
        --config tree/euclid/expv1/expv1_0/config.yaml \
        --run-pass3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Clean and canonicalize extracted concepts")
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
        "--pass1-only",
        action="store_true",
        help="Run only Pass 1 (regex rules). Fast, no ML dependencies.",
    )
    parser.add_argument(
        "--run-pass3",
        action="store_true",
        help="Also run Pass 3 (LLM verification). Requires LLM client.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for Pass 2",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for clustering (default: 0.85)",
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
    print()

    # Determine which passes to run
    run_pass2 = not args.pass1_only
    run_pass3 = args.run_pass3

    # Build LLM client for Pass 3 if needed
    llm_client = None
    if run_pass3:
        print("Setting up LLM client for Pass 3...")
        # Use the concept extractor LLM from config (or could add separate config)
        ce_cfg = cfg.get("concept_extraction", {})
        vllm_cfg = ce_cfg.get("vllm", {}) or {}
        llm_client = build_llm_client(
            backend=ce_cfg.get("concept_extractor_backend", "vllm"),
            model_name=ce_cfg.get("concept_extractor_model", "Qwen/Qwen2.5-32B-Instruct"),
            seed=int(cfg["experiment"]["seed"]),
            vllm_gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.9)),
            vllm_tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
        )

    # Run cleaning
    result = clean_concepts(
        vocab_path=vocab_path,
        output_dir=output_dir,
        embedding_model=args.embedding_model,
        embedding_threshold=args.embedding_threshold,
        llm_client=llm_client,
        run_pass2=run_pass2,
        run_pass3=run_pass3,
    )

    print(f"\nSaved artifacts to {output_dir}/")
    print("  - canonical_mapping.json  (original_key -> canonical_key)")
    print("  - canonical_vocab.json    (deduplicated vocab with merged counts)")
    if result.embedding_clusters:
        print("  - embedding_clusters.json (clusters found in Pass 2)")
    if result.llm_verified_merges:
        print("  - llm_verification.json   (Pass 3 results)")

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
