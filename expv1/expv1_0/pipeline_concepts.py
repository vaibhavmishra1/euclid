from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List

from .concept_extractor import LLMConceptExtractor
from .concept_graph import ConceptGraph
from .llm import build_llm_client
from .seeds import load_seed_examples_from_hendrycks_math, load_seed_examples_from_jsonl
from .types import Concept, SeedExample
from .utils import ensure_dir, load_yaml, write_json, write_jsonl


def build_concepts_and_graph(config_path: str) -> str:
    """
    Pipeline A (GRIP Step 1+2):
    - Load seed problems (problem + full solution)
    - Extract knowledge-point concepts using an LLM
    - Build the explicit Key Concept Relationship Graph (KCRG)
    - Save artifacts for reuse by the generation pipeline

    Returns: concepts_dir (directory containing saved artifacts)
    """

    cfg = load_yaml(config_path)
    random.seed(int(cfg["experiment"]["seed"]))

    out_dir = ensure_dir(cfg["io"]["output_dir"])
    concepts_dir = ensure_dir(cfg.get("io", {}).get("concepts_dir", f"{out_dir}/concepts"))

    # ------------------------------------------------------------------ #
    # A1) Load seeds
    # ------------------------------------------------------------------ #
    seed_source = str(cfg["seed_data"].get("source", "hf")).lower()
    if seed_source == "hf":
        seeds: List[SeedExample] = load_seed_examples_from_hendrycks_math(
            dataset_config=str(cfg["seed_data"]["dataset_config"]),
            split=str(cfg["seed_data"]["split"]),
            max_seeds=int(cfg["seed_data"]["max_seeds"]),
            seed=int(cfg["experiment"]["seed"]),
        )
    elif seed_source == "jsonl":
        seeds = load_seed_examples_from_jsonl(
            cfg["seed_data"]["toy_jsonl"],
            max_seeds=int(cfg["seed_data"]["max_seeds"]),
        )
    else:
        raise ValueError(f"Unknown seed_data.source: {seed_source}")

    # ------------------------------------------------------------------ #
    # A2) Concept extraction (problem + solution)
    # ------------------------------------------------------------------ #
    concept_extractor_vllm = cfg["concept_extraction"].get("vllm", {}) or {}
    concept_client = build_llm_client(
        cfg["concept_extraction"]["concept_extractor_backend"],
        cfg["concept_extraction"]["concept_extractor_model"],
        seed=int(cfg["experiment"]["seed"]),
        vllm_gpu_memory_utilization=float(concept_extractor_vllm.get("gpu_memory_utilization", 0.9)),
        vllm_tensor_parallel_size=int(concept_extractor_vllm.get("tensor_parallel_size", 1)),
    )
    extractor = LLMConceptExtractor(
        llm=concept_client,
        prompt_path=cfg["concept_extraction"]["prompt_path"],
        max_tokens=int(cfg["concept_extraction"].get("max_tokens", 512)),
        temperature=float(cfg["concept_extraction"].get("temperature", 0.0)),
        top_p=float(cfg["concept_extraction"].get("top_p", 1.0)),
    )

    seed_concept_sets: List[List[Concept]] = []
    seed_concepts_rows: List[Dict[str, Any]] = []
    concept_counts: Dict[str, int] = defaultdict(int)
    concepts_by_key: Dict[str, Concept] = {}

    for s in seeds:
        concepts = extractor.extract(s.problem, s.solution)

        # Dedup within a seed concept list (preserve order)
        seen = set()
        uniq: List[Concept] = []
        for c in concepts:
            if c.key in seen:
                continue
            seen.add(c.key)
            uniq.append(c)
        concepts = uniq

        seed_concept_sets.append(concepts)
        seed_concepts_rows.append(
            {
                "problem_id": s.problem_id,
                "domain": s.domain,
                "concepts": [{"type": c.type, "name": c.name} for c in concepts],
            }
        )
        for c in concepts:
            concept_counts[c.key] += 1
            concepts_by_key[c.key] = c

    # ------------------------------------------------------------------ #
    # A3) Build explicit co-occurrence graph
    # ------------------------------------------------------------------ #
    min_co = int(cfg["concept_graph"]["min_cooccurrence"])
    graph = ConceptGraph.build_from_concept_sets(seed_concept_sets, min_cooccurrence=min_co)

    # ------------------------------------------------------------------ #
    # A4) Save artifacts
    # ------------------------------------------------------------------ #
    write_jsonl(f"{concepts_dir}/seeds.jsonl", seeds)
    write_jsonl(f"{concepts_dir}/seed_concepts.jsonl", seed_concepts_rows)

    vocab = [
        {"key": k, "type": concepts_by_key[k].type, "name": concepts_by_key[k].name, "count": int(concept_counts[k])}
        for k in sorted(concept_counts.keys(), key=lambda x: (-concept_counts[x], x))
    ]
    write_json(
        f"{concepts_dir}/concept_vocab.json",
        {
            "dataset": {"source": seed_source, "dataset_config": cfg["seed_data"].get("dataset_config"), "split": cfg["seed_data"].get("split")},
            "num_seeds": len(seeds),
            "min_cooccurrence": min_co,
            "num_unique_concepts": len(vocab),
            "concepts": vocab,
        },
    )

    write_json(
        f"{concepts_dir}/concept_graph.json",
        {
            "min_cooccurrence": min_co,
            "adjacency": graph.adjacency,
            "concepts_by_key": {k: {"type": c.type, "name": c.name} for k, c in graph.concepts_by_key.items()},
        },
    )

    return concepts_dir

