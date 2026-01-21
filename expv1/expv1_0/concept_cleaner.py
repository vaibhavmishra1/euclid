"""
Concept Cleaning Pipeline: Pairwise LLM Comparison Approach

Uses vLLM batched inference to compare all concept pairs and determine
which should be merged. Uses Union-Find to build clusters from matches.
"""

from __future__ import annotations

import json
import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .types import Concept
from .utils import write_json


# ============================================================================ #
# Data Structures
# ============================================================================ #


@dataclass
class CleaningResult:
    """Result of the concept cleaning pipeline."""

    # Mapping: original key -> canonical key
    canonical_mapping: Dict[str, str]
    # Canonical concepts with merged counts
    canonical_vocab: List[Dict[str, Any]]
    # Pairwise comparison results (for debugging)
    pairwise_matches: List[Tuple[str, str]]
    # Clusters formed
    clusters: List[List[str]]
    # Stats
    stats: Dict[str, Any]


@dataclass
class ConceptEntry:
    """A concept with its metadata from vocab."""

    key: str
    type: str
    name: str
    count: int


# ============================================================================ #
# Union-Find for clustering
# ============================================================================ #


class UnionFind:
    """Union-Find data structure for building clusters."""

    def __init__(self, items: List[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_clusters(self) -> List[List[str]]:
        clusters: Dict[str, List[str]] = defaultdict(list)
        for item in self.parent:
            root = self.find(item)
            clusters[root].append(item)
        return list(clusters.values())


# ============================================================================ #
# Pairwise LLM Comparison
# ============================================================================ #

PAIRWISE_PROMPT_TEMPLATE = """Are these two mathematical concepts the same thing (just different phrasings, parameterizations, or levels of specificity)?

Concept A: {concept_a}
Concept B: {concept_b}

Rules:
- "prime factorization of 100,000" IS the same as "prime factorization" (specific instance)
- "divisibility rule for 3" IS the same as "divisibility rules" (specific instance)
- "modular inverse" IS the same as "modular multiplicative inverse" (same thing)
- "GCD" and "LCM" are NOT the same (different operations)
- "prime number" and "composite number" are NOT the same (different concepts)

Answer with ONLY "YES" or "NO"."""


def _parse_yes_no(response: str) -> bool:
    """Parse LLM response to boolean."""
    response = response.strip().upper()
    if response.startswith("YES"):
        return True
    return False


def pairwise_llm_comparison(
    concepts: List[ConceptEntry],
    llm_client,
    batch_size: int = 64,
    max_tokens: int = 8,
    temperature: float = 0.0,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """
    Compare all concept pairs using batched LLM inference.

    Args:
        concepts: List of concept entries
        llm_client: LLM client with batch_generate method
        batch_size: Number of pairs per batch
        max_tokens: Max tokens for response (just YES/NO)
        temperature: LLM temperature

    Returns:
        - List of matching pairs (concept_a, concept_b)
        - Stats dict
    """
    # Generate all unique pairs
    names = [c.name for c in concepts]
    pairs = list(itertools.combinations(range(len(names)), 2))
    total_pairs = len(pairs)

    print(f"[Pairwise] Total pairs to compare: {total_pairs:,}")

    # Build prompts for all pairs
    prompts = [
        PAIRWISE_PROMPT_TEMPLATE.format(
            concept_a=names[i],
            concept_b=names[j],
        )
        for i, j in pairs
    ]

    # Batch inference
    matches: List[Tuple[str, str]] = []
    num_batches = (total_pairs + batch_size - 1) // batch_size

    print(f"[Pairwise] Running {num_batches:,} batches (batch_size={batch_size})...")

    for batch_idx in tqdm(range(num_batches), desc="Pairwise comparison"):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_pairs)
        batch_prompts = prompts[start:end]
        batch_pairs = pairs[start:end]

        # Call LLM with batch
        try:
            responses = llm_client.batch_generate(
                batch_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Parse responses
            for (i, j), response in zip(batch_pairs, responses):
                if _parse_yes_no(response):
                    matches.append((names[i], names[j]))

        except Exception as e:
            print(f"[Pairwise] Batch {batch_idx} failed: {e}")
            # Fall back to individual calls if batch fails
            for prompt, (i, j) in zip(batch_prompts, batch_pairs):
                try:
                    response = llm_client.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if _parse_yes_no(response):
                        matches.append((names[i], names[j]))
                except Exception as e2:
                    print(f"[Pairwise] Individual call failed: {e2}")

    stats = {
        "total_pairs": total_pairs,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "num_matches": len(matches),
        "match_rate": len(matches) / total_pairs if total_pairs > 0 else 0,
    }

    print(f"[Pairwise] Found {len(matches):,} matching pairs ({stats['match_rate']:.2%})")

    return matches, stats


# ============================================================================ #
# Main Pipeline
# ============================================================================ #


def clean_concepts(
    vocab_path: str,
    output_dir: str,
    llm_client,
    batch_size: int = 64,
    max_tokens: int = 8,
    temperature: float = 0.0,
) -> CleaningResult:
    """
    Run the pairwise LLM concept cleaning pipeline.

    Args:
        vocab_path: Path to concept_vocab.json
        output_dir: Directory to save cleaned artifacts
        llm_client: LLM client for comparisons
        batch_size: Batch size for LLM inference
        max_tokens: Max tokens per response
        temperature: LLM temperature

    Returns:
        CleaningResult with all outputs
    """
    # Load vocab
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    concepts_raw = vocab_data.get("concepts", [])
    concepts = [
        ConceptEntry(
            key=c["key"],
            type=c["type"],
            name=c["name"],
            count=c["count"],
        )
        for c in concepts_raw
    ]

    print(f"[Clean] Loaded {len(concepts)} concepts from {vocab_path}")

    # ------------------------------------------------------------------ #
    # Pairwise LLM comparison
    # ------------------------------------------------------------------ #
    matches, pairwise_stats = pairwise_llm_comparison(
        concepts=concepts,
        llm_client=llm_client,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # ------------------------------------------------------------------ #
    # Build clusters using Union-Find
    # ------------------------------------------------------------------ #
    print("[Clean] Building clusters from matches...")

    names = [c.name for c in concepts]
    name_to_count = {c.name: c.count for c in concepts}

    uf = UnionFind(names)
    for a, b in matches:
        uf.union(a, b)

    clusters = uf.get_clusters()
    multi_clusters = [c for c in clusters if len(c) > 1]

    print(f"[Clean] Found {len(multi_clusters)} clusters with >1 member")

    # ------------------------------------------------------------------ #
    # Pick canonical name for each cluster
    # ------------------------------------------------------------------ #
    name_to_canonical: Dict[str, str] = {}

    for cluster in clusters:
        # Sort by count (desc), then length (asc), then alphabetically
        sorted_cluster = sorted(
            cluster,
            key=lambda x: (-name_to_count.get(x, 0), len(x), x)
        )
        canonical = sorted_cluster[0]

        for name in cluster:
            name_to_canonical[name] = canonical

    # ------------------------------------------------------------------ #
    # Build final mapping: original_key -> canonical_key
    # ------------------------------------------------------------------ #
    final_mapping: Dict[str, str] = {}
    for c in concepts:
        canonical_name = name_to_canonical.get(c.name, c.name)
        final_mapping[c.key] = f"concept:{canonical_name.lower()}"

    # ------------------------------------------------------------------ #
    # Build final vocab
    # ------------------------------------------------------------------ #
    final_counts: Dict[str, int] = defaultdict(int)
    for c in concepts:
        final_key = final_mapping[c.key]
        final_counts[final_key] += c.count

    canonical_vocab = [
        {
            "key": k,
            "name": k.replace("concept:", ""),
            "count": final_counts[k],
        }
        for k in sorted(final_counts.keys(), key=lambda x: (-final_counts[x], x))
    ]

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    stats = {
        "original_count": len(concepts),
        "final_count": len(canonical_vocab),
        "reduction_ratio": 1 - len(canonical_vocab) / len(concepts) if len(concepts) > 0 else 0,
        "num_clusters": len(clusters),
        "num_multi_clusters": len(multi_clusters),
        "pairwise": pairwise_stats,
    }

    print(f"\n[Summary]")
    print(f"  Original concepts: {stats['original_count']}")
    print(f"  Final count: {stats['final_count']}")
    print(f"  Reduction: {stats['reduction_ratio']:.1%}")
    print(f"  Clusters with merges: {stats['num_multi_clusters']}")

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    write_json(f"{output_dir}/canonical_mapping.json", final_mapping)
    write_json(f"{output_dir}/canonical_vocab.json", {
        "source_vocab": vocab_path,
        "method": "pairwise_llm",
        "stats": stats,
        "concepts": canonical_vocab,
    })
    write_json(f"{output_dir}/pairwise_matches.json", [
        {"a": a, "b": b} for a, b in matches
    ])
    write_json(f"{output_dir}/clusters.json", [
        {"canonical": name_to_canonical.get(c[0], c[0]), "members": c}
        for c in clusters if len(c) > 1
    ])

    return CleaningResult(
        canonical_mapping=final_mapping,
        canonical_vocab=canonical_vocab,
        pairwise_matches=matches,
        clusters=multi_clusters,
        stats=stats,
    )


# ============================================================================ #
# Utility: Apply mapping to seed_concepts.jsonl
# ============================================================================ #


def apply_mapping_to_seed_concepts(
    seed_concepts_path: str,
    mapping: Dict[str, str],
    output_path: str,
):
    """
    Apply canonical mapping to seed_concepts.jsonl.
    """
    from .utils import write_jsonl

    with open(seed_concepts_path, "r") as f:
        rows = [json.loads(line) for line in f]

    transformed = []
    for row in rows:
        new_concepts = []
        seen_keys = set()
        for c in row.get("concepts", []):
            old_key = f"{c['type']}:{c['name'].lower()}"
            new_key = mapping.get(old_key, old_key)
            if new_key not in seen_keys:
                seen_keys.add(new_key)
                new_concepts.append({"canonical_key": new_key})
        transformed.append({
            "problem_id": row.get("problem_id"),
            "domain": row.get("domain"),
            "canonical_concepts": new_concepts,
        })

    write_jsonl(output_path, transformed)
    print(f"Saved canonicalized seed concepts to {output_path}")


def rebuild_graph_with_canonical(
    seed_concepts_path: str,
    mapping: Dict[str, str],
    output_path: str,
    min_cooccurrence: int = 1,
):
    """
    Rebuild the concept graph using canonical keys.
    """
    from .concept_graph import ConceptGraph
    from .utils import write_json

    with open(seed_concepts_path, "r") as f:
        rows = [json.loads(line) for line in f]

    concept_sets: List[List[Concept]] = []
    for row in rows:
        concepts = []
        seen = set()
        for c in row.get("concepts", []):
            old_key = f"{c['type']}:{c['name'].lower()}"
            new_key = mapping.get(old_key, old_key)
            if new_key not in seen:
                seen.add(new_key)
                name = new_key.replace("concept:", "")
                concepts.append(Concept(type="canonical", name=name))
        concept_sets.append(concepts)

    graph = ConceptGraph.build_from_concept_sets(concept_sets, min_cooccurrence=min_cooccurrence)

    write_json(output_path, {
        "min_cooccurrence": min_cooccurrence,
        "adjacency": graph.adjacency,
        "concepts_by_key": {k: {"type": c.type, "name": c.name} for k, c in graph.concepts_by_key.items()},
    })
    print(f"Saved canonical concept graph to {output_path}")
