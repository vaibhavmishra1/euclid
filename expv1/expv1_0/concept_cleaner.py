"""
Concept Cleaning Pipeline: Canonicalizes and deduplicates extracted concepts.

Three passes:
1. Rule-based normalization (regex patterns) - fast, deterministic
2. Embedding-based clustering (semantic dedup) - catches near-duplicates
3. LLM verification (optional) - resolves ambiguous clusters
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .types import Concept
from .utils import load_yaml, write_json


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
    # Clusters found in Pass 2 (for debugging/inspection)
    embedding_clusters: List[List[str]]
    # LLM verification results (Pass 3)
    llm_verified_merges: List[Dict[str, Any]]
    # Stats
    stats: Dict[str, Any]


@dataclass
class ConceptEntry:
    """A concept with its metadata from vocab."""

    key: str
    type: str
    name: str
    count: int

    @property
    def normalized_name(self) -> str:
        """Lowercase, stripped, single-spaced name."""
        return " ".join(self.name.lower().split())


# ============================================================================ #
# Pass 1: Rule-Based Normalization
# ============================================================================ #

# Pattern rules: (regex_pattern, canonical_name)
# Applied in order; first match wins.
NORMALIZATION_RULES: List[Tuple[str, str]] = [
    # Divisibility rules - collapse all "divisibility rule for N"
    (r"divisibility\s+rules?\s+for\s+\d+(\s+and\s+\d+)?", "divisibility rules"),
    (r"divisibility\s+by\s+\d+(\s+and\s+\d+)?(\s+rule)?", "divisibility"),
    (r"divisibility\s+rules?$", "divisibility rules"),
    (r"divisibility\s+property(\s+of\s+(integers|remainders|consecutive integers))?", "divisibility"),
    (r"divisibility\s+of\s+integers", "divisibility"),
    # Prime numbers
    (r"(definition\s+of\s+)?prime\s+numbers?$", "prime number"),
    (r"even\s+prime\s+number", "prime number"),
    # Relatively prime / coprimality
    (r"(pairwise\s+)?relatively\s+prime(\s+(numbers|integers))?", "coprimality"),
    (r"coprime(\s+integers)?", "coprimality"),
    # Euler's theorem and totient function (separate concepts)
    (r"euler'?s\s+totient\s+(theorem|function)", "euler's totient function"),
    (r"fermat-euler'?s\s+theorem", "euler's theorem"),
    (r"euler'?s\s+theorem(\s+\(.*\))?", "euler's theorem"),
    # Modular inverse variants
    (r"(modular\s+)?(multiplicative\s+)?inverse(\s+in\s+modular\s+arithmetic)?", "modular inverse"),
    (r"modular\s+multiplicative\s+inverse", "modular inverse"),
    # Modular arithmetic (core concept)
    (r"modular\s+arithmetic(\s+(properties|property|congruence|addition))?", "modular arithmetic"),
    (r"modular\s+arithmetic\s+for\s+.*", "modular arithmetic"),
    (r"modular\s+(addition|multiplication|subtraction)", "modular arithmetic"),
    (r"modular\s+equivalence", "modular congruence"),
    (r"modular\s+exponentiation", "modular exponentiation"),
    # Congruence
    (r"congruence\s+(modulo\s+n|properties?|property\s+of\s+.*)", "modular congruence"),
    (r"congruence\s+properties?\s+of\s+modular\s+arithmetic", "modular congruence"),
    # Prime factorization
    (r"prime\s+factorization(\s+and\s+divisors\s+relationship)?", "prime factorization"),
    (r"uniqueness\s+of\s+prime\s+factorization", "prime factorization"),
    # Chinese Remainder Theorem
    (r"chinese\s+remainder\s+theorem", "chinese remainder theorem"),
    # GCD / LCM
    (r"greatest\s+common\s+divisor(\s+\(gcd\))?", "gcd"),
    (r"gcd(\s+\(greatest\s+common\s+divisor\))?", "gcd"),
    (r"least\s+common\s+multiple(\s+\(lcm\))?", "lcm"),
    (r"lcm(\s+\(least\s+common\s+multiple\))?", "lcm"),
    # Euclidean algorithm
    (r"(extended\s+)?euclidean\s+algorithm", "euclidean algorithm"),
    # Fermat's little theorem
    (r"fermat'?s\s+(little\s+)?theorem", "fermat's little theorem"),
    # Base conversion
    (r"base\s+conversion(\s+formula)?", "base conversion"),
    (r"positional\s+notation(\s+in\s+base\s+\d+)?", "positional notation"),
    # Factorial
    (r"factorial(\s+function)?", "factorial"),
    (r"legendre'?s\s+formula(\s+for\s+factorials)?", "legendre's formula"),
    # Divisor function / number of divisors
    (r"(number\s+of\s+)?divisors?\s+(function\s+)?formula", "divisor function"),
    (r"number\s+of\s+divisors(\s+of\s+.*)?", "divisor counting"),
    (r"divisors?\s+from\s+prime\s+factorization", "divisor counting"),
    # Perfect squares/cubes
    (r"perfect\s+square", "perfect square"),
    (r"perfect\s+cube", "perfect cube"),
    # Pigeonhole principle
    (r"pigeonhole\s+principle", "pigeonhole principle"),
    # Floor/ceiling
    (r"floor\s+function", "floor function"),
    (r"ceiling\s+function", "ceiling function"),
    # Sum of digits
    (r"sum\s+of\s+digits(\s+property)?", "digit sum"),
    (r"digit\s+sum", "digit sum"),
    # Periodicity / cycles
    (r"(cyclic\s+nature|periodicity)(\s+of\s+.*)?(\s+in\s+modular\s+arithmetic)?", "periodicity"),
    (r"cyclic\s+pattern(\s+of\s+.*)?", "periodicity"),
]


def _normalize_name_pass1(name: str) -> str:
    """Apply rule-based normalization to a concept name."""
    normalized = " ".join(name.lower().split())

    for pattern, canonical in NORMALIZATION_RULES:
        if re.fullmatch(pattern, normalized):
            return canonical

    return normalized


def pass1_rule_based(concepts: List[ConceptEntry]) -> Dict[str, str]:
    """
    Pass 1: Rule-based normalization.

    Returns mapping: original_key -> canonical_key (type-agnostic)
    """
    mapping: Dict[str, str] = {}

    for c in concepts:
        canonical_name = _normalize_name_pass1(c.name)
        # We ignore type for canonicalization - just use the name
        canonical_key = f"concept:{canonical_name}"
        mapping[c.key] = canonical_key

    return mapping


# ============================================================================ #
# Pass 2: Embedding-Based Clustering
# ============================================================================ #


def _load_embedding_model(model_name: str):
    """Load sentence transformer model (lazy import)."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding-based dedup. "
            "Install with: pip install sentence-transformers"
        )


def _compute_embeddings(model, names: List[str]) -> np.ndarray:
    """Compute embeddings for a list of concept names."""
    return model.encode(names, convert_to_numpy=True, show_progress_bar=True)


def _find_clusters(
    embeddings: np.ndarray,
    names: List[str],
    threshold: float = 0.85,
) -> List[List[int]]:
    """
    Find clusters of similar concepts using cosine similarity.

    Uses Union-Find to group concepts with similarity > threshold.
    """
    n = len(names)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute pairwise cosine similarities
    similarities = normalized @ normalized.T

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build clusters based on threshold
    for i in range(n):
        for j in range(i + 1, n):
            if similarities[i, j] > threshold:
                union(i, j)

    # Group by root
    clusters_dict: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters_dict[find(i)].append(i)

    # Filter to clusters with > 1 member
    return [indices for indices in clusters_dict.values() if len(indices) > 1]


def _pick_canonical_from_cluster(
    cluster_indices: List[int],
    names: List[str],
    counts: List[int],
) -> str:
    """Pick the canonical name from a cluster (highest count, then shortest)."""
    candidates = [(names[i], counts[i]) for i in cluster_indices]
    # Sort by count (desc), then by length (asc), then alphabetically
    candidates.sort(key=lambda x: (-x[1], len(x[0]), x[0]))
    return candidates[0][0]


def pass2_embedding_clustering(
    canonical_names: List[str],
    counts: List[int],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.85,
) -> Tuple[Dict[str, str], List[List[str]]]:
    """
    Pass 2: Embedding-based clustering.

    Args:
        canonical_names: List of canonical names from Pass 1
        counts: Corresponding counts for each name
        embedding_model: Sentence transformer model name
        threshold: Cosine similarity threshold for clustering

    Returns:
        - Mapping: name -> cluster_canonical_name
        - List of clusters (for inspection)
    """
    if len(canonical_names) == 0:
        return {}, []

    # Load model and compute embeddings
    model = _load_embedding_model(embedding_model)
    embeddings = _compute_embeddings(model, canonical_names)

    # Find clusters
    clusters = _find_clusters(embeddings, canonical_names, threshold)

    # Build mapping
    mapping: Dict[str, str] = {name: name for name in canonical_names}
    cluster_names: List[List[str]] = []

    for cluster_indices in clusters:
        canonical = _pick_canonical_from_cluster(cluster_indices, canonical_names, counts)
        cluster_group = [canonical_names[i] for i in cluster_indices]
        cluster_names.append(cluster_group)

        for idx in cluster_indices:
            mapping[canonical_names[idx]] = canonical

    return mapping, cluster_names


# ============================================================================ #
# Pass 3: LLM Verification
# ============================================================================ #

LLM_MERGE_PROMPT_TEMPLATE = """You are a mathematics concept expert. Given a cluster of potentially related mathematical concepts, determine if they should be merged into a single canonical concept.

## Concepts in cluster:
{concepts_list}

## Instructions:
1. If these concepts are essentially the same thing (just different phrasings, parameterizations, or levels of specificity), they should be MERGED.
2. If they are genuinely different mathematical concepts that happen to be related, they should be KEPT SEPARATE.

## Response format (strict JSON):
{{
  "decision": "merge" | "separate",
  "canonical_name": "<the best canonical name if merging, else null>",
  "reasoning": "<brief explanation>"
}}

Respond with ONLY the JSON, no other text.
"""


def pass3_llm_verification(
    clusters: List[List[str]],
    llm_client,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Pass 3: LLM verification of ambiguous clusters.

    Args:
        clusters: List of clusters from Pass 2
        llm_client: LLM client with a `generate(prompt, ...)` method
        max_tokens: Max tokens for LLM response
        temperature: LLM temperature

    Returns:
        - Mapping: name -> verified_canonical_name (or unchanged)
        - List of verification results (for logging)
    """
    mapping: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []

    for cluster in clusters:
        if len(cluster) <= 1:
            continue

        # Format cluster for prompt
        concepts_list = "\n".join(f"- {name}" for name in cluster)
        prompt = LLM_MERGE_PROMPT_TEMPLATE.format(concepts_list=concepts_list)

        # Call LLM
        try:
            response = llm_client.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Parse JSON response
            response_text = response.strip()
            # Handle markdown code blocks
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\n?", "", response_text)
                response_text = re.sub(r"\n?```$", "", response_text)

            parsed = json.loads(response_text)
            decision = parsed.get("decision", "separate")
            canonical = parsed.get("canonical_name")
            reasoning = parsed.get("reasoning", "")

            result = {
                "cluster": cluster,
                "decision": decision,
                "canonical_name": canonical,
                "reasoning": reasoning,
            }
            results.append(result)

            if decision == "merge" and canonical:
                for name in cluster:
                    mapping[name] = canonical
            else:
                # Keep separate - identity mapping
                for name in cluster:
                    mapping[name] = name

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # On error, keep cluster as-is (conservative)
            results.append({
                "cluster": cluster,
                "decision": "error",
                "error": str(e),
            })
            for name in cluster:
                mapping[name] = name

    return mapping, results


# ============================================================================ #
# Main Pipeline
# ============================================================================ #


def clean_concepts(
    vocab_path: str,
    output_dir: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_threshold: float = 0.85,
    llm_client=None,
    run_pass2: bool = True,
    run_pass3: bool = False,
) -> CleaningResult:
    """
    Run the full concept cleaning pipeline.

    Args:
        vocab_path: Path to concept_vocab.json
        output_dir: Directory to save cleaned artifacts
        embedding_model: Model for Pass 2
        embedding_threshold: Similarity threshold for Pass 2
        llm_client: LLM client for Pass 3 (optional)
        run_pass2: Whether to run embedding clustering
        run_pass3: Whether to run LLM verification

    Returns:
        CleaningResult with all outputs
    """
    from pathlib import Path

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

    print(f"[Pass 0] Loaded {len(concepts)} concepts from {vocab_path}")

    # ------------------------------------------------------------------ #
    # Pass 1: Rule-based normalization
    # ------------------------------------------------------------------ #
    print("[Pass 1] Running rule-based normalization...")
    pass1_mapping = pass1_rule_based(concepts)

    # Build intermediate canonical vocab (aggregate counts)
    canonical_counts: Dict[str, int] = defaultdict(int)
    for c in concepts:
        canonical_key = pass1_mapping[c.key]
        canonical_counts[canonical_key] += c.count

    canonical_names = list(canonical_counts.keys())
    counts = [canonical_counts[name] for name in canonical_names]

    print(f"[Pass 1] Reduced to {len(canonical_names)} canonical concepts")

    # ------------------------------------------------------------------ #
    # Pass 2: Embedding-based clustering
    # ------------------------------------------------------------------ #
    pass2_mapping: Dict[str, str] = {name: name for name in canonical_names}
    embedding_clusters: List[List[str]] = []

    if run_pass2 and len(canonical_names) > 1:
        print("[Pass 2] Running embedding-based clustering...")
        # Extract just the name part from "concept:name" keys
        just_names = [k.replace("concept:", "") for k in canonical_names]
        pass2_name_mapping, embedding_clusters = pass2_embedding_clustering(
            just_names,
            counts,
            embedding_model=embedding_model,
            threshold=embedding_threshold,
        )
        # Map back to full keys
        for i, name in enumerate(canonical_names):
            just_name = just_names[i]
            new_name = pass2_name_mapping.get(just_name, just_name)
            pass2_mapping[name] = f"concept:{new_name}"

        # Update canonical names and counts after Pass 2
        new_canonical_counts: Dict[str, int] = defaultdict(int)
        for name, count in zip(canonical_names, counts):
            new_key = pass2_mapping[name]
            new_canonical_counts[new_key] += count

        canonical_names = list(new_canonical_counts.keys())
        counts = [new_canonical_counts[name] for name in canonical_names]

        print(f"[Pass 2] Found {len(embedding_clusters)} clusters, reduced to {len(canonical_names)} concepts")

    # ------------------------------------------------------------------ #
    # Pass 3: LLM verification (optional)
    # ------------------------------------------------------------------ #
    pass3_mapping: Dict[str, str] = {name: name for name in canonical_names}
    llm_results: List[Dict[str, Any]] = []

    if run_pass3 and llm_client is not None and len(embedding_clusters) > 0:
        print("[Pass 3] Running LLM verification...")
        # Only verify clusters that still have > 1 member after Pass 2
        remaining_clusters = [
            cluster for cluster in embedding_clusters
            if len(set(pass2_mapping.get(f"concept:{n}", f"concept:{n}") for n in cluster)) > 1
        ]

        if remaining_clusters:
            pass3_mapping, llm_results = pass3_llm_verification(
                [[n for n in cluster] for cluster in remaining_clusters],
                llm_client,
            )

            # Update canonical names after Pass 3
            final_canonical_counts: Dict[str, int] = defaultdict(int)
            for name, count in zip(canonical_names, counts):
                just_name = name.replace("concept:", "")
                new_name = pass3_mapping.get(just_name, just_name)
                final_key = f"concept:{new_name}"
                final_canonical_counts[final_key] += count

            canonical_names = list(final_canonical_counts.keys())
            counts = [final_canonical_counts[name] for name in canonical_names]

        print(f"[Pass 3] Verified {len(llm_results)} clusters, final count: {len(canonical_names)} concepts")

    # ------------------------------------------------------------------ #
    # Build final mapping: original_key -> final_canonical_key
    # ------------------------------------------------------------------ #
    final_mapping: Dict[str, str] = {}
    for c in concepts:
        # Pass 1
        p1_key = pass1_mapping[c.key]
        # Pass 2
        p2_key = pass2_mapping.get(p1_key, p1_key)
        # Pass 3
        just_name = p2_key.replace("concept:", "")
        p3_name = pass3_mapping.get(just_name, just_name)
        final_key = f"concept:{p3_name}"
        final_mapping[c.key] = final_key

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
        "after_pass1": len(set(pass1_mapping.values())),
        "after_pass2": len(set(pass2_mapping.values())) if run_pass2 else None,
        "after_pass3": len(canonical_vocab) if run_pass3 else None,
        "final_count": len(canonical_vocab),
        "reduction_ratio": 1 - len(canonical_vocab) / len(concepts) if len(concepts) > 0 else 0,
        "num_embedding_clusters": len(embedding_clusters),
        "num_llm_verified": len(llm_results),
    }

    print(f"\n[Summary]")
    print(f"  Original concepts: {stats['original_count']}")
    print(f"  After Pass 1 (rules): {stats['after_pass1']}")
    if run_pass2:
        print(f"  After Pass 2 (embeddings): {stats['after_pass2']}")
    if run_pass3:
        print(f"  After Pass 3 (LLM): {stats['after_pass3']}")
    print(f"  Final count: {stats['final_count']}")
    print(f"  Reduction: {stats['reduction_ratio']:.1%}")

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    write_json(f"{output_dir}/canonical_mapping.json", final_mapping)
    write_json(f"{output_dir}/canonical_vocab.json", {
        "source_vocab": vocab_path,
        "passes": {
            "pass1": True,
            "pass2": run_pass2,
            "pass3": run_pass3,
        },
        "embedding_model": embedding_model if run_pass2 else None,
        "embedding_threshold": embedding_threshold if run_pass2 else None,
        "stats": stats,
        "concepts": canonical_vocab,
    })

    if embedding_clusters:
        write_json(f"{output_dir}/embedding_clusters.json", embedding_clusters)

    if llm_results:
        write_json(f"{output_dir}/llm_verification.json", llm_results)

    return CleaningResult(
        canonical_mapping=final_mapping,
        canonical_vocab=canonical_vocab,
        embedding_clusters=embedding_clusters,
        llm_verified_merges=llm_results,
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

    Transforms each seed's concepts to use canonical keys.
    """
    from .utils import write_jsonl

    with open(seed_concepts_path, "r") as f:
        rows = [json.loads(line) for line in f]

    transformed = []
    for row in rows:
        new_concepts = []
        seen_keys: Set[str] = set()
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

    # Build concept sets with canonical keys
    concept_sets: List[List[Concept]] = []
    for row in rows:
        concepts = []
        seen: Set[str] = set()
        for c in row.get("concepts", []):
            old_key = f"{c['type']}:{c['name'].lower()}"
            new_key = mapping.get(old_key, old_key)
            if new_key not in seen:
                seen.add(new_key)
                # Create a Concept with type="canonical" and name=the canonical name
                name = new_key.replace("concept:", "")
                concepts.append(Concept(type="canonical", name=name))
        concept_sets.append(concepts)

    # Build graph
    graph = ConceptGraph.build_from_concept_sets(concept_sets, min_cooccurrence=min_cooccurrence)

    write_json(output_path, {
        "min_cooccurrence": min_cooccurrence,
        "adjacency": graph.adjacency,
        "concepts_by_key": {k: {"type": c.type, "name": c.name} for k, c in graph.concepts_by_key.items()},
    })
    print(f"Saved canonical concept graph to {output_path}")
