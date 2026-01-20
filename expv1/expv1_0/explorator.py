from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

from .concept_graph import ConceptGraph
from .types import Concept, Spec


def _ensure_type_mix(concepts: List[Concept]) -> List[Concept]:
    """Heuristic constraint: include at least one object and one operation/constraint if possible."""
    has_object = any(c.type == "object" for c in concepts)
    has_op_or_con = any(c.type in ("operation", "constraint") for c in concepts)
    if has_object and has_op_or_con:
        return concepts
    # If missing, we don't have enough type info; return as-is.
    return concepts


class ExploratorPolicy(ABC):
    @abstractmethod
    def sample_spec(self) -> Spec:
        raise NotImplementedError


class GRIPStage0Explorator(ExploratorPolicy):
    """
    Cold-start-friendly sampling:
    - sample bundle mostly from a single seed example's concept set
    - optionally swap one concept with an explicit neighbor in the graph
    """

    def __init__(
        self,
        seed_concept_sets: List[List[Concept]],
        graph: ConceptGraph,
        *,
        bundle_size: int = 4,
        allow_swap_one_neighbor: bool = True,
        hop_mode: str = "explicit",
        target_domain: str = "",
        answer_type: str = "final_answer",
    ):
        self.seed_concept_sets = seed_concept_sets
        self.graph = graph
        self.bundle_size = bundle_size
        self.allow_swap_one_neighbor = allow_swap_one_neighbor
        self.hop_mode = hop_mode
        self.target_domain = target_domain
        self.answer_type = answer_type

    def sample_spec(self) -> Spec:
        base = random.choice(self.seed_concept_sets)
        if len(base) <= self.bundle_size:
            bundle = list(base)
        else:
            bundle = random.sample(base, self.bundle_size)

        if self.allow_swap_one_neighbor and bundle:
            idx = random.randrange(len(bundle))
            orig = bundle[idx]
            if self.hop_mode == "explicit":
                neigh_key = self.graph.sample_neighbor(orig.key)
            elif self.hop_mode == "implicit_2hop":
                neigh_key = self.graph.sample_2hop_neighbor(orig.key)
            else:
                neigh_key = self.graph.sample_2hop_neighbor(orig.key)
            if neigh_key and neigh_key in self.graph.concepts_by_key:
                bundle[idx] = self.graph.concepts_by_key[neigh_key]

        bundle = _ensure_type_mix(bundle)
        return Spec(required_concepts=bundle, hop_mode=self.hop_mode, target_domain=self.target_domain, answer_type=self.answer_type)

