from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

from .concept_graph import ConceptGraph
from .pipeline_types import Concept, Spec


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
    GRIP-style sampling with mixed explicit and implicit edges:
    - Sample bundle mostly from a single seed example's concept set
    - Optionally swap concepts with neighbors (explicit, 2-hop, or 3-hop)
    - Supports mixing explicit and implicit sampling as in GRIP paper
    """

    def __init__(
        self,
        seed_concept_sets: List[List[Concept]],
        graph: ConceptGraph,
        *,
        bundle_size: int = 4,
        allow_swap_one_neighbor: bool = True,
        hop_mode: str = "mixed",  # explicit | implicit_2hop | implicit_3hop | mixed
        explicit_ratio: float = 0.5,  # Probability of using explicit edge (only for mixed mode)
        implicit_2hop_ratio: float = 0.3,  # Probability of using 2-hop (only for mixed mode)
        implicit_3hop_ratio: float = 0.2,  # Probability of using 3-hop (only for mixed mode)
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
        
        # Normalize ratios for mixed mode
        if hop_mode == "mixed":
            total = explicit_ratio + implicit_2hop_ratio + implicit_3hop_ratio
            if total > 0:
                self.explicit_ratio = explicit_ratio / total
                self.implicit_2hop_ratio = implicit_2hop_ratio / total
                self.implicit_3hop_ratio = implicit_3hop_ratio / total
            else:
                # Default: equal mix
                self.explicit_ratio = 1.0 / 3.0
                self.implicit_2hop_ratio = 1.0 / 3.0
                self.implicit_3hop_ratio = 1.0 / 3.0
        else:
            self.explicit_ratio = 0.0
            self.implicit_2hop_ratio = 0.0
            self.implicit_3hop_ratio = 0.0

    def _sample_neighbor_by_mode(self, key: str, hop_mode: str) -> tuple[str | None, str]:
        """
        Sample a neighbor based on hop mode.
        Returns: (neighbor_key, actual_hop_mode_used)
        """
        if hop_mode == "explicit":
            return self.graph.sample_neighbor(key), "explicit"
        elif hop_mode == "implicit_2hop":
            return self.graph.sample_2hop_neighbor(key), "implicit_2hop"
        elif hop_mode == "implicit_3hop":
            return self.graph.sample_3hop_neighbor(key), "implicit_3hop"
        elif hop_mode == "mixed":
            # Sample based on ratios
            r = random.random()
            if r < self.explicit_ratio:
                return self.graph.sample_neighbor(key), "explicit"
            elif r < self.explicit_ratio + self.implicit_2hop_ratio:
                return self.graph.sample_2hop_neighbor(key), "implicit_2hop"
            else:
                return self.graph.sample_3hop_neighbor(key), "implicit_3hop"
        else:
            # Fallback to explicit
            return self.graph.sample_neighbor(key), "explicit"

    def sample_spec(self) -> Spec:
        base = random.choice(self.seed_concept_sets)
        if len(base) <= self.bundle_size:
            bundle = list(base)
        else:
            bundle = random.sample(base, self.bundle_size)

        # Determine actual hop mode for this sample (for spec tracking)
        actual_hop_mode = self.hop_mode

        if self.allow_swap_one_neighbor and bundle:
            idx = random.randrange(len(bundle))
            orig = bundle[idx]
            neigh_key, used_hop_mode = self._sample_neighbor_by_mode(orig.key, self.hop_mode)
            
            if neigh_key and neigh_key in self.graph.concepts_by_key:
                bundle[idx] = self.graph.concepts_by_key[neigh_key]
                # Track which hop mode was actually used
                actual_hop_mode = used_hop_mode

        bundle = _ensure_type_mix(bundle)
        return Spec(
            required_concepts=bundle, 
            hop_mode=actual_hop_mode, 
            target_domain=self.target_domain, 
            answer_type=self.answer_type
        )

