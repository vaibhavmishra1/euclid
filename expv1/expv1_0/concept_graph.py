from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from .pipeline_types import Concept


@dataclass
class ConceptGraph:
    """
    Minimal undirected weighted co-occurrence graph over concepts.
    Node key is Concept.key (type:name).
    """

    # adjacency[u][v] = weight
    adjacency: Dict[str, Dict[str, int]]
    # id->Concept for pretty printing (best-effort)
    concepts_by_key: Dict[str, Concept]

    @classmethod
    def build_from_concept_sets(cls, concept_sets: Iterable[List[Concept]], min_cooccurrence: int = 1) -> "ConceptGraph":
        adjacency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        concepts_by_key: Dict[str, Concept] = {}

        for concepts in concept_sets:
            keys = [c.key for c in concepts]
            for c in concepts:
                concepts_by_key[c.key] = c
            # add edges for all pairs
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    u, v = keys[i], keys[j]
                    adjacency[u][v] += 1
                    adjacency[v][u] += 1

        # filter by min_cooccurrence
        if min_cooccurrence > 1:
            for u in list(adjacency.keys()):
                for v in list(adjacency[u].keys()):
                    if adjacency[u][v] < min_cooccurrence:
                        del adjacency[u][v]
                if not adjacency[u]:
                    del adjacency[u]

        return cls(adjacency=dict(adjacency), concepts_by_key=concepts_by_key)

    def neighbors(self, key: str) -> List[str]:
        return list(self.adjacency.get(key, {}).keys())

    def sample_neighbor(self, key: str) -> str | None:
        neigh = self.neighbors(key)
        if not neigh:
            return None
        return random.choice(neigh)

    def sample_2hop_neighbor(self, key: str) -> str | None:
        """Sample a 2-hop neighbor (neighbor of a neighbor)."""
        n1 = self.sample_neighbor(key)
        if not n1:
            return None
        n2 = self.sample_neighbor(n1)
        return n2

    def sample_3hop_neighbor(self, key: str) -> str | None:
        """Sample a 3-hop neighbor (neighbor of neighbor of neighbor)."""
        n1 = self.sample_neighbor(key)
        if not n1:
            return None
        n2 = self.sample_neighbor(n1)
        if not n2:
            return None
        n3 = self.sample_neighbor(n2)
        return n3

    def induced_density(self, keys: List[str]) -> float:
        """Edge density of the induced subgraph over keys (0..1)."""
        if len(keys) < 2:
            return 0.0
        possible = len(keys) * (len(keys) - 1) / 2
        present = 0
        s = set(keys)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keys[j] in self.adjacency.get(keys[i], {}):
                    present += 1
        return float(present) / float(possible)

