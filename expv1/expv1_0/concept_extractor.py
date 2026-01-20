from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import List

from .llm import LLMClient
from .types import Concept
from .utils import read_text, render_template


class ConceptExtractor(ABC):
    @abstractmethod
    def extract(self, problem: str) -> List[Concept]:
        raise NotImplementedError


class HeuristicConceptExtractor(ConceptExtractor):
    """
    A light, non-LLM extractor used for smoke tests.
    It is not intended to match GRIP fidelity.
    """

    def extract(self, problem: str) -> List[Concept]:
        p = problem.lower()
        concepts: List[Concept] = []

        # Objects
        if "triangle" in p:
            concepts.append(Concept(type="object", name="triangle"))
        if "polynomial" in p or "x^2" in p:
            concepts.append(Concept(type="object", name="polynomial"))
        if "coin" in p or "probability" in p:
            concepts.append(Concept(type="object", name="coin_flip"))

        # Operations / techniques
        if "sum" in p or "\\sum" in p:
            concepts.append(Concept(type="operation", name="summation"))
        if "divisible" in p or "mod" in p:
            concepts.append(Concept(type="operation", name="modular_arithmetic"))
        if "area" in p:
            concepts.append(Concept(type="operation", name="area_computation"))

        # Constraints
        if "integer" in p or "integers" in p:
            concepts.append(Concept(type="constraint", name="integer"))
        if "real number" in p or "real" in p:
            concepts.append(Concept(type="constraint", name="real"))

        # Fallback padding
        if not concepts:
            concepts = [
                Concept(type="object", name="generic_object"),
                Concept(type="operation", name="generic_operation"),
                Concept(type="constraint", name="generic_constraint"),
            ]

        # Ensure at least 3
        while len(concepts) < 3:
            concepts.append(Concept(type="operation", name=f"generic_op_{len(concepts)}"))

        # Dedup by key
        seen = set()
        uniq: List[Concept] = []
        for c in concepts:
            if c.key in seen:
                continue
            seen.add(c.key)
            uniq.append(c)
        return uniq[:8]


class LLMConceptExtractor(ConceptExtractor):
    def __init__(self, llm: LLMClient, prompt_path: str, *, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0):
        self.llm = llm
        self.prompt_template = read_text(prompt_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def extract(self, problem: str) -> List[Concept]:
        prompt = render_template(self.prompt_template, {"problem": problem})
        out = self.llm.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=1)[0]
        try:
            data = json.loads(out)
            concepts_raw = data.get("concepts", [])
            concepts: List[Concept] = []
            for c in concepts_raw:
                t = str(c.get("type", "")).strip()
                n = str(c.get("name", "")).strip()
                if t and n:
                    concepts.append(Concept(type=t, name=n))
            return concepts[:8] if concepts else [Concept(type="object", name="generic_object")]
        except Exception:
            # Fallback: very rough keyword extraction if JSON parse fails.
            return HeuristicConceptExtractor().extract(problem)

