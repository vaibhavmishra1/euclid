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
    def extract(self, problem: str, solution: str) -> List[Concept]:
        raise NotImplementedError


class LLMConceptExtractor(ConceptExtractor):
    def __init__(self, llm: LLMClient, prompt_path: str, *, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0):
        self.llm = llm
        self.prompt_template = read_text(prompt_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def extract(self, problem: str, solution: str) -> List[Concept]:
        prompt = render_template(self.prompt_template, {"problem": problem, "solution": solution})
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
            # Fallback: return generic concept if JSON parse fails.
            return [Concept(type="object", name="generic_object")]

