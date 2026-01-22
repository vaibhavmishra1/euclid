from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from .llm import LLMClient
from .text_parse import extract_tag_content
from .types import CandidateSample, Concept, Spec
from .utils import read_text, render_template


def _concepts_to_list(concepts: List[Concept]) -> List[str]:
    """Format concepts as a simple list of names for the prompt."""
    return [c.name for c in concepts]


@dataclass
class LLMQuestionGenerator:
    llm: LLMClient
    prompt_path: str
    max_tokens: int
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        self.prompt_template = read_text(self.prompt_path)

    def generate_one(self, spec: Spec, *, log_path: Optional[str] = None) -> CandidateSample:
        prompt = render_template(
            self.prompt_template,
            {
                "required_concepts": _concepts_to_list(spec.required_concepts),
            },
        )
        raw = self.llm.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=1)[0]

        # Optional: write prompt+output to a single file for easy inspection/debugging.
        if log_path:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                "\n".join(
                    [
                        "===== PROMPT =====",
                        prompt.rstrip(),
                        "",
                        "===== OUTPUT =====",
                        raw.rstrip(),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

        q = extract_tag_content(raw, "question") or raw.strip()
        # Only generate questions - answers will be verified by solvers
        return CandidateSample(spec=spec, problem=q, answer="", raw_output=raw, metadata={})

