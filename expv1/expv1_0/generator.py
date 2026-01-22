from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from .llm import LLMClient
from .text_parse import extract_tag_content, has_boxed_content, strip_boxed_content
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

        # Extract question from <question> tags - DO NOT fallback to full raw output
        q = extract_tag_content(raw, "question")
        
        # Determine if format is valid
        format_valid = q is not None and len(q.strip()) > 10  # Must have <question> tags with content
        
        if not format_valid:
            # Mark as invalid format - will be rejected in pipeline
            return CandidateSample(
                spec=spec, 
                problem="", 
                answer="", 
                raw_output=raw, 
                metadata={"format_invalid": True, "reason": "no_question_tags"}
            )
        
        # Strip any \boxed{} content to prevent answer leakage to solver
        answer_leaked = has_boxed_content(q)
        q_clean = strip_boxed_content(q)
        
        return CandidateSample(
            spec=spec, 
            problem=q_clean, 
            answer="", 
            raw_output=raw, 
            metadata={"format_invalid": False, "answer_leaked": answer_leaked}
        )

