"""
Prompt builders for extraction and normalization.

Extraction outputs raw concepts, theorems, formulae, and axioms/definitions.
Normalization maps raw items back to the controlled ontology, keeping NEW: labels
when something is out of ontology.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

from .ontology import ONTOLOGY, ontology_as_text


def build_extraction_prompt(
    question: str,
    ontology: Sequence[str] | None = None,
    max_concepts: int = 8,
) -> str:
    """
    Build a single-string prompt to feed to a text-generation model.
    """
    ontology = ontology or ONTOLOGY
    ontology_text = ontology_as_text(prefix="- ")

    return f"""You are an expert math tagger. Given a math question, extract the minimal set of key concepts and supporting items needed to solve it.

Rules:
- Output JSON only, no prose.
- concepts: choose from the ontology list below; if missing, use NEW:<short label>. Max {max_concepts}, deduped, most central first.
- named_theorems: canonical names only (e.g., "Power of a Point", "Law of Cosines"). If unsure, UNKNOWN_THEOREM:<short description>. Do not invent names.
- formulae: minimal symbolic forms (LaTeX-friendly). Examples: a^2+b^2=c^2, Vieta: x1+x2=-b/a, x1*x2=c/a, AreaSector = (θ/2π)πr^2. No long prose.
- axioms_or_definitions: e.g., "definition of derivative", "triangle inequality". Prefer canonical names; otherwise NEW:<short label>.
- Preserve the question text verbatim in question.

Ontology list (choose concepts from here unless you must use NEW:):
{ontology_text}

Return JSON with fields:
{{
  "id": "<id-string>",
  "question": "<question text>",
  "concepts": ["<ontology or NEW:...>", ...],
  "named_theorems": ["..."],
  "formulae": ["..."],
  "axioms_or_definitions": ["..."]
}}

Question:
{question}
"""


def build_normalization_prompt(
    raw: Mapping[str, object],
    ontology: Sequence[str] | None = None,
    max_concepts: int = 8,
) -> str:
    """
    Build a normalization prompt that maps raw extraction output to the ontology.
    """
    ontology = ontology or ONTOLOGY
    ontology_text = ontology_as_text(prefix="- ")
    raw_json = json.dumps(raw, ensure_ascii=False, indent=2)

    return f"""You are a math concept normalizer. Map raw extracted items to the controlled ontology. If no good match, keep as NEW:<canonical label> (concise). Deduplicate and cap concepts at {max_concepts}.

Rules:
- concepts: map each raw entry to the closest ontology label (list below). If none fits, keep as NEW:<canonical label>; make labels short and specific. Deduplicate, preserve importance order where possible, cap at {max_concepts}.
- named_theorems: map to canonical theorem names when obvious; otherwise keep original or set UNKNOWN_THEOREM:<short description>. Deduplicate.
- formulae: keep symbolic; normalize trivial reorderings; deduplicate.
- axioms_or_definitions: map to canonical names where possible; else keep as NEW:<canonical label>. Deduplicate.

Ontology list:
{ontology_text}

Input JSON:
{raw_json}

Return JSON (normalized):
{{
  "id": "...",
  "question": "...",
  "concepts": ["..."],
  "named_theorems": ["..."],
  "formulae": ["..."],
  "axioms_or_definitions": ["..."]
}}
"""
