"""
End-to-end scaffold for extracting and normalizing math concepts on the MATH dataset.

This is model-agnostic but assumes access to a strong open-weight model for
both extraction and normalization. Defaults target the MATH dataset from
Hugging Face (`hendrycks/competition_math`).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .ontology import ONTOLOGY
from .prompts import build_extraction_prompt, build_normalization_prompt

JsonDict = Dict[str, Any]


# --------------------------------------------------------------------------- #
# Generation helpers
# --------------------------------------------------------------------------- #
@dataclass
class GeneratorConfig:
    model_name_or_path: str
    device_map: str | None = "auto"
    torch_dtype: str | None = None  # "auto", "bfloat16", "float16"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9


class HFGenerator:
    """
    Thin wrapper over a causal LM to generate text for prompts.
    """

    def __init__(self, config: GeneratorConfig):
        dtype = None
        if config.torch_dtype == "float16":
            dtype = torch.float16
        elif config.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif config.torch_dtype == "auto":
            dtype = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            device_map=config.device_map,
            torch_dtype=dtype,
        )
        self.config = config

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Strip the prompt prefix if the model echoes it
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def extract_first_json(text: str) -> JsonDict:
    """
    Extract the first JSON object from a string. Raises ValueError on failure.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    snippet = match.group(0)
    return json.loads(snippet)


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #
def run_single_example(
    example: Mapping[str, Any],
    extractor: HFGenerator,
    normalizer: HFGenerator,
    ontology: List[str],
    max_concepts: int,
) -> JsonDict:
    """
    Run extraction then normalization on a single example.
    """
    qid = str(example.get("id") or example.get("problem_id") or example.get("idx") or "")
    question = str(example["problem"] if "problem" in example else example["question"])

    extraction_prompt = build_extraction_prompt(question, ontology=ontology, max_concepts=max_concepts)
    raw_text = extractor.generate(extraction_prompt)
    raw_json = extract_first_json(raw_text)

    # Ensure id/question are set
    raw_json.setdefault("id", qid)
    raw_json.setdefault("question", question)

    normalization_prompt = build_normalization_prompt(raw_json, ontology=ontology, max_concepts=max_concepts)
    norm_text = normalizer.generate(normalization_prompt)
    norm_json = extract_first_json(norm_text)

    # Ensure id/question are preserved
    norm_json.setdefault("id", qid)
    norm_json.setdefault("question", question)
    return norm_json


def process_dataset(
    split: str,
    extractor: HFGenerator,
    normalizer: HFGenerator,
    ontology: List[str],
    output_path: Path,
    limit: Optional[int],
    max_concepts: int,
) -> None:
    """
    Stream through the MATH dataset, extract concepts, normalize, and write JSONL.
    """
    ds = load_dataset("hendrycks/competition_math", split=split, streaming=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, example in enumerate(ds):
            if limit is not None and written >= limit:
                break
            try:
                norm = run_single_example(
                    example=example,
                    extractor=extractor,
                    normalizer=normalizer,
                    ontology=ontology,
                    max_concepts=max_concepts,
                )
                fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] failed on idx={idx}: {exc}")
    print(f"[done] wrote {written} rows to {output_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and normalize math concepts from MATH dataset.")
    parser.add_argument("--extractor", required=True, help="Model name or path for extraction.")
    parser.add_argument("--normalizer", required=True, help="Model name or path for normalization (can be same).")
    parser.add_argument("--split", default="train", help="Dataset split (train/test).")
    parser.add_argument("--output", default="math_concepts.jsonl", help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=100, help="Optional cap on number of examples.")
    parser.add_argument("--max-concepts", type=int, default=8, help="Max concepts per question.")
    parser.add_argument("--device-map", default="auto", help="Device map for HF models.")
    parser.add_argument("--dtype", default=None, choices=["auto", "float16", "bfloat16", None], help="Torch dtype.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens per generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    extractor_cfg = GeneratorConfig(
        model_name_or_path=args.extractor,
        device_map=args.device_map,
        torch_dtype=args.dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    normalizer_cfg = GeneratorConfig(
        model_name_or_path=args.normalizer,
        device_map=args.device_map,
        torch_dtype=args.dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    extractor = HFGenerator(extractor_cfg)
    normalizer = HFGenerator(normalizer_cfg)

    process_dataset(
        split=args.split,
        extractor=extractor,
        normalizer=normalizer,
        ontology=list(ONTOLOGY),
        output_path=output_path,
        limit=args.limit,
        max_concepts=args.max_concepts,
    )


if __name__ == "__main__":
    main()
