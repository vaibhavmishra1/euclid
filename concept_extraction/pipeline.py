"""
End-to-end scaffold for extracting and normalizing math concepts on the MATH dataset.

This is model-agnostic but assumes access to a strong open-weight model for
both extraction and normalization. Defaults target the MATH dataset from
Hugging Face (`hendrycks/math`).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol

from datasets import load_dataset
from tqdm import tqdm

from .prompts import build_extraction_prompt, build_normalization_prompt

JsonDict = Dict[str, Any]


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


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
    # OpenAI
    base_url: str | None = None
    max_retries: int = 6


class HFGenerator:
    """
    Thin wrapper over a causal LM to generate text for prompts.
    """

    def __init__(self, config: GeneratorConfig):
        # Lazy imports so OpenAI-only runs don't require torch/transformers.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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


class OpenAIGenerator:
    """
    Generator backed by OpenAI Responses API.

    Requires environment variable OPENAI_API_KEY.
    """

    def __init__(self, config: GeneratorConfig):
        try:
            from openai import OpenAI
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "OpenAI backend selected but package 'openai' is not installed. "
                "Install with: pip install openai"
            ) from exc

        client_kwargs: Dict[str, Any] = {}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self.client = OpenAI(**client_kwargs)
        self.config = config

    def generate(self, prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.client.responses.create(
                    model=self.config.model_name_or_path,
                    input=prompt,
                    text={"format": {"type": "json_object"}},
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_output_tokens=self.config.max_new_tokens,
                )
                text = getattr(resp, "output_text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()

                out: List[str] = []
                for item in getattr(resp, "output", []) or []:
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            out.append(getattr(c, "text", ""))
                return "".join(out).strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt >= self.config.max_retries:
                    break
                sleep_s = min(60.0, (2**attempt) * 0.5) + random.random() * 0.25
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI request failed after retries: {last_err}") from last_err


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def extract_first_json(text: str) -> JsonDict:
    """
    Extract the first JSON object from a string. Raises ValueError on failure.
    """
    s = text.strip()
    candidates: List[str] = []
    if s.startswith("{") and s.endswith("}"):
        candidates.append(s)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    if not candidates:
        raise ValueError("No JSON object found in model output.")

    last_err: Exception | None = None
    for snippet in candidates:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:
            # Common failure mode for math text: backslashes like \left, \frac, etc.
            # JSON requires escaping backslashes, so we patch invalid escapes and retry once.
            last_err = exc
            repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", snippet)
            try:
                return json.loads(repaired)
            except Exception as exc2:  # noqa: BLE001
                last_err = exc2
                continue
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise ValueError(f"Failed to parse JSON from model output: {last_err}") from last_err


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #
def run_single_example(
    example: Mapping[str, Any],
    extractor: TextGenerator,
    normalizer: TextGenerator,
    max_knowledge_points: int,
) -> JsonDict:
    """
    Run extraction then normalization on a single example.
    """
    qid = str(example.get("id") or example.get("problem_id") or example.get("idx") or "")
    question = str(example["problem"] if "problem" in example else example["question"])
    solution = str(example.get("solution") or example.get("answer") or "")
    level = str(example.get("level") or "")

    extraction_prompt = build_extraction_prompt(qid, question, solution=solution, max_concepts=max_knowledge_points)
    raw_text = extractor.generate(extraction_prompt)
    raw_json = extract_first_json(raw_text)

    # Always set id/question from the original example (prompts tell model to leave these empty)
    raw_json["id"] = qid
    raw_json["question"] = question

    normalization_prompt = build_normalization_prompt(raw_json, max_concepts=max_knowledge_points)
    norm_text = normalizer.generate(normalization_prompt)
    norm_json = extract_first_json(norm_text)

    # Always set id/question/level from the original example
    norm_json["id"] = qid
    norm_json["question"] = question
    norm_json["level"] = level
    
    return norm_json


def process_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    extractor: TextGenerator,
    normalizer: TextGenerator,
    output_path: Path,
    limit: Optional[int],
    max_knowledge_points: int,
) -> None:
    """
    Stream through the MATH dataset, extract knowledge points, normalize, and write JSONL.
    """
    written = 0

    # The Hendrycks MATH dataset is commonly mirrored as EleutherAI/hendrycks_math,
    # which requires selecting a config (subject). If none is provided, we iterate
    # all available configs.
    configs_to_process: List[str | None]
    if dataset_name == "EleutherAI/hendrycks_math" and not dataset_config:
        configs_to_process = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
    else:
        configs_to_process = [dataset_config]

    with output_path.open("w", encoding="utf-8") as fout:
        for cfg in configs_to_process:
            if limit is not None and written >= limit:
                break
            try:
                if cfg:
                    ds = load_dataset(dataset_name, cfg, split=split, streaming=True)
                else:
                    ds = load_dataset(dataset_name, split=split, streaming=True)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to load dataset={dataset_name!r} config={cfg!r} split={split!r}. "
                    "For the MATH dataset, use --dataset EleutherAI/hendrycks_math "
                    "and optionally --dataset-config algebra|geometry|... . "
                    f"Original error: {exc}"
                ) from exc

            # Calculate remaining items for this config
            remaining = (limit - written) if limit is not None else None
            desc = f"Processing {cfg or dataset_name}"
            
            pbar = tqdm(enumerate(ds), total=remaining, desc=desc, unit="ex")
            for idx, example in pbar:
                if limit is not None and written >= limit:
                    break
                try:
                    norm = run_single_example(
                        example=example,
                        extractor=extractor,
                        normalizer=normalizer,
                        max_knowledge_points=max_knowledge_points,
                    )
                    # Add dataset provenance (handy when iterating multiple configs)
                    norm.setdefault("_dataset", dataset_name)
                    if cfg:
                        norm.setdefault("_dataset_config", cfg)
                    fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    written += 1
                    pbar.set_postfix({"written": written})
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"[warn] failed on cfg={cfg!r} idx={idx}: {exc}")

    print(f"[done] wrote {written} rows to {output_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and normalize math concepts from MATH dataset.")
    parser.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "openai"],
        help="Generation backend: hf (local transformers) or openai (Responses API).",
    )
    parser.add_argument(
        "--dataset",
        default="EleutherAI/hendrycks_math",
        help="Hugging Face dataset name (default: EleutherAI/hendrycks_math).",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config name (e.g., algebra). If omitted for EleutherAI/hendrycks_math, processes all configs.",
    )
    parser.add_argument("--extractor", required=True, help="Model name or path for extraction.")
    parser.add_argument("--normalizer", required=True, help="Model name or path for normalization (can be same).")
    parser.add_argument("--split", default="train", help="Dataset split (train/test).")
    parser.add_argument("--output", default="math_knowledge_points.jsonl", help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=100, help="Max number of examples to process (0 = no limit).")
    parser.add_argument("--max-knowledge-points", type=int, default=5, help="Max knowledge points per question (1-5).")
    parser.add_argument("--device-map", default="auto", help="Device map for HF models.")
    parser.add_argument("--dtype", default=None, choices=["auto", "float16", "bfloat16"], help="Torch dtype (HF only).")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens per generation.")
    parser.add_argument("--openai-base-url", default=None, help="Optional OpenAI-compatible base URL (advanced).")
    parser.add_argument("--max-retries", type=int, default=6, help="Max retries for API calls (OpenAI backend).")
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
        base_url=args.openai_base_url,
        max_retries=args.max_retries,
    )
    normalizer_cfg = GeneratorConfig(
        model_name_or_path=args.normalizer,
        device_map=args.device_map,
        torch_dtype=args.dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        base_url=args.openai_base_url,
        max_retries=args.max_retries,
    )

    if args.backend == "openai":
        extractor: TextGenerator = OpenAIGenerator(extractor_cfg)
        normalizer: TextGenerator = OpenAIGenerator(normalizer_cfg)
    else:
        extractor = HFGenerator(extractor_cfg)
        normalizer = HFGenerator(normalizer_cfg)

    process_dataset(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        extractor=extractor,
        normalizer=normalizer,
        output_path=output_path,
        limit=args.limit or None,  # 0 means no limit
        max_knowledge_points=args.max_knowledge_points,
    )


if __name__ == "__main__":
    main()
