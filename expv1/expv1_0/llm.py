from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float, n: int = 1) -> List[str]:
        """Return a list of n generated completions (strings)."""
        raise NotImplementedError


@dataclass
class MockLLMClient(LLMClient):
    """
    A deterministic fallback client so the pipeline can run without real models.
    This is for plumbing validation only.
    """

    mode: str = "generic"  # concept_extract | generate_problem | teacher_verify | solver

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float, n: int = 1) -> List[str]:
        outs: List[str] = []
        for _ in range(n):
            outs.append(self._one(prompt))
        return outs

    def _one(self, prompt: str) -> str:
        p = prompt.lower()

        if "concept extraction engine" in p or '"concepts"' in p:
            return json.dumps(
                {
                    "concepts": [
                        {"type": "object", "name": "toy_object"},
                        {"type": "operation", "name": "toy_operation"},
                        {"type": "constraint", "name": "toy_constraint"},
                    ]
                }
            )

        if "expert competition-math problem setter" in p:
            # Produce a small arithmetic question with a known answer.
            a = random.randint(2, 20)
            b = random.randint(2, 20)
            return f"<question>\nCompute {a}+{b}.\n</question>\n\\\\boxed{{{a+b}}}\n"

        if "strict mathematical verifier" in p and '"well_posed"' in p:
            return json.dumps(
                {
                    "well_posed": True,
                    "correct": True,
                    "corrected_answer": None,
                    "difficulty_1_to_5": 1,
                    "ambiguity_flags": [],
                    "notes": "mock",
                }
            )

        if "careful mathematical problem solver" in p:
            # Parse simple "Compute a+b" format; be correct ~50% to simulate ZPD.
            m = re.search(r"compute\s+(\d+)\s*\+\s*(\d+)", p)
            if m:
                a = int(m.group(1))
                b = int(m.group(2))
                correct = (a + b)
                if random.random() < 0.5:
                    pred = correct
                else:
                    pred = correct + 1
                return f"Solution: {a}+{b}={pred}. \\\\boxed{{{pred}}}"
            return "Solution: \\boxed{0}"

        return ""


class HFLLMClient(LLMClient):
    """
    Simple HuggingFace Transformers client (single GPU/CPU/MPS).
    Suitable for small-scale runs; for throughput use vLLM.
    """

    def __init__(self, model_name_or_path: str, device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if self.device in ("cuda", "mps"):
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float, n: int = 1) -> List[str]:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else None,
                top_p=top_p if do_sample else None,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        # If n>1, outputs includes multiple sequences.
        out_texts: List[str] = []
        for seq in outputs:
            gen = seq[inputs["input_ids"].shape[1] :]
            out_texts.append(self.tokenizer.decode(gen, skip_special_tokens=True))
        return out_texts


class VLLMLLMClient(LLMClient):
    """vLLM-backed client (lazy import)."""

    def __init__(self, model_name_or_path: str, gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1, seed: int = 42):
        from vllm import LLM

        self.model_name_or_path = model_name_or_path
        self.model = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
        )

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float, n: int = 1) -> List[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
        )
        outputs = self.model.generate([prompt], params)
        return [o.text for o in outputs[0].outputs]


def build_llm_client(
    backend: str,
    model_name: str,
    seed: int = 42,
    *,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_tensor_parallel_size: int = 1,
) -> LLMClient:
    backend = backend.lower()
    if backend == "mock" or not model_name:
        return MockLLMClient()
    if backend == "hf":
        return HFLLMClient(model_name)
    if backend == "vllm":
        return VLLMLLMClient(
            model_name,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            tensor_parallel_size=vllm_tensor_parallel_size,
            seed=seed,
        )
    raise ValueError(f"Unknown LLM backend: {backend}")

