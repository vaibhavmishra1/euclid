from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .llm import build_llm_client
from .pipeline_types import AcceptedSample, CandidateSample, Spec
from .utils import ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl
from .zpd import ZPDScorer


def filter_questions_with_zpd(config_path: str, questions_path: str | None = None, p_min: float | None = None, p_max: float | None = None) -> str:
    """
    Pipeline B2: ZPD Filtering + COT Generation
    - Load generated questions from generated_questions.jsonl
    - Run solver ZPD scoring (multiple rollouts)
    - Extract full COT + answer from best rollout
    - Apply ZPD threshold filter
    - Save accepted questions with full solutions
    """

    cfg = load_yaml(config_path)

    # Determine input questions path
    if questions_path is None:
        # Try to infer from config or use default
        base_out_dir = cfg["io"]["output_dir"]
        generator_model = str(cfg["generation"]["generator_model"])
        model_name_sanitized = generator_model.split("/")[-1].replace(" ", "_")
        questions_dir = f"{base_out_dir}_{model_name_sanitized}_questions"
        questions_path = f"{questions_dir}/generated_questions.jsonl"
    
    if not Path(questions_path).exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}. Run Pipeline B1 first.")

    # Determine output directory
    base_out_dir = cfg["io"]["output_dir"]
    solver_model = str(cfg["zpd"]["solver_model"])
    model_name_sanitized = solver_model.split("/")[-1].replace(" ", "_")
    
    # Include ZPD thresholds in output dir name if custom
    zpd_suffix = ""
    if p_min is not None or p_max is not None:
        p_min_val = p_min if p_min is not None else cfg["zpd"]["p_min"]
        p_max_val = p_max if p_max is not None else cfg["zpd"]["p_max"]
        zpd_suffix = f"_zpd{p_min_val:.1f}-{p_max_val:.1f}"
    
    out_dir = ensure_dir(f"{base_out_dir}_{model_name_sanitized}_accepted{zpd_suffix}")
    
    print(f"[Pipeline B2] Loading questions from: {questions_path}")
    print(f"[Pipeline B2] Output directory: {out_dir}")

    # Load generated questions
    questions = read_jsonl(questions_path)
    print(f"[Pipeline B2] Loaded {len(questions)} questions to filter")

    # ZPD threshold (use provided or config defaults)
    p_min_val = float(p_min) if p_min is not None else float(cfg["zpd"]["p_min"])
    p_max_val = float(p_max) if p_max is not None else float(cfg["zpd"]["p_max"])
    print(f"[Pipeline B2] ZPD filter: {p_min_val} <= p_succ <= {p_max_val}")

    # Build solver client
    zpd_vllm = cfg["zpd"].get("vllm", {}) or {}
    solver_client = build_llm_client(
        cfg["zpd"]["solver_backend"],
        cfg["zpd"]["solver_model"],
        seed=int(cfg["experiment"]["seed"]),
        vllm_gpu_memory_utilization=float(zpd_vllm.get("gpu_memory_utilization", 0.9)),
        vllm_tensor_parallel_size=int(zpd_vllm.get("tensor_parallel_size", 1)),
    )

    zpd = ZPDScorer(
        llm=solver_client,
        prompt_path=cfg["zpd"]["prompt_path"],
        rollouts=int(cfg["zpd"]["rollouts"]),
        max_tokens=int(cfg["zpd"]["max_tokens"]),
        temperature=float(cfg["zpd"]["temperature"]),
        top_p=float(cfg["zpd"]["top_p"]),
    )

    # Process each question
    accepted: List[AcceptedSample] = []
    rejected_logs: List[Dict[str, Any]] = []

    for idx, q in enumerate(questions):
        problem = q.get("problem", "")
        if not problem:
            rejected_logs.append({
                "idx": q.get("idx", idx),
                "accepted": False,
                "reject_reason": "empty_problem",
                "question": q,
            })
            continue

        # Run ZPD scoring
        zpd_result = zpd.score(problem, "")
        modal_answer = zpd_result.details.get("modal_answer", "")
        full_outputs = zpd_result.details.get("full_outputs", [])

        if not modal_answer:
            rejected_logs.append({
                "idx": q.get("idx", idx),
                "accepted": False,
                "reject_reason": "solver_no_answer",
                "question": q,
                "zpd": asdict(zpd_result),
            })
            continue

        # Check ZPD threshold
        if not (p_min_val <= zpd_result.p_succ <= p_max_val):
            rejected_logs.append({
                "idx": q.get("idx", idx),
                "accepted": False,
                "reject_reason": "zpd_out_of_band",
                "question": q,
                "zpd": asdict(zpd_result),
            })
            continue

        # Extract best solution (the one matching modal_answer)
        best_solution = ""
        if full_outputs:
            # Find the output that produced the modal_answer
            for out in full_outputs:
                from .text_parse import extract_boxed_answer
                pred = extract_boxed_answer(out) or ""
                if pred == modal_answer:
                    best_solution = out
                    break
            
            # If no exact match, use the first non-empty output
            if not best_solution:
                for out in full_outputs:
                    if out.strip():
                        best_solution = out
                        break

        # Reconstruct spec from saved question
        spec_dict = q.get("spec", {})
        required_concepts = []
        for c_dict in spec_dict.get("required_concepts", []):
            from .pipeline_types import Concept
            required_concepts.append(Concept(
                type=str(c_dict.get("type", "")),
                name=str(c_dict.get("name", ""))
            ))
        
        spec = Spec(
            required_concepts=required_concepts,
            hop_mode=spec_dict.get("hop_mode", "explicit"),
            target_domain=spec_dict.get("target_domain", ""),
            answer_type=spec_dict.get("answer_type", "final_answer"),
        )

        # Create candidate sample
        cand = CandidateSample(
            spec=spec,
            problem=problem,
            answer=modal_answer,
            raw_output=q.get("raw_output", ""),
            metadata=q.get("metadata", {}),
        )

        # Create accepted sample with full solution
        accepted.append(
            AcceptedSample(
                candidate=cand,
                verification={
                    "modal_answer": modal_answer,
                    "p_succ": zpd_result.p_succ,
                    "solution": best_solution,  # Full COT solution
                },
                zpd=zpd_result,
            )
        )

        if (idx + 1) % 50 == 0:
            print(f"[Pipeline B2] Processed {idx + 1}/{len(questions)} questions, accepted {len(accepted)}")

    # Save outputs
    write_jsonl(f"{out_dir}/accepted.jsonl", accepted)
    write_jsonl(f"{out_dir}/rejected.jsonl", rejected_logs)

    write_json(
        f"{out_dir}/metrics.json",
        {
            "num_questions": len(questions),
            "num_accepted": len(accepted),
            "num_rejected": len(rejected_logs),
            "accept_rate": (len(accepted) / len(questions)) if questions else 0.0,
            "zpd_threshold": {"p_min": p_min_val, "p_max": p_max_val},
            "questions_source": questions_path,
        },
    )

    print(f"[Pipeline B2] Accepted {len(accepted)} questions out of {len(questions)} (accept rate: {len(accepted)/len(questions)*100:.1f}%)")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="expv1_0: filter questions with ZPD and generate COT (Pipeline B2)")
    parser.add_argument("--config", type=str, required=True, help="Path to expv1_0 config.yaml")
    parser.add_argument("--questions", type=str, default=None, help="Path to generated_questions.jsonl (auto-detected if not provided)")
    parser.add_argument("--p-min", type=float, default=None, help="Minimum p_succ threshold (overrides config)")
    parser.add_argument("--p-max", type=float, default=None, help="Maximum p_succ threshold (overrides config)")
    args = parser.parse_args()

    out_dir = filter_questions_with_zpd(args.config, args.questions, args.p_min, args.p_max)
    print(f"[expv1_0] ZPD filtering complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
