from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from .concept_extractor import HeuristicConceptExtractor, LLMConceptExtractor
from .concept_graph import ConceptGraph
from .dedup import Deduper
from .explorator import GRIPStage0Explorator
from .generator import LLMQuestionGenerator
from .llm import build_llm_client
from .seeds import load_seed_examples_from_jsonl
from .teacher import LLMTeacherVerifier
from .types import AcceptedSample, CandidateSample, Concept, SeedExample, Spec
from .utils import ensure_dir, load_yaml, write_json, write_jsonl
from .zpd import ZPDScorer
from tqdm import tqdm

def build_dataset(config_path: str) -> str:
    cfg = load_yaml(config_path)
    random.seed(int(cfg["experiment"]["seed"]))

    out_dir = ensure_dir(cfg["io"]["output_dir"])

    # ------------------------------------------------------------------ #
    # 1) Load seeds
    # ------------------------------------------------------------------ #
    seeds: List[SeedExample] = load_seed_examples_from_jsonl(
        cfg["seed_data"]["toy_jsonl"],
        max_seeds=int(cfg["seed_data"]["max_seeds"]),
    )

    # ------------------------------------------------------------------ #
    # 2) Concept extraction on seeds → concept sets
    # ------------------------------------------------------------------ #
    concept_backend = str(cfg["concept_extraction"]["backend"]).lower()
    extractor = HeuristicConceptExtractor()

    seed_concept_sets: List[List[Concept]] = []
    for s in seeds:
        seed_concept_sets.append(extractor.extract(s.problem))

    # ------------------------------------------------------------------ #
    # 3) Build concept graph (explicit co-occurrence)
    # ------------------------------------------------------------------ #
    graph = ConceptGraph.build_from_concept_sets(seed_concept_sets, min_cooccurrence=int(cfg["concept_graph"]["min_cooccurrence"]))

    # ------------------------------------------------------------------ #
    # 4) Explorator policy (cold-start stage)
    # ------------------------------------------------------------------ #
    explorator = GRIPStage0Explorator(
        seed_concept_sets=seed_concept_sets,
        graph=graph,
        bundle_size=int(cfg["explorator"]["bundle_size"]),
        allow_swap_one_neighbor=bool(cfg["explorator"]["allow_swap_one_neighbor"]),
        hop_mode=str(cfg["explorator"]["hop_mode"]),
        target_domain="",
        answer_type="final_answer",
    )

    # ------------------------------------------------------------------ #
    # 5) Generator + teacher
    # ------------------------------------------------------------------ #
    gen_vllm = cfg["generation"].get("vllm", {}) or {}
    gen_client = build_llm_client(
        cfg["generation"]["generator_backend"],
        cfg["generation"]["generator_model"],
        seed=int(cfg["experiment"]["seed"]),
        vllm_gpu_memory_utilization=float(gen_vllm.get("gpu_memory_utilization", 0.9)),
        vllm_tensor_parallel_size=int(gen_vllm.get("tensor_parallel_size", 1)),
    )
    generator = LLMQuestionGenerator(
        llm=gen_client,
        prompt_path=cfg["generation"]["prompt_path"],
        max_tokens=int(cfg["generation"]["max_tokens"]),
        temperature=float(cfg["generation"]["temperature"]),
        top_p=float(cfg["generation"]["top_p"]),
    )

    teacher_vllm = cfg["teacher"].get("vllm", {}) or {}
    teacher_client = build_llm_client(
        cfg["teacher"]["teacher_backend"],
        cfg["teacher"]["teacher_model"],
        seed=int(cfg["experiment"]["seed"]),
        vllm_gpu_memory_utilization=float(teacher_vllm.get("gpu_memory_utilization", 0.9)),
        vllm_tensor_parallel_size=int(teacher_vllm.get("tensor_parallel_size", 1)),
    )
    teacher = LLMTeacherVerifier(
        llm=teacher_client,
        prompt_path=cfg["teacher"]["prompt_path"],
        max_tokens=int(cfg["teacher"]["max_tokens"]),
        temperature=float(cfg["teacher"]["temperature"]),
        top_p=1.0,
    )

    # If concept extraction uses an LLM, reuse the teacher client to avoid double-loading the 7B model.
    if concept_backend != "heuristic":
        extractor = LLMConceptExtractor(
            teacher_client,
            cfg["concept_extraction"]["prompt_path"],
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
        )
        seed_concept_sets = [extractor.extract(s.problem) for s in seeds]
        # rebuild graph based on extracted concept sets
        graph = ConceptGraph.build_from_concept_sets(seed_concept_sets, min_cooccurrence=int(cfg["concept_graph"]["min_cooccurrence"]))
        explorator.seed_concept_sets = seed_concept_sets
        explorator.graph = graph

    # ------------------------------------------------------------------ #
    # 6) Dedup + ZPD
    # ------------------------------------------------------------------ #
    deduper = Deduper(
        lexical=bool(cfg["dedup"]["lexical"]),
        embedding=bool(cfg["dedup"]["embedding"]),
        embedding_model=str(cfg["dedup"]["embedding_model"]),
        cosine_threshold=float(cfg["dedup"]["cosine_threshold"]),
        max_nn=int(cfg["dedup"]["max_nn"]),
    )

    zpd_enabled = bool(cfg["zpd"]["enabled"])
    zpd: ZPDScorer | None = None
    if zpd_enabled:
        # If solver==generator (same backend + same model), reuse to avoid double-loading.
        if (
            str(cfg["zpd"]["solver_backend"]).lower() == str(cfg["generation"]["generator_backend"]).lower()
            and str(cfg["zpd"]["solver_model"]) == str(cfg["generation"]["generator_model"])
        ):
            solver_client = gen_client
        else:
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

    p_min = float(cfg["zpd"]["p_min"])
    p_max = float(cfg["zpd"]["p_max"])

    # ------------------------------------------------------------------ #
    # 7) Main generation loop
    # ------------------------------------------------------------------ #
    num_candidates = int(cfg["generation"]["num_candidates"])
    candidate_logs: List[Dict[str, Any]] = []
    accepted: List[AcceptedSample] = []

    
    for i in tqdm(range(num_candidates), desc="Generating candidates"):
        spec = explorator.sample_spec()
        cand = generator.generate_one(spec)

        # Teacher verify
        tv = teacher.verify(cand.problem, cand.answer)

        # If answer is wrong but teacher provides corrected answer, accept corrected answer.
        if tv.well_posed and (not tv.correct) and tv.corrected_answer:
            cand.answer = str(tv.corrected_answer)
            tv.correct = True

        # Gate 1: validity
        if not (tv.well_posed and tv.correct):
            candidate_logs.append(
                {
                    "idx": i,
                    "accepted": False,
                    "reject_reason": "teacher_reject",
                    "candidate": asdict(cand),
                    "teacher": asdict(tv),
                }
            )
            continue

        # Gate 2: dedup
        dup_reason = deduper.check_duplicate(cand.problem)
        if dup_reason:
            candidate_logs.append(
                {
                    "idx": i,
                    "accepted": False,
                    "reject_reason": dup_reason,
                    "candidate": asdict(cand),
                    "teacher": asdict(tv),
                }
            )
            continue

        # Gate 3: ZPD
        zpd_result = None
        if zpd_enabled and zpd is not None:
            zpd_result = zpd.score(cand.problem, cand.answer)
            if not (p_min <= zpd_result.p_succ <= p_max):
                candidate_logs.append(
                    {
                        "idx": i,
                        "accepted": False,
                        "reject_reason": "zpd_out_of_band",
                        "candidate": asdict(cand),
                        "teacher": asdict(tv),
                        "zpd": asdict(zpd_result),
                    }
                )
                continue

        # Accept
        deduper.add(cand.problem)
        acc = AcceptedSample(candidate=cand, teacher=tv, zpd=zpd_result)
        accepted.append(acc)
        candidate_logs.append(
            {
                "idx": i,
                "accepted": True,
                "candidate": asdict(cand),
                "teacher": asdict(tv),
                "zpd": asdict(zpd_result) if zpd_result else None,
            }
        )

    # ------------------------------------------------------------------ #
    # 8) Save outputs
    # ------------------------------------------------------------------ #
    candidates_path = f"{out_dir}/candidates.jsonl"
    accepted_path = f"{out_dir}/accepted.jsonl"
    metrics_path = f"{out_dir}/metrics.json"

    write_jsonl(candidates_path, candidate_logs)
    write_jsonl(accepted_path, accepted)

    metrics = {
        "num_candidates": num_candidates,
        "num_accepted": len(accepted),
        "accept_rate": (len(accepted) / num_candidates) if num_candidates else 0.0,
        "zpd_enabled": zpd_enabled,
        "dedup": cfg["dedup"],
    }
    write_json(metrics_path, metrics)

    return out_dir

