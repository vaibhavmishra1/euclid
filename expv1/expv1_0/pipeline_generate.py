from __future__ import annotations

import json
import random
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from .concept_graph import ConceptGraph
from .dedup import Deduper
from .explorator import GRIPStage0Explorator
from .generator import LLMQuestionGenerator
from .llm import build_llm_client
from .types import AcceptedSample, CandidateSample, Concept
from .utils import ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl
from .zpd import ZPDScorer


def _load_concepts_and_graph(concepts_dir: str) -> Tuple[List[List[Concept]], ConceptGraph]:
    """
    Load seed concepts and concept graph from concepts_dir.
    Supports both original format and canonical (cleaned) format.
    """
    import os

    # Determine which files to load (prefer canonical versions)
    seed_file = f"{concepts_dir}/seed_concepts_canonical.jsonl"
    graph_file = f"{concepts_dir}/concept_graph_canonical.json"

    if not os.path.exists(seed_file):
        seed_file = f"{concepts_dir}/seed_concepts.jsonl"
    if not os.path.exists(graph_file):
        graph_file = f"{concepts_dir}/concept_graph.json"

    print(f"[Pipeline B] Loading seeds from: {seed_file}")
    print(f"[Pipeline B] Loading graph from: {graph_file}")

    # Load seed concepts
    seed_concepts_rows = read_jsonl(seed_file)
    seed_concept_sets: List[List[Concept]] = []

    for row in seed_concepts_rows:
        concepts: List[Concept] = []

        # Try canonical format first (canonical_concepts with canonical_key)
        if "canonical_concepts" in row:
            for c in row.get("canonical_concepts", []) or []:
                key = str(c.get("canonical_key", ""))
                # canonical_key format: "concept:name" - extract just the name
                name = key.replace("concept:", "") if key.startswith("concept:") else key
                concepts.append(Concept(type="canonical", name=name.strip()))
        # Fall back to original format (concepts with type/name)
        else:
            for c in row.get("concepts", []) or []:
                concepts.append(Concept(type=str(c.get("type", "")).strip(), name=str(c.get("name", "")).strip()))

        seed_concept_sets.append([c for c in concepts if c.name])

    # Load concept graph
    with open(graph_file, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    concepts_by_key: Dict[str, Concept] = {}
    for k, v in (graph_data.get("concepts_by_key", {}) or {}).items():
        # Handle canonical format (canonical:name) or original format (type:name)
        if k.startswith("canonical:"):
            name = k.replace("canonical:", "")
            concepts_by_key[k] = Concept(type="canonical", name=name.strip())
        else:
            concepts_by_key[k] = Concept(type=str(v.get("type", "")).strip(), name=str(v.get("name", "")).strip())

    adjacency = graph_data.get("adjacency", {}) or {}
    graph = ConceptGraph(adjacency=adjacency, concepts_by_key=concepts_by_key)

    print(f"[Pipeline B] Loaded {len(seed_concept_sets)} seed concept sets, {len(concepts_by_key)} concepts in graph")
    return seed_concept_sets, graph


def generate_dataset_from_graph(config_path: str) -> str:
    """
    Pipeline B (GRIP Step 3+4):
    - Load prebuilt concept artifacts (seed concepts + KCRG)
    - Sample concept bundles (explorator)
    - Generate problems (generator)
    - Filter via dedup + solver-based ZPD/self-consistency
    - Save dataset outputs
    """

    cfg = load_yaml(config_path)
    random.seed(int(cfg["experiment"]["seed"]))

    out_dir = ensure_dir(cfg["io"]["output_dir"])
    concepts_dir = str(cfg.get("io", {}).get("concepts_dir", f"{out_dir}/concepts"))
    gen_log_dir = ensure_dir(f"{out_dir}/generator_io")

    seed_concept_sets, graph = _load_concepts_and_graph(concepts_dir)

    # ------------------------------------------------------------------ #
    # B1) Explorator policy
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
    # B2) Generator
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

    # ------------------------------------------------------------------ #
    # B3) Dedup + ZPD
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
    # B4) Main generation loop
    # ------------------------------------------------------------------ #
    num_candidates = int(cfg["generation"]["num_candidates"])
    candidate_logs: List[Dict[str, Any]] = []
    accepted: List[AcceptedSample] = []

    for i in range(num_candidates):
        spec = explorator.sample_spec()
        log_path = f"{gen_log_dir}/{i:06d}.txt"
        cand: CandidateSample = generator.generate_one(spec, log_path=log_path)

        # Gate 1: dedup
        dup_reason = deduper.check_duplicate(cand.problem)
        if dup_reason:
            candidate_logs.append({"idx": i, "accepted": False, "reject_reason": dup_reason, "candidate": asdict(cand)})
            continue

        # Gate 2: ZPD (self-consistency used as verification + difficulty)
        zpd_result = None
        if zpd_enabled and zpd is not None:
            zpd_result = zpd.score(cand.problem, "")
            modal_answer = zpd_result.details.get("modal_answer", "")
            if not modal_answer:
                candidate_logs.append(
                    {
                        "idx": i,
                        "accepted": False,
                        "reject_reason": "solver_no_answer",
                        "candidate": asdict(cand),
                        "verification": {"modal_answer": modal_answer, "p_succ": zpd_result.p_succ},
                        "zpd": asdict(zpd_result),
                    }
                )
                continue
            cand.answer = modal_answer
            if not (p_min <= zpd_result.p_succ <= p_max):
                candidate_logs.append(
                    {
                        "idx": i,
                        "accepted": False,
                        "reject_reason": "zpd_out_of_band",
                        "candidate": asdict(cand),
                        "verification": {"modal_answer": modal_answer, "p_succ": zpd_result.p_succ},
                        "zpd": asdict(zpd_result),
                    }
                )
                continue

        # Accept
        deduper.add(cand.problem)
        accepted.append(
            AcceptedSample(
                candidate=cand,
                verification={"modal_answer": cand.answer, "p_succ": zpd_result.p_succ if zpd_result else None},
                zpd=zpd_result,
            )
        )
        candidate_logs.append(
            {
                "idx": i,
                "accepted": True,
                "candidate": asdict(cand),
                "verification": {"modal_answer": cand.answer, "p_succ": zpd_result.p_succ if zpd_result else None},
                "zpd": asdict(zpd_result) if zpd_result else None,
            }
        )

    # ------------------------------------------------------------------ #
    # B5) Save outputs
    # ------------------------------------------------------------------ #
    write_jsonl(f"{out_dir}/candidates.jsonl", candidate_logs)
    write_jsonl(f"{out_dir}/accepted.jsonl", accepted)

    write_json(
        f"{out_dir}/metrics.json",
        {
            "num_candidates": num_candidates,
            "num_accepted": len(accepted),
            "accept_rate": (len(accepted) / num_candidates) if num_candidates else 0.0,
            "zpd_enabled": zpd_enabled,
            "dedup": cfg["dedup"],
            "concepts_dir": concepts_dir,
        },
    )

    return out_dir

