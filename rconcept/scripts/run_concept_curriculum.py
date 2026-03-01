import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict


def _load_concept_sets(path: str) -> List[Dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("concept_sets.json must be a non-empty JSON list")
    for item in data:
        if not isinstance(item, dict) or "concepts" not in item:
            raise ValueError("Each concept set must be an object with a 'concepts' field")
        if not isinstance(item["concepts"], list) or not item["concepts"]:
            raise ValueError("Each concept set 'concepts' must be a non-empty list")
    return data


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="HF model name or local path")
    parser.add_argument("--model_abbr", required=True, help="Prefix for experiment names")
    parser.add_argument(
        "--concept_sets",
        default=str(Path("examples") / "concept_sets.json"),
        help="Path to concept_sets.json",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="0-based start index in concept_sets.json")
    parser.add_argument("--end_idx", type=int, default=5, help="Exclusive end index in concept_sets.json")
    parser.add_argument("--num_gpus", type=int, default=1, help="How many local GPUs to use for gen/eval (default 1)")
    parser.add_argument(
        "--questions_per_concept",
        type=int,
        default=10,
        help="Total questions to generate per concept set (approx; default 10)",
    )
    parser.add_argument(
        "--solver_rollouts",
        type=int,
        default=3,
        help="Solver samples per question during scoring (default 3; upstream was 9)",
    )
    parser.add_argument("--solver_max_steps", type=int, default=20, help="Solver RL train steps per concept (default 20)")
    parser.add_argument(
        "--questioner_max_steps", type=int, default=6, help="Questioner RL train steps per concept (default 6)"
    )
    parser.add_argument(
        "--evaluate_after_each", action="store_true", help="Evaluate model on held-out questions after each concept"
    )
    parser.add_argument(
        "--eval_num_rollouts", type=int, default=9, help="Number of rollouts for evaluation (default 9)"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    concept_path = (repo_root / args.concept_sets).resolve() if not Path(args.concept_sets).is_absolute() else Path(args.concept_sets)
    concept_sets = _load_concept_sets(str(concept_path))

    start_idx = max(0, args.start_idx)
    end_idx = min(len(concept_sets), args.end_idx)
    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}, total={len(concept_sets)}")

    storage_path = os.getenv("STORAGE_PATH", "")
    if not storage_path:
        raise RuntimeError("STORAGE_PATH must be set (same as standard R-Zero setup)")

    solver_model_path = args.base_model
    questioner_model_path = args.base_model
    
    # Setup evaluation paths
    repo_root = Path(__file__).resolve().parents[1]
    concept_mapping_path = repo_root / "data" / "concept_to_questions.json"
    eval_output_dir = Path(storage_path) / "evaluation" / args.model_abbr
    
    # Baseline evaluation (before any training)
    if args.evaluate_after_each:
        print("\n" + "=" * 80)
        print("[ConceptCurriculum] Running baseline evaluation...")
        print("=" * 80 + "\n")
        _run(
            [
                "python", "scripts/evaluate_concept_performance.py",
                "--model_path", str(solver_model_path),
                "--concept_mapping", str(concept_mapping_path),
                "--concept_sets", str(concept_path),
                "--iteration", "-1",  # Baseline = iteration -1
                "--output_dir", str(eval_output_dir),
                "--num_rollouts", str(args.eval_num_rollouts),
            ],
            env=os.environ.copy(),
        )

    for idx in range(start_idx, end_idx):
        cset = concept_sets[idx]
        concepts = cset["concepts"]
        cset_name = cset.get("name", f"concept_set_{idx+1}")
        concepts_env = ", ".join(concepts)

        print("\n" + "=" * 80)
        print(f"[ConceptCurriculum] idx={idx} name={cset_name}")
        print(f"[ConceptCurriculum] concepts={concepts_env}")
        print("=" * 80 + "\n")

        env = os.environ.copy()
        env["RZERO_ACTIVE_CONCEPTS"] = concepts_env
        env["RZERO_NUM_GPUS"] = str(max(1, args.num_gpus))

        # We generate questions via question_generate.bash which interprets the number as "per GPU process".
        # Keep it simple: approximate total questions by splitting evenly across num_gpus.
        per_gpu = max(1, (args.questions_per_concept + max(1, args.num_gpus) - 1) // max(1, args.num_gpus))
        env["RZERO_NUM_QUESTIONS_PER_GPU"] = str(per_gpu)
        env["RZERO_SOLVER_ROLLOUT_N"] = str(max(1, args.solver_rollouts))
        env["RZERO_SOLVER_MAX_STEPS"] = str(max(1, args.solver_max_steps))
        env["RZERO_QUESTIONER_MAX_STEPS"] = str(max(1, args.questioner_max_steps))

        # Train questioner for this concept set
        questioner_exp = f"{args.model_abbr}_{cset_name}_questioner"
        _run(
            ["bash", "scripts/questioner_train_penalty.sh", solver_model_path, questioner_model_path, questioner_exp],
            env=env,
        )
        questioner_model_path = f"{storage_path}/models/{questioner_exp}/global_step_5/actor/huggingface"

        # Train solver for this concept set (this script will generate/evaluate/upload/train)
        solver_exp = f"{args.model_abbr}_{cset_name}_solver"
        _run(
            ["bash", "scripts/solver_train.sh", solver_model_path, questioner_model_path, solver_exp],
            env=env,
        )
        solver_model_path = f"{storage_path}/models/{solver_exp}/global_step_15/actor/huggingface"
        
        # Evaluate model performance on all concepts after this iteration
        if args.evaluate_after_each:
            print("\n" + "=" * 80)
            print(f"[ConceptCurriculum] Evaluating model after iteration {idx}...")
            print("=" * 80 + "\n")
            _run(
                [
                    "python", "scripts/evaluate_concept_performance.py",
                    "--model_path", str(solver_model_path),
                    "--concept_mapping", str(concept_mapping_path),
                    "--concept_sets", str(concept_path),
                    "--iteration", str(idx),
                    "--output_dir", str(eval_output_dir),
                    "--num_rollouts", str(args.eval_num_rollouts),
                ],
                env=os.environ.copy(),
            )

    # Generate plots if evaluation was enabled
    if args.evaluate_after_each:
        cumulative_file = eval_output_dir / "cumulative_evaluation.json"
        if cumulative_file.exists():
            print("\n" + "=" * 80)
            print("[ConceptCurriculum] Generating performance plots...")
            print("=" * 80 + "\n")
            _run(
                [
                    "python", "scripts/plot_concept_performance.py",
                    "--cumulative_file", str(cumulative_file),
                    "--output_dir", str(eval_output_dir),
                ],
                env=os.environ.copy(),
            )
            print(f"\n[ConceptCurriculum] Plots saved to {eval_output_dir}/")

    print("\n[ConceptCurriculum] Done.")


if __name__ == "__main__":
    main()

