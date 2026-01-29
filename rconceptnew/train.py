"""
Simple GRPO training using HuggingFace TRL.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Callable
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# NOTE: `trl` is intentionally optional in this repo environment (no network install in some setups).
# If you run this pipeline, ensure you've installed deps from requirements.txt.
from trl import GRPOConfig, GRPOTrainer  # type: ignore
from vllm import LLM  # type: ignore
from config import Config
from question_generator import QuestionGenerator
from solver import Solver
from reward import compute_rewards_for_batch


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...} content. Returns empty string if not found."""
    if "\\boxed{" in text:
        start = text.find("\\boxed{") + len("\\boxed{")
        brace_count = 1
        end = start
        while end < len(text) and brace_count > 0:
            if text[end] == "{":
                brace_count += 1
            elif text[end] == "}":
                brace_count -= 1
            end += 1
        if brace_count == 0:
            return text[start : end - 1].strip()
    return ""


def shutdown_vllm_engine(llm: object) -> None:
    """
    Best-effort vLLM shutdown to release GPU memory held by engine core processes.

    vLLM v0.12 does not expose LLM.shutdown(), but the underlying engine core client
    does have a shutdown method.
    """
    try:
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None) if engine is not None else None
        shutdown = getattr(core, "shutdown", None) if core is not None else None
        if callable(shutdown):
            shutdown()
    except Exception:
        # Best effort: avoid crashing training on cleanup.
        pass


def cleanup_vllm_distributed_state() -> None:
    """
    Best-effort cleanup of vLLM's global distributed state.

    When using TRL's `use_vllm=True` with `vllm_mode="colocate"`, vLLM initializes
    torch.distributed process groups and also keeps global group handles inside
    vLLM's `distributed.parallel_state`. If we don't reset those globals, a
    subsequent vLLM initialization in the *same Python process* can fail with:
      "Process group ... is not initialized in the world group map".
    """
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception:
        # Best effort: training should not fail because cleanup failed.
        pass


def make_self_consistency_reward(num_generations: int) -> Callable[[List[str], List[str]], List[float]]:
    """
    Build a GRPO-compatible reward function.

    TRL GRPO calls reward funcs with flat lists (one reward per completion).
    Prompts are repeated `num_generations` times consecutively, once per completion.
    """
    group_size = max(1, int(num_generations))

    def reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # Self-consistency-style reward: 1.0 if completion agrees with the group's majority boxed answer, else 0.0.
        # Encourages consistent "final answers" across sampled completions.
        _ = prompts  # prompts are not needed beyond grouping assumption
        answers = [extract_boxed_answer(c) for c in completions]
        rewards: List[float] = []

        from collections import Counter

        for i in range(0, len(completions), group_size):
            group_answers = answers[i : i + group_size]
            non_empty = [a for a in group_answers if a]
            if not non_empty:
                rewards.extend([0.0] * len(group_answers))
                continue

            majority_answer = Counter(non_empty).most_common(1)[0][0]
            for a in group_answers:
                rewards.append(1.0 if a and a == majority_answer else 0.0)

        # In case TRL ever calls with non-multiple sizes, trim to exact length.
        return rewards[: len(completions)]

    return reward_func


class SimpleRZeroTrainer:
    """Simplified R-Zero trainer using TRL GRPO."""
    
    def __init__(self, config: Config):
        """Initialize trainer."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load concept sets
        with open(config.concept_sets_file, 'r') as f:
            self.concept_sets = json.load(f)

        # Load held-out question bank for evaluation/progress tracking (optional)
        self._eval_question_bank: Dict[str, List[str]] = {}
        if getattr(config, "eval_enabled", False):
            try:
                with open(config.eval_question_bank_file, "r") as f:
                    bank = json.load(f)
                if isinstance(bank, dict):
                    self._eval_question_bank = bank
                else:
                    print(f"[Eval] Warning: eval_question_bank_file='{config.eval_question_bank_file}' is not a dict.")
            except Exception as e:
                print(f"[Eval] Warning: failed to load eval_question_bank_file='{config.eval_question_bank_file}': {e}")
                self._eval_question_bank = {}

        # Track the current checkpoint to use (we update this after each iteration)
        self.current_model_path = config.base_model

    def _free_cuda(self, aggressive: bool = False) -> None:
        """Best-effort GPU memory cleanup between vLLM inference and HF training."""
        import gc
        import time
        # Multiple gc passes to break reference cycles.
        for _ in range(3 if aggressive else 1):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if aggressive:
            # Give CUDA a moment to finish releasing memory asynchronously.
            time.sleep(1.0)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def generate_questions(self, concepts: List[str]) -> List[Dict]:
        """Generate questions for given concepts."""
        print(f"Generating {self.config.questions_per_concept} questions for concepts: {concepts}")
        # vLLM inference phase (only keep vLLM loaded during gen/eval)
        vllm_model = LLM(
            model=self.current_model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=0.2,
            max_num_seqs=self.config.vllm_max_num_seqs,
        )
        vllm_tokenizer = AutoTokenizer.from_pretrained(self.current_model_path)
        question_generator = QuestionGenerator(model_path=self.current_model_path, llm=vllm_model, tokenizer=vllm_tokenizer)

        questions = question_generator.generate(
            concepts=concepts,
            num_questions=self.config.questions_per_concept,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
        )

        # Free vLLM before moving to next phase
        del question_generator
        shutdown_vllm_engine(vllm_model)
        del vllm_model
        del vllm_tokenizer
        cleanup_vllm_distributed_state()
        self._free_cuda(aggressive=True)

        print(f"Generated {len(questions)} questions")
        return questions
    
    def evaluate_questions(self, questions: List[Dict]) -> List[Dict]:
        """Evaluate questions using solver and compute rewards."""
        question_texts = [q["question"] for q in questions]
        
        print(f"Solving {len(question_texts)} questions with {self.config.num_rollouts} rollouts each...")
        # vLLM inference phase (re-load vLLM for solving)
        vllm_model = LLM(
            model=self.current_model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=0.2,
            max_num_seqs=self.config.vllm_max_num_seqs,
        )
        vllm_tokenizer = AutoTokenizer.from_pretrained(self.current_model_path)
        solver = Solver(model_path=self.current_model_path, llm=vllm_model, tokenizer=vllm_tokenizer)

        rollouts = solver.solve(
            questions=question_texts,
            num_rollouts=self.config.num_rollouts,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
        )

        # Free vLLM before training (HF model + colocated vLLM need the memory)
        del solver
        shutdown_vllm_engine(vllm_model)
        del vllm_model
        del vllm_tokenizer
        cleanup_vllm_distributed_state()
        self._free_cuda(aggressive=True)
        
        # Compute rewards
        rewards = compute_rewards_for_batch(question_texts, rollouts)
        
        # Combine with questions
        evaluated_questions = []
        for q, r, rollout_list in zip(questions, rewards, rollouts):
            evaluated_questions.append({
                **q,
                "reward": r,
                "rollouts": rollout_list
            })
        
        return evaluated_questions
    
    def prepare_dataset(self, questions: List[Dict]) -> Dataset:
        """Prepare dataset for GRPO training."""
        # Filter questions with reasonable rewards (ZPD filtering)
        filtered = [q for q in questions if 0.3 <= q["reward"] <= 1.0]
        print(f"Filtered {len(filtered)}/{len(questions)} questions in ZPD range")
        
        if not filtered:
            print("Warning: No questions in ZPD range, using all questions")
            filtered = questions
        
        # Prepare prompts for GRPO.
        # In TRL v0.27+, GRPO is *online*: it generates completions during training and calls `reward_funcs`.
        # The dataset must include a "prompt" column; any additional columns are ignored by GRPOTrainer.
        prompts = []
        
        for q in filtered:
            # Format prompt with system message
            question_text = q["question"]
            # Tokenizer from the *training* checkpoint (not vLLM tokenizer) is created in train_on_concept.
            if self.tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": question_text}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {question_text}\n\nAnswer:"
            prompts.append(prompt)
        
        # Create dataset
        dataset_dict = {
            "prompt": prompts
        }
        
        return Dataset.from_dict(dataset_dict)

    def _find_concept_key_in_eval_bank(self, concept: str) -> str | None:
        """Best-effort mapping from a concept name to a key in concept_to_questions.json."""
        if not self._eval_question_bank:
            return None
        possible_keys = [
            f"concept:{concept}",
            concept,
            concept.lower(),
            f"concept:{concept.lower()}",
        ]
        for key in possible_keys:
            if key in self._eval_question_bank:
                return key
        return None

    def evaluate_concepts_after_iteration(self, concepts: List[str], iteration: int, model_path: str) -> None:
        """
        Track per-concept performance after each concept iteration.

        This uses a held-out static question bank (config.eval_question_bank_file) and a simple
        self-consistency score (majority agreement fraction), matching `reward.py`.
        """
        if not getattr(self.config, "eval_enabled", False):
            return
        if not self._eval_question_bank:
            print("[Eval] Skipping: eval question bank is empty/unavailable.")
            return

        eval_dir = self.output_dir / getattr(self.config, "eval_output_subdir", "evals")
        eval_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Eval] Evaluating concepts after iteration {iteration} using checkpoint: {model_path}")

        results: List[Dict] = []
        rng = random.Random(1337 + int(iteration))

        # One vLLM instance for all concepts to keep eval fast.
        # Use a lower gpu_memory_utilization for eval since it runs right after GRPO
        # which may not fully release GPU memory immediately.
        eval_mem_util = 0.1
        vllm_model = LLM(
            model=model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=0.2,
            max_num_seqs=self.config.vllm_max_num_seqs,
        )
        vllm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        solver = Solver(model_path=model_path, llm=vllm_model, tokenizer=vllm_tokenizer)

        try:
            for concept in concepts:
                concept_key = self._find_concept_key_in_eval_bank(concept)
                if concept_key is None:
                    print(f"[Eval] Warning: concept '{concept}' not found in eval bank; skipping.")
                    continue

                bank_questions = list(self._eval_question_bank.get(concept_key, []))
                if not bank_questions:
                    continue

                k = max(1, int(getattr(self.config, "eval_questions_per_concept", 20)))
                if len(bank_questions) > k:
                    bank_questions = rng.sample(bank_questions, k)

                rollouts = solver.solve(
                    questions=bank_questions,
                    num_rollouts=self.config.eval_num_rollouts,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_new_tokens=self.config.max_new_tokens,
                )
                scores = compute_rewards_for_batch(bank_questions, rollouts)
                avg_score = float(sum(scores) / len(scores)) if scores else 0.0

                results.append(
                    {
                        "iteration": int(iteration),
                        "concept": concept,
                        "concept_key": concept_key,
                        "num_questions": len(bank_questions),
                        "num_rollouts": int(self.config.eval_num_rollouts),
                        "avg_score": avg_score,
                        "scores": scores,
                    }
                )

                print(f"[Eval] {concept}: avg_score={avg_score:.3f} ({len(bank_questions)} qs)")
        finally:
            # Free vLLM after eval — must be aggressive to release GPU memory
            # before the next iteration's question generation starts.
            del solver
            shutdown_vllm_engine(vllm_model)
            del vllm_model
            del vllm_tokenizer
            cleanup_vllm_distributed_state()
            self._free_cuda(aggressive=True)

        # Write per-iteration evaluation + cumulative
        out_file = eval_dir / f"iteration_{int(iteration)}_evaluation.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Eval] Saved {out_file}")

        cumulative_file = eval_dir / "cumulative_evaluation.json"
        if cumulative_file.exists():
            try:
                with open(cumulative_file, "r") as f:
                    cumulative = json.load(f)
                if not isinstance(cumulative, list):
                    cumulative = []
            except Exception:
                cumulative = []
        else:
            cumulative = []
        cumulative.extend(results)
        with open(cumulative_file, "w") as f:
            json.dump(cumulative, f, indent=2)
    
    def train_on_concept(self, concepts: List[str], iteration: int) -> str:
        """Train model on a single concept."""
        print(f"\n{'='*60}")
        print(f"Training on concepts: {concepts} (Iteration {iteration})")
        print(f"{'='*60}\n")
        
        # Generate questions
        questions = self.generate_questions(concepts)
        
        # Evaluate questions
        evaluated_questions = self.evaluate_questions(questions)
        
        # Load model/tokenizer for training *after* vLLM is freed (avoids OOM).
        self.model = AutoModelForCausalLM.from_pretrained(
            self.current_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.current_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # TRL GRPO requires left padding for generation batching.
        self.tokenizer.padding_side = "left"

        # Prepare dataset (needs self.tokenizer)
        dataset = self.prepare_dataset(evaluated_questions)
        
        # Setup GRPO training
        # GRPOConfig contains training arguments
        grpo_config = GRPOConfig(
            output_dir=str(self.output_dir / f"concept_{iteration}"),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_steps=self.config.max_steps,
            logging_steps=self.config.max_steps,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            # Generation params (online GRPO sampling)
            num_generations=self.config.num_rollouts,
            max_completion_length=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            # Use vLLM for faster online generation during GRPO (TRL v0.27+).
            use_vllm=self.config.grpo_use_vllm,
            vllm_mode=self.config.grpo_vllm_mode,
            vllm_gpu_memory_utilization=0.1,
            vllm_tensor_parallel_size=self.config.grpo_vllm_tensor_parallel_size,
            vllm_max_model_length=self.config.grpo_vllm_max_model_length,
        )
        
        # Create trainer
        reward_func = make_self_consistency_reward(num_generations=grpo_config.num_generations or 1)
        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_func,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        # Train
        print("Starting GRPO training...")
        trainer.train()
        
        # Save model (include concept + iteration to avoid overwriting across concept sets)
        concept_slug = "_".join(str(c).strip().replace(" ", "_") for c in concepts)
        save_path = self.output_dir / f"{concept_slug}_iter_{iteration}_model"
        trainer.save_model(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        print(f"Model saved to {save_path}")

        # If TRL started a colocated vLLM instance inside GRPOTrainer, attempt to
        # shut it down before freeing CUDA / starting a new vLLM instance in the
        # next iteration.
        try:
            if getattr(self.config, "grpo_use_vllm", False) and getattr(self.config, "grpo_vllm_mode", "") == "colocate":
                llm = getattr(trainer, "llm", None)
                if llm is not None:
                    shutdown_vllm_engine(llm)
        except Exception:
            pass

        # Free training model from memory
        del trainer
        del self.model
        # Avoid lingering vLLM + torch.distributed state across iterations (critical
        # when mixing TRL colocated vLLM + standalone vLLM LLM usage).
        cleanup_vllm_distributed_state()
        # Aggressive cleanup to ensure GPU memory is actually released before
        # evaluation (which will spin up another vLLM instance).
        self._free_cuda(aggressive=True)

        # Update current checkpoint for next iteration / next concept
        self.current_model_path = str(save_path)
        return self.current_model_path
    
    def run(self):
        """Run the full training loop."""
        print("Starting R-Zero training pipeline...")
        print(f"Concepts: {len(self.concept_sets)}")
        print(f"Iterations per concept: {self.config.num_iterations}")
        
        for concept_idx, concept_set in enumerate(self.concept_sets[:self.config.num_concepts]):
            concepts = concept_set["concepts"]
            concept_name = concept_set.get("name", f"concept_{concept_idx}")
            
            print(f"\n{'#'*60}")
            print(f"Concept Set {concept_idx + 1}/{self.config.num_concepts}: {concept_name}")
            print(f"Concepts: {concepts}")
            print(f"{'#'*60}\n")
            
            # Train for multiple iterations
            for iteration in range(self.config.num_iterations):
                model_path = self.train_on_concept(concepts, iteration)
                # Track held-out concept performance after each iteration.
                self.evaluate_concepts_after_iteration(concepts=concepts, iteration=iteration, model_path=model_path)
                print(f"Completed iteration {iteration + 1}/{self.config.num_iterations}")
        
        print("\n" + "="*60)
        print("Training complete!")
        print(f"Final model saved at: {model_path}")
        print("="*60)


def main():
    """Main entry point."""
    config = Config()
    trainer = SimpleRZeroTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
