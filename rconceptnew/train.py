"""
Simple GRPO training using HuggingFace TRL.
"""
import json
from pathlib import Path
from typing import List, Dict
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

        # Track the current checkpoint to use (we update this after each iteration)
        self.current_model_path = config.base_model

    def _free_cuda(self) -> None:
        """Best-effort GPU memory cleanup between vLLM inference and HF training."""
        import gc
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
            gpu_memory_utilization=self.config.gpu_memory_utilization,
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

        # Free vLLM before training
        del question_generator
        del vllm_model
        del vllm_tokenizer
        self._free_cuda()

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
            gpu_memory_utilization=self.config.gpu_memory_utilization,
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

        # Free vLLM before training
        del solver
        del vllm_model
        del vllm_tokenizer
        self._free_cuda()
        
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
        filtered = [q for q in questions if 0.3 <= q["reward"] <= 0.8]
        print(f"Filtered {len(filtered)}/{len(questions)} questions in ZPD range")
        
        if not filtered:
            print("Warning: No questions in ZPD range, using all questions")
            filtered = questions
        
        # Prepare prompts and responses for GRPO
        # GRPO expects: prompt + response pairs, with rewards
        prompts = []
        responses = []
        rewards = []
        
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
            
            # Use best rollout as response
            # Find rollout with answer matching majority
            rollouts = q["rollouts"]
            if rollouts:
                # Get majority answer
                from collections import Counter
                answers = [r["answer"] for r in rollouts if r.get("answer")]
                if answers:
                    majority_answer = Counter(answers).most_common(1)[0][0]
                    # Find rollout with majority answer
                    best_rollout = next(
                        (r for r in rollouts if r.get("answer") == majority_answer),
                        rollouts[0]
                    )
                    response = best_rollout["solution"]
                else:
                    response = rollouts[0]["solution"]
            else:
                response = ""
            
            prompts.append(prompt)
            responses.append(response)
            rewards.append(q["reward"])
        
        # Create dataset
        dataset_dict = {
            "prompt": prompts,
            "response": responses,
            "reward": rewards
        }
        
        return Dataset.from_dict(dataset_dict)
    
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
            logging_steps=5,
            save_steps=self.config.max_steps,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
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

        # Free training model from memory
        del trainer
        del self.model
        self._free_cuda()

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
