"""
Curriculum Trainer for Knowledge-Set-Based R-Zero

This module implements the main training loop that:
1. Generates questions for knowledge SETS (groups of KPs)
2. Evaluates questions using the Solver (uses R-Zero's reward as-is)
3. Adjusts difficulty levels based on average reward per set
4. Trains the Solver and Challenger
5. Iterates the co-evolutionary process

Key points:
- 868 knowledge sets = 868 original questions
- 5 questions generated per set = 4,340 questions per iteration
- Reward per set = average of 5 question rewards
- Difficulty adjustment based on alpha/beta thresholds
"""

import json
import os
import subprocess
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from .knowledge_manager import KnowledgeSetManager


@dataclass
class CurriculumConfig:
    """Configuration for curriculum-based training."""
    # Model paths
    base_model: str = "Qwen/Qwen3-4B"
    model_abbr: str = "qwen3-4b-ks"
    
    # Knowledge set settings
    knowledge_points_path: str = ""
    alpha: float = 0.7  # Upper threshold - reward > alpha means too easy
    beta: float = 0.3   # Lower threshold - reward < beta means too hard
    questions_per_set: int = 5  # Questions per knowledge set
    
    # Training settings
    num_iterations: int = 3
    challenger_steps: int = 5
    solver_steps: int = 15
    
    # Paths
    storage_path: str = "/tmp/rzero_storage"
    state_save_path: str = ""
    
    def __post_init__(self):
        if not self.state_save_path:
            self.state_save_path = f"{self.storage_path}/ks_state/{self.model_abbr}_state.json"


class CurriculumTrainer:
    """
    Main trainer for knowledge-set-based curriculum learning.
    
    Training loop:
    1. Initialize 868 knowledge sets with difficulty=1
    2. For each iteration:
       a. Generate 5 questions per set using Challenger
       b. Evaluate all questions using Solver (R-Zero reward)
       c. Compute average reward per knowledge set
       d. Update difficulties: avg > alpha → increase, avg < beta → decrease
       e. Train Solver on questions
       f. Train Challenger on updated prompts
    """
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        
        # Ensure directories exist
        os.makedirs(config.storage_path, exist_ok=True)
        os.makedirs(f"{config.storage_path}/models", exist_ok=True)
        os.makedirs(f"{config.storage_path}/generated_question", exist_ok=True)
        os.makedirs(os.path.dirname(config.state_save_path), exist_ok=True)
        
        # Initialize knowledge set manager
        self.ks_manager = KnowledgeSetManager(
            knowledge_points_path=config.knowledge_points_path,
            alpha=config.alpha,
            beta=config.beta,
            questions_per_set=config.questions_per_set,
            state_save_path=config.state_save_path,
        )
        
        print(f"\n{'='*60}")
        print("Curriculum Trainer Initialized")
        print(f"{'='*60}")
        print(f"Base Model: {config.base_model}")
        print(f"Knowledge Sets: {len(self.ks_manager.knowledge_sets)}")
        print(f"Questions per Set: {config.questions_per_set}")
        print(f"Total questions per iteration: {len(self.ks_manager.knowledge_sets) * config.questions_per_set}")
        print(f"Alpha (increase if reward >): {config.alpha}")
        print(f"Beta (decrease if reward <): {config.beta}")
        print(f"Iterations: {config.num_iterations}")
        print(f"{'='*60}\n")
    
    def run(self):
        """Run the full training loop."""
        solver_model = self.config.base_model
        challenger_model = self.config.base_model
        
        for iteration in range(1, self.config.num_iterations + 1):
            print(f"\n{'#'*60}")
            print(f"# ITERATION {iteration}/{self.config.num_iterations}")
            print(f"{'#'*60}\n")
            
            # Print current statistics
            stats = self.ks_manager.get_statistics()
            print(f"Current Statistics:")
            print(f"  Knowledge Sets: {stats['total_knowledge_sets']}")
            print(f"  Avg Difficulty: {stats['avg_difficulty']:.2f}")
            print(f"  Distribution: {stats['difficulty_distribution']}")
            
            # Step 1: Train Challenger
            print(f"\n[Step 1] Training Challenger...")
            challenger_save_name = f"{self.config.model_abbr}_challenger_v{iteration}"
            new_challenger_model = self._train_challenger(
                solver_model=solver_model,
                challenger_model=challenger_model,
                save_name=challenger_save_name,
            )
            
            if new_challenger_model:
                challenger_model = new_challenger_model
            
            # Step 2: Generate Questions
            print(f"\n[Step 2] Generating {len(self.ks_manager.knowledge_sets) * self.config.questions_per_set} questions...")
            questions_path = self._generate_questions(
                challenger_model=challenger_model,
                save_name=f"{self.config.model_abbr}_iter{iteration}",
            )
            
            # Step 3: Evaluate Questions with Solver (uses R-Zero's reward)
            print(f"\n[Step 3] Evaluating questions with Solver...")
            evaluated_path = self._evaluate_questions(
                solver_model=solver_model,
                questions_path=questions_path,
                save_name=f"{self.config.model_abbr}_iter{iteration}",
            )
            
            # Step 4: Update difficulties based on average rewards per set
            print(f"\n[Step 4] Updating difficulties...")
            self._update_difficulties_from_results(evaluated_path)
            
            # Print updated statistics
            stats = self.ks_manager.get_statistics()
            print(f"Updated Statistics:")
            print(f"  Avg Difficulty: {stats['avg_difficulty']:.2f}")
            print(f"  Distribution: {stats['difficulty_distribution']}")
            
            # Save state
            self.ks_manager.save_state()
            
            # Step 5: Train Solver
            print(f"\n[Step 5] Training Solver...")
            solver_save_name = f"{self.config.model_abbr}_solver_v{iteration}"
            new_solver_model = self._train_solver(
                solver_model=solver_model,
                questions_path=evaluated_path,
                save_name=solver_save_name,
            )
            
            if new_solver_model:
                solver_model = new_solver_model
            
            print(f"\nIteration {iteration} complete!")
            print(f"  Challenger: {challenger_model}")
            print(f"  Solver: {solver_model}")
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Final Solver: {solver_model}")
        print(f"Final Challenger: {challenger_model}")
        print(f"{'='*60}\n")
        
        return solver_model, challenger_model
    
    def _train_challenger(
        self,
        solver_model: str,
        challenger_model: str,
        save_name: str,
    ) -> Optional[str]:
        """Train the challenger model."""
        # Export training prompts for this iteration
        prompts_path = f"{self.config.storage_path}/training_prompts_{save_name}.jsonl"
        self.ks_manager.export_training_prompts(prompts_path)
        
        # Run challenger training script
        cmd = [
            "bash", "scripts/kp_challenger_train.sh",
            solver_model,
            challenger_model,
            save_name,
            self.config.knowledge_points_path,
            str(self.config.alpha),
            str(self.config.beta),
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
            
            new_model_path = f"{self.config.storage_path}/models/{save_name}/global_step_{self.config.challenger_steps}/actor/huggingface"
            if os.path.exists(new_model_path):
                return new_model_path
        except subprocess.CalledProcessError as e:
            print(f"Warning: Challenger training failed: {e}")
        except FileNotFoundError:
            print("Warning: Training script not found, skipping challenger training")
        
        return None
    
    def _generate_questions(
        self,
        challenger_model: str,
        save_name: str,
    ) -> str:
        """Generate questions using the challenger model."""
        output_path = f"{self.config.storage_path}/generated_question/{save_name}_0.json"
        
        cmd = [
            "python", "-m", "knowledge_curriculum.question_generate",
            "--model", challenger_model,
            "--save_name", save_name,
            "--suffix", "0",
            "--knowledge_points_path", self.config.knowledge_points_path,
            "--state_path", self.config.state_save_path,
            "--alpha", str(self.config.alpha),
            "--beta", str(self.config.beta),
            "--questions_per_set", str(self.config.questions_per_set),
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        except subprocess.CalledProcessError as e:
            print(f"Warning: Question generation failed: {e}")
        except FileNotFoundError:
            print("Warning: Generation script not found")
        
        return output_path
    
    def _evaluate_questions(
        self,
        solver_model: str,
        questions_path: str,
        save_name: str,
    ) -> str:
        """Evaluate generated questions using the solver (R-Zero's reward)."""
        output_path = f"{self.config.storage_path}/generated_question/{save_name}_evaluated.json"
        
        cmd = [
            "bash", "question_evaluate/evaluate.sh",
            solver_model,
            save_name,
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        except subprocess.CalledProcessError as e:
            print(f"Warning: Evaluation failed: {e}")
        except FileNotFoundError:
            print("Warning: Evaluation script not found")
        
        # Combine results from all GPUs
        combined_results = []
        for i in range(8):
            try:
                result_file = f"{self.config.storage_path}/generated_question/{save_name}_{i}_results.json"
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        combined_results.extend(json.load(f))
            except:
                pass
        
        # If no multi-GPU results, try loading original
        if not combined_results:
            try:
                with open(questions_path, 'r') as f:
                    combined_results = json.load(f)
            except:
                pass
        
        with open(output_path, 'w') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _update_difficulties_from_results(self, results_path: str):
        """
        Update knowledge set difficulties based on evaluation results.
        
        For each set:
        - Compute average reward of its 5 questions
        - If avg > alpha: increase difficulty (too easy)
        - If avg < beta: decrease difficulty (too hard)
        """
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
        except:
            print(f"Warning: Could not load results from {results_path}")
            return
        
        # Record rewards for each knowledge set
        for result in results:
            set_id = result.get('set_id')
            # R-Zero's reward is stored in 'score' field
            reward = result.get('score', result.get('reward', 0.5))
            
            if set_id is not None:
                self.ks_manager.record_reward(set_id, reward)
        
        # Update difficulties based on average rewards
        adjustments = self.ks_manager.update_difficulties()
        
        # Print summary
        increased = sum(1 for a in adjustments.values() if a['action'] == 'increased')
        decreased = sum(1 for a in adjustments.values() if a['action'] == 'decreased')
        maintained = sum(1 for a in adjustments.values() if a['action'] == 'maintained')
        skipped = sum(1 for a in adjustments.values() if a['action'] == 'skipped')
        
        print(f"Difficulty adjustments:")
        print(f"  Increased (too easy): {increased}")
        print(f"  Decreased (too hard): {decreased}")
        print(f"  Maintained (sweet spot): {maintained}")
        print(f"  Skipped (no data): {skipped}")
    
    def _train_solver(
        self,
        solver_model: str,
        questions_path: str,
        save_name: str,
    ) -> Optional[str]:
        """Train the solver model on questions."""
        cmd = [
            "bash", "scripts/kp_solver_train.sh",
            solver_model,
            questions_path,
            save_name,
            str(self.config.beta),
            str(self.config.alpha),
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
            
            new_model_path = f"{self.config.storage_path}/models/{save_name}/global_step_{self.config.solver_steps}/actor/huggingface"
            if os.path.exists(new_model_path):
                return new_model_path
        except subprocess.CalledProcessError as e:
            print(f"Warning: Solver training failed: {e}")
        except FileNotFoundError:
            print("Warning: Training script not found, skipping solver training")
        
        return None


def main():
    """Run curriculum training from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge-Set-Based Curriculum Training")
    
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B",
                        help="Base model path")
    parser.add_argument("--model_abbr", type=str, default="qwen3-4b-ks",
                        help="Model abbreviation for saving")
    parser.add_argument("--knowledge_points_path", type=str, required=True,
                        help="Path to knowledge points JSONL file")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Upper threshold - increase difficulty if reward > alpha")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Lower threshold - decrease difficulty if reward < beta")
    parser.add_argument("--questions_per_set", type=int, default=5,
                        help="Questions to generate per knowledge set")
    parser.add_argument("--num_iterations", type=int, default=3,
                        help="Number of training iterations")
    parser.add_argument("--storage_path", type=str, default=None,
                        help="Storage path for models and data")
    
    args = parser.parse_args()
    
    storage_path = args.storage_path or os.getenv("STORAGE_PATH", "/tmp/rzero_storage")
    
    config = CurriculumConfig(
        base_model=args.base_model,
        model_abbr=args.model_abbr,
        knowledge_points_path=args.knowledge_points_path,
        alpha=args.alpha,
        beta=args.beta,
        questions_per_set=args.questions_per_set,
        num_iterations=args.num_iterations,
        storage_path=storage_path,
    )
    
    trainer = CurriculumTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
