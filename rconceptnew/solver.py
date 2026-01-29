"""
Solver that generates solutions using vLLM.
"""
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class Solver:
    """Solve questions using vLLM."""
    
    def __init__(self, model_path: str = None, llm = None, tokenizer = None, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        """Initialize the solver.
        
        Args:
            model_path: Path to model (if creating new vLLM instance)
            llm: Existing vLLM instance to reuse (optional)
            tokenizer: Existing tokenizer to reuse (optional)
            tensor_parallel_size: For new vLLM instance
            gpu_memory_utilization: For new vLLM instance
        """
        if llm is not None:
            # Reuse existing vLLM instance
            self.llm = llm
            self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_path)
        else:
            # Create new vLLM instance
            print(f"Loading solver model {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.2,
            )
            print("Solver model loaded!")
    
    def solve(
        self,
        questions: List[str],
        num_rollouts: int = 4,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        top_p: float = 0.95,
    ) -> List[List[Dict]]:
        """Solve questions with multiple rollouts."""
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        
        # Build prompts
        prompts = []
        for question in questions:
            if self.tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"system: {system_prompt}\nuser: {question}\nassistant: "
            prompts.append(prompt)
        
        # Generate with multiple rollouts
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=num_rollouts,  # Number of rollouts
            stop_token_ids=[self.tokenizer.eos_token_id],
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Process outputs
        results = []
        for output in outputs:
            question_rollouts = []
            for choice in output.outputs:
                solution = choice.text
                answer = self._extract_answer(solution)
                question_rollouts.append({
                    "solution": solution,
                    "answer": answer
                })
            results.append(question_rollouts)
        
        return results
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from boxed content."""
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
                return text[start:end-1].strip()
        return ""
