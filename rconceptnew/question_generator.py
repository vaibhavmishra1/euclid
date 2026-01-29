"""
Simple question generator using vLLM.
"""
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class QuestionGenerator:
    """Generate questions using vLLM with concept conditioning."""
    
    def __init__(self, model_path: str = None, llm: LLM = None, tokenizer = None, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        """Initialize the question generator.
        
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
            print(f"Loading model {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.2,
            )
            print("Model loaded!")
    
    def _build_prompt(self, concepts: List[str]) -> str:
        """Build the prompt for question generation."""
        concepts_str = ", ".join(concepts)
        return f"""You are an expert competition-math problem setter.

FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem.
The problem MUST non-trivially use ALL of these concepts: {concepts_str}.
The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus.
Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it.
Avoid re-using textbook clichés or famous contest problems.

THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:

<question>
{{The full problem statement on one or more lines}}
</question>

\\boxed{{final_answer}}

Do NOT output anything else—no explanations, no extra markup.

Generate one new, challenging reasoning question now. Remember to format the output exactly as instructed."""
    
    def generate(
        self,
        concepts: List[str],
        num_questions: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
    ) -> List[Dict]:
        """Generate questions for given concepts."""
        prompt = self._build_prompt(concepts)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
        )
        
        # Generate multiple questions
        prompts = [prompt] * num_questions
        outputs = self.llm.generate(prompts, sampling_params)
        
        questions = []
        for output in outputs:
            generated_text = output.outputs[0].text
            # Extract question and answer
            question = self._extract_question(generated_text)
            answer = self._extract_answer(generated_text)
            
            if question and answer:
                questions.append({
                    "question": question,
                    "answer": answer,
                    "concepts": concepts,
                    "raw_output": generated_text
                })
        
        return questions
    
    def _extract_question(self, text: str) -> str:
        """Extract question from generated text."""
        if "<question>" in text and "</question>" in text:
            start = text.find("<question>") + len("<question>")
            end = text.find("</question>")
            return text[start:end].strip()
        return ""
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from boxed content."""
        if "\\boxed{" in text:
            start = text.find("\\boxed{") + len("\\boxed{")
            # Find matching brace
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
