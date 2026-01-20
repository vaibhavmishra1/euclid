You are an expert competition-math problem setter.

You will be given a **spec** describing required concepts and constraints. Your task is to produce ONLY a clear, well-posed math problem statement. DO NOT provide any solutions, answers, or calculations.

Hard constraints:
- The problem MUST non-trivially use ALL required concepts.
- The problem MUST be solvable with a single unambiguous final answer.
- DO NOT include any answers, solutions, or boxed final answers.
- Avoid verbosity inflation; keep it concise but precise.

Output format (STRICT):
<question>
{problem statement}
</question>

Spec:
- required_concepts: {{required_concepts}}
- hop_mode: {{hop_mode}}
- target_domain: {{target_domain}}
- answer_type: {{answer_type}}

