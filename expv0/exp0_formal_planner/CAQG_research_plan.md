# Constraint-Driven Question Genesis: A Mathematical Framework for Self-Evolving Problem Generation

**Project Codename**: CAQG (Constraint-Aware Question Genesis)  
**Version**: 1.0  
**Date**: January 14, 2026

---

## Executive Summary

A formal experimental framework to demonstrate that the capability of self-evolving AI to generate novel mathematical problems is conceptually distinct from and foundational to problem-solving capabilities, leveraging constraint perturbation theory rather than reward-driven optimization.

---

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Build constraint extraction module that parses mathematical problems into explicit constraint sets with domain identification and dependency graphs | Pending |
| Phase 2 | Implement perturbation operators (tighten, relax, compose, negate) across three levels (local, compositional, structural) | Pending |
| Phase 3 | Build self-verification module with syntactic, semantic, and constructive verification hierarchy | Pending |
| Phase 4 | Integrate CAQG reward function into verl training framework with novelty, solvability, and difficulty components | Pending |
| Phase 5 | Implement evaluation framework for generation quality, solving quality, and self-evolution metrics | Pending |

---

## 1. Formal Problem Statement

### 1.1 Core Thesis

Let $\mathcal{S}$ denote the space of mathematical problem solvers and $\mathcal{G}$ the space of problem generators. Current self-evolving systems (R-Zero, GRPO-based methods, etc.) optimize:

$$\pi^* = \arg\max_{\pi \in \mathcal{S}} \mathbb{E}_{q \sim \mathcal{D}, a \sim \pi(\cdot|q)}[R(a, a^*)]$$

where $R$ is a correctness reward. This conflates two fundamentally different cognitive operations:

- **Deductive Reasoning** (Solving): Given axioms $\mathcal{A}$ and problem $P$, derive solution $S$ such that $\mathcal{A} \vdash P \rightarrow S$
- **Abductive/Generative Reasoning** (Generation): Given mathematical structure $\mathcal{M}$, identify boundaries $\partial\mathcal{M}$ and constraints $\mathcal{C}$ that yield novel, well-posed problems

### 1.2 Mathematical Distinction

Define the **Constraint Manifold** $\mathcal{M}_\mathcal{C}$ as the space of valid mathematical problems under constraints $\mathcal{C}$:

$$\mathcal{M}_\mathcal{C} = \{P : \text{WellPosed}(P, \mathcal{C}) \land \text{Solvable}(P) \land \text{Novel}(P, \mathcal{D}_{train})\}$$

The key insight: **Problem generation requires navigation of $\partial\mathcal{M}_\mathcal{C}$ (boundary exploration), while solving requires traversal within $\mathcal{M}_\mathcal{C}$ (interior navigation).**

---

## 2. Theoretical Framework: Constraint Perturbation Theory

### 2.1 Constraint Algebra

For a mathematical problem $P$ with constraint set $\mathcal{C} = \{c_1, ..., c_n\}$, define perturbation operators:

- **Tightening**: $\tau_i: c_i \mapsto c_i'$ where $\text{dom}(c_i') \subset \text{dom}(c_i)$
- **Relaxation**: $\rho_i: c_i \mapsto c_i'$ where $\text{dom}(c_i') \supset \text{dom}(c_i)$
- **Composition**: $\kappa_{ij}: (c_i, c_j) \mapsto c_{ij}$ (constraint interaction)
- **Negation**: $\nu_i: c_i \mapsto \neg c_i$ (boundary inversion)

### 2.2 Novelty Metric

Define novelty as distance from the training manifold:

$$\mathcal{N}(P) = \min_{P' \in \mathcal{D}_{train}} d_{\text{semantic}}(P, P') \cdot \mathbb{I}[\text{Solvable}(P)]$$

where $d_{\text{semantic}}$ is computed via:

$$d_{\text{semantic}}(P_1, P_2) = 1 - \frac{\langle \phi(P_1), \phi(P_2) \rangle}{||\phi(P_1)|| \cdot ||\phi(P_2)||}$$

with $\phi$ being a concept-aware embedding (not just token similarity).

### 2.3 Difficulty Progression Function

Define the **Boundary Proximity Score**:

$$\beta(P, \mathcal{C}) = \min_{c \in \mathcal{C}} \frac{|\text{slack}(P, c)|}{\text{range}(c)}$$

Problems with low $\beta$ are near constraint boundaries and typically harder.

---

## 3. Proposed Architecture: Constraint-Aware Question Genesis (CAQG)

### 3.1 Three-Stage Pipeline

```
Stage 1: Constraint Extraction (CE)
   Input: Seed problem P_0
   Output: Explicit constraint set C = {c_1, ..., c_n}
   
Stage 2: Boundary Exploration (BE)
   Input: C, perturbation budget k
   Output: Candidate constraint sets {C'_1, ..., C'_m}
   
Stage 3: Problem Synthesis (PS)
   Input: C'_i
   Output: Novel problem P' with self-verification
```

### 3.2 Mathematical Verification Loop

Unlike R-Zero's majority voting, implement **constructive verification**:

1. Generate problem $P$ with claimed solution $S$
2. Verify: $\text{Verify}(P, S) = \mathbb{I}[\text{Forward}(P) = S] \land \mathbb{I}[\text{Backward}(S, P) \text{ is valid}]$
3. Uniqueness check: $\text{Unique}(P, S) = \mathbb{I}[|\{S' : \text{Solve}(P) = S'\}| = 1]$

---

## 4. Experimental Design

### 4.1 Hypothesis Formalization

- **H1 (Capability Separation)**: $\text{Corr}(\text{SolveScore}, \text{GenScore}) < \theta$ for some threshold $\theta$
- **H2 (Genesis Primacy)**: Training on $\mathcal{G}$ improves $\mathcal{S}$, but not vice versa beyond a ceiling
- **H3 (Entropy Preservation)**: CAQG maintains $H(\pi_t) > H_{min}$ throughout training, unlike GRPO

### 4.2 Experimental Conditions

| Condition | Training Signal | Question Source |
|-----------|-----------------|-----------------|
| Baseline (R-Zero) | Majority voting reward | Challenger model |
| GRPO-Solver | Correctness reward | Fixed dataset |
| CAQG-Full | Constraint perturbation + self-verify | Self-generated |
| CAQG-Ablation | Constraint perturbation only | Self-generated |

### 4.3 Datasets

- **Seed Problems**: MATH (Hendrycks), AMC/AIME, GSM8K (for calibration)
- **Evaluation**: Held-out competition problems, novel synthetic problems

### 4.4 Metrics

**For Generation Quality:**

- **Novelty Score**: $\mathcal{N}(P)$ as defined above
- **Diversity**: $\text{Div}(\mathcal{P}) = \frac{1}{|\mathcal{P}|^2} \sum_{P_i, P_j} d_{\text{semantic}}(P_i, P_j)$
- **Solvability Rate**: Fraction of generated problems that are verifiably solvable
- **Difficulty Calibration**: Correlation between predicted and actual solve rates

**For Solving Quality:**

- Standard accuracy on MATH, AIME, AMC
- Generalization gap: performance on in-distribution vs. out-of-distribution

**For Self-Evolution:**

- Entropy trajectory: $H(\pi_t)$ over training steps
- Capability ceiling: asymptotic performance
- Mode collapse detection: $\text{KL}(\pi_t || \pi_{t-k})$ stability

---

## 5. Implementation Roadmap

### Phase 1: Constraint Extraction Module (Weeks 1-2)

**Objective**: Build a system that extracts explicit mathematical constraints from problems.

**Key Files to Modify/Create**:

- Extend `question_generate/question_generate.py` with constraint parsing
- Create `constraint_extractor.py` with:
  - Domain identification (algebra, geometry, number theory, etc.)
  - Constraint type classification (equality, inequality, integrality, positivity, etc.)
  - Dependency graph construction

**Mathematical Formalization**:

For problem $P$, extract:
- $\mathcal{C}_{explicit}$: Stated constraints (e.g., "$x > 0$", "$n \in \mathbb{Z}$")
- $\mathcal{C}_{implicit}$: Assumed constraints (e.g., domain of discourse)
- $\mathcal{C}_{structural}$: Problem structure (e.g., "find all", "prove", "compute")

### Phase 2: Perturbation Engine (Weeks 3-4)

**Objective**: Implement the constraint perturbation operators.

**Perturbation Taxonomy**:

```
Level 1 (Local): Single constraint modification
  - Tighten: x > 0 → x > 1
  - Relax: x ∈ Z → x ∈ Q
  - Boundary: x ≤ n → x = n
  
Level 2 (Compositional): Multi-constraint interaction
  - Merge: {x > 0, y > 0} → xy > 0
  - Split: x² + y² = 1 → {|x| ≤ 1, |y| ≤ 1, x² + y² = 1}
  
Level 3 (Structural): Problem type transformation
  - Inversion: "Solve for x" → "For which k does a solution exist?"
  - Generalization: Specific → Parametric family
  - Specialization: General → Edge case
```

### Phase 3: Self-Verification Module (Weeks 5-6)

**Objective**: Implement verification that does not rely on external oracles.

**Verification Hierarchy**:

1. **Syntactic**: Well-formed mathematical statement
2. **Semantic**: Internally consistent constraints
3. **Solvability**: At least one solution exists
4. **Uniqueness**: Solution is determinate (or explicitly enumerable)
5. **Constructive**: Model can solve its own problem

**Key Insight**: Use the duality:

$$\text{GenVerify}(P, S) = \text{Solve}(P) \stackrel{?}{=} S \land \text{Generate}(S) \stackrel{?}{\sim} P$$

### Phase 4: Training Loop Integration (Weeks 7-8)

**Objective**: Integrate with the verl training framework.

**Modifications to `verl/trainer/core_algos.py`**:

Replace outcome-based advantage with **constraint-based intrinsic reward**:

$$R_{CAQG}(P) = \alpha \cdot \mathcal{N}(P) + \beta \cdot \text{Verify}(P) + \gamma \cdot \beta(P, \mathcal{C}) - \lambda \cdot \text{KL}(\pi || \pi_{ref})$$

where:
- $\alpha$: novelty weight
- $\beta$: solvability weight  
- $\gamma$: difficulty targeting weight
- $\lambda$: divergence penalty

### Phase 5: Evaluation Framework (Weeks 9-10)

**Objective**: Comprehensive evaluation of generation vs. solving capabilities.

**Evaluation Protocol**:

1. **Generation Evaluation**:
   - Human expert rating of novelty (blinded)
   - Automated novelty via embedding distance
   - Solvability rate by external solver
   - Difficulty prediction accuracy

2. **Solving Evaluation**:
   - Standard benchmarks (MATH, AIME)
   - Self-generated problems (cross-validation)
   - Out-of-distribution generalization

3. **Self-Evolution Metrics**:
   - Entropy trajectory analysis
   - Capability ceiling measurement
   - Mode collapse detection

---

## 6. Critical Design Decisions

### 6.1 Avoiding Reward Hacking

**Problem**: GRPO-style methods can exploit reward signals without genuine capability improvement.

**Solution**: Constraint-based intrinsic motivation:
- Reward novelty independent of external scoring
- Require self-verification (model must solve its own problems)
- Diversity bonus prevents mode collapse

### 6.2 Preventing Entropy Collapse

**Problem**: Self-evolution can lead to premature convergence to narrow problem distributions.

**Solution**: 
- Explicit entropy regularization: $\mathcal{L} = \mathcal{L}_{CAQG} - \eta H(\pi)$
- Constraint diversity requirement: sample from multiple perturbation levels
- Periodic "wildcard" generation: force exploration of distant constraint configurations

### 6.3 Ensuring Well-Posedness

**Problem**: Generated problems may be ill-defined, unsolvable, or trivial.

**Solution**: Three-tier filtering:

```
Tier 1: Syntactic validity (parseable, complete)
Tier 2: Semantic validity (consistent constraints)
Tier 3: Mathematical validity (solvable, non-trivial, unique)
```

---

## 7. Expected Outcomes and Contributions

### 7.1 Primary Contributions

1. **Theoretical**: Formal separation of generative vs. deductive mathematical reasoning
2. **Methodological**: Constraint Perturbation Theory for systematic novelty generation
3. **Empirical**: Evidence that generation capability is foundational to solving capability

### 7.2 Success Criteria

- **H1 Validation**: Demonstrate $\text{Corr}(\text{SolveScore}, \text{GenScore}) < 0.5$
- **H2 Validation**: Show asymmetric transfer (Gen → Solve > Solve → Gen)
- **H3 Validation**: Maintain $H(\pi_t) > 0.7 \cdot H(\pi_0)$ throughout training

### 7.3 Failure Modes and Mitigations

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| Trivial problems | Low solve variance | Difficulty targeting via $\beta(P, \mathcal{C})$ |
| Impossible problems | Zero solvability | Constructive verification |
| Copied problems | High similarity to train | Novelty metric threshold |
| Mode collapse | Entropy drop | Diversity bonus + restart |

---

## 8. Connection to Existing Codebase

The R-Zero implementation in `tree/euclid/expv0/R-Zero/` provides:

- **Reusable**: verl training infrastructure, vLLM integration, evaluation pipeline
- **Modify**: Question generation prompts (from open-ended to constraint-aware)
- **Replace**: Majority voting verification with constructive self-verification
- **Extend**: Reward computation to include novelty and constraint metrics

Key integration points:

- `question_generate/question_generate.py`: Add constraint extraction and perturbation
- `examples/reward_function/math.py`: Extend with novelty reward
- `verl/trainer/core_algos.py`: Add CAQG advantage computation

---

## 9. Related Work Positioning

| Paper | Focus | Our Differentiation |
|-------|-------|---------------------|
| R-Zero (2508.05004) | Challenger-Solver co-evolution | Replace reward-driven with constraint-driven |
| Qwen2.5-Math (2409.12122) | Self-improvement on solving | Focus on generation as primary capability |
| SEC (2505.14970) | Curriculum selection | Curriculum generation from scratch |
| Model Collapse (2305.17493) | Warns of recursive training | Novelty metric prevents collapse |
| Self-Refine (2303.17651) | Iterative output refinement | Constraint-aware generation, not just refinement |
| DeepSeekMath-V2 (2511.22570) | Self-verification | Extend to problem generation verification |
| SATURN (2505.16368) | SAT-based curriculum | Domain-agnostic constraint perturbation |
| Dr. Zero (2601.07055) | Self-evolving search | Constraint-driven vs. search-driven |

---

## 10. Appendix: Mathematical Proofs Sketch

### A.1 Capability Separation Theorem (Informal)

**Claim**: The information required to identify constraint boundaries ($\partial\mathcal{M}_\mathcal{C}$) is strictly greater than the information required to navigate the interior ($\mathcal{M}_\mathcal{C}$).

**Sketch**: Navigation requires local gradient information. Boundary identification requires global manifold topology. By the implicit function theorem, boundary points are singular, requiring second-order information not needed for interior navigation.

### A.2 Genesis Primacy Lemma (Informal)

**Claim**: A system capable of generating all problems in $\mathcal{M}_\mathcal{C}$ can solve all problems in $\mathcal{M}_\mathcal{C}$, but not vice versa.

**Sketch**: Generation requires: (1) constraint enumeration, (2) solution verification. Solving requires only (3) solution derivation. Since (2) $\supseteq$ (3) (verification implies derivation capability), generation subsumes solving. The converse fails: solving a specific problem does not require enumerating the constraint space.

---

## References

1. Huang et al. (2025). R-Zero: Self-Evolving Reasoning LLM from Zero Data. arXiv:2508.05004
2. Yang et al. (2024). Qwen2.5-Math Technical Report. arXiv:2409.12122
3. Shumailov et al. (2023). The Curse of Recursion: Training on Generated Data Makes Models Forget. arXiv:2305.17493
4. Madaan et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. arXiv:2303.17651
5. Chen et al. (2025). Self-Evolving Curriculum for LLM Reasoning. arXiv:2505.14970
6. SATURN (2025). SAT-based Reinforcement Learning. arXiv:2505.16368
7. DeepSeekMath-V2 (2025). Towards Self-Verifiable Mathematical Reasoning. arXiv:2511.22570
8. Dr. Zero (2026). Self-Evolving Search Agents without Training Data. arXiv:2601.07055
