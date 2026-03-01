### ExpV1_0 — First runnable Method‑1 experiment (offline): GRIP‑style exploration + Teacher verification + Dedup + ZPD → SFT → Eval

This file specifies the **first experiment we will actually run** in ExpV1 (Method 1). It is intentionally designed to be:
- **Identifiable**: improvements are attributable to exploration/selection, not co-evolution RL dynamics.
- **Cold-startable**: exploration starts from seed-induced structure (not arbitrary random bundles).
- **Budgeted**: clear limits on teacher calls and generation volume.

Primary references:
- GRIP ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864))
- R‑FEW (ZPD/curriculum principle) ([`arXiv:2512.02472`](https://arxiv.org/pdf/2512.02472))

---

### 1) Objective
Demonstrate that a **student-aware selection layer (ZPD)** + **semantic/embedding dedup** on top of GRIP-style concept sampling yields a synthetic dataset that:
- has **higher learning efficiency** (solver gain per accepted sample / per teacher budget),
- has **lower collapse** (higher coverage entropy / lower near-duplicate rate),
than a naive “generate+filter” baseline.

This experiment is primarily a **pipeline validation** and **baseline establishment** for Method 1. It does not yet require online co-training or policy-gradient RL for the explorator.

---

### 2) What is “missing in GRIP” that we test here
GRIP’s concept-graph recombination produces scale and novelty, but does not explicitly optimize:
- **student learnability** (ZPD) for a particular base solver,
- **semantic uniqueness** beyond overlap + concept-combination novelty.

ExpV1_0 adds those two missing pieces and measures whether they improve downstream solver training efficiency.

---

### 3) Setup (fixed across conditions)
#### 3.1 Seed set
- \(D_{seed}\): a small, diverse subset of math problems with verified answers (e.g., MATH subset).  
- We use seeds only to build the **concept universe** and **concept graph**, not as examples in generation prompts.

#### 3.2 Models (roles)
- **Concept extractor**: LLM used to extract typed key concepts from seed problems (and reverse-check generated problems).
- **Explorator**: algorithmic policy sampling concept bundles \(S\) from the concept graph.
- **Question generator** \(G\): LLM that takes spec \(s=(S,\text{hop},\text{attrs})\) and outputs \((q,a,\text{meta})\).
- **Teacher/verifier** \(T\): strong model that checks well-posedness + correctness + (optional) equivalence.
- **Solver/student** \(S_0\): base model to be improved via offline SFT.

Notes:
- In the first run, generator and teacher can be the same model (teacher-as-generator) for simplicity, but we log roles separately.

#### 3.3 Budgets
We fix:
- total candidate specs \(N_{spec}\),
- total generator calls \(N_{gen}\),
- total teacher calls \(N_{teach}\),
- total accepted samples \(|D_{train}|\) target,
- solver training compute (steps/epochs).

---

### 4) Cold start strategy (so explorator doesn’t produce garbage)
We do **not** start from uniform random bundles over concepts.

Staged warm start:
- **Stage 0**: sample bundles from seed co-occurrence (explicit edges / one-seed concept set), optionally swap in **one** explicit neighbor; small bundle size (k=3–4).
- Only after we have a stable acceptance rate do we expand to:
  - **Stage 1**: 2-hop implicit combos,
  - **Stage 2**: 3-hop implicit combos (with cohesion constraints).

---

### 5) End-to-end pipeline (single batch)
1) **Concept extraction + normalization** on \(D_{seed}\) → concept vocabulary \(V\)
2) **Build KCRG concept graph** \(G=(V,E)\) with explicit + implicit edges
3) **Explorator** samples specs \(s_i\) (bundle + hop + target attrs)
4) **Generator** produces candidates \((q_i,a_i,\text{meta}_i)\)
5) **Teacher validity/correctness** gate:
   - reject ill-posed / incorrect
6) **Dedup / novelty gate**:
   - lexical + embedding near-neighbor rejection
   - (optional) teacher equivalence check for top-k neighbors
7) **ZPD gate** (student-aware):
   - run solver \(S_0\) multiple rollouts on \(q\)
   - grade via teacher or symbolic checker
   - accept if \(p_{min} \le \hat{p}_{succ}(q) \le p_{max}\)
8) Accepted items form \(D_{train}\)
9) **Offline SFT**: train \(S_0 \to S_1\) on \(D_{train}\)
10) **Evaluation**: benchmarks + held-out concept-combination slices; compute learning efficiency metrics

---

### 6) Conditions (minimal ablation for ExpV1_0)
We run two conditions with matched budgets:

- **C0 (GRIP-like baseline)**:
  - GRIP-style sampling (Stage 0/1)
  - teacher validity/correctness
  - lexical dedup only
  - no ZPD filter

- **C2 (Method‑1 core upgrade)**:
  - same sampling
  - teacher validity/correctness
  - lexical + embedding dedup (+ optional teacher equivalence)
  - ZPD filter using \(S_0\)

This isolates the effect of **semantic novelty gating + student-aware learnability**.

---

### 7) Metrics (must be reported)
#### 7.1 Dataset metrics
- teacher acceptance rate (well-posed + correct)
- dedup rejection rate (lexical vs embedding vs equivalence)
- ZPD pass rate
- yield per teacher budget (accepted / teacher_calls, accepted / teacher_tokens)

#### 7.2 Diversity metrics
- concept coverage entropy over nodes/types
- unseen concept-pair/triple rate (within accepted bundles)
- embedding NN distance distribution within dataset

#### 7.3 Solver metrics
- benchmark accuracy (before vs after SFT): \(S_0\) vs \(S_1\)
- held-out concept-combination generalization
- learning efficiency: \(\Delta\)score per accepted sample and per teacher token

---

### 8) Success criteria (go/no-go)
We consider ExpV1_0 a success if (C2 vs C0):
- increases solver improvement per accepted sample **or** per teacher budget,
- reduces near-duplicate rate and increases coverage entropy,
- does not degrade benchmark performance (no drift).

If successful, ExpV1_1 will add the **intelligent explorator** (yield/cohesion scorer + constrained bandit sampling) as the next axis.

