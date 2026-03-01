### Research Proposal v0 — Method 1 (Offline): Intelligent Explorator + Teacher-Verified Synthesis + Student-Aware Curriculum for Mathematical Reasoning

This document specifies **Method 1** in full detail: we build a *better explorator* than GRIP’s largely random concept sampling, use a strong teacher/verifier to generate and validate synthetic math problems, apply **student-aware ZPD filtering**, and then fine-tune a base solver. No co-evolutionary RL loop is required in v0; the goal is a stable, identifiable experiment that isolates the value of a better explorator and better selection.

Primary references:
- GRIP: A Graph-Based Reasoning Instruction Producer ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864))
- R‑Zero: Self-Evolving Reasoning LLM from Zero Data ([`arXiv:2508.05004`](https://arxiv.org/pdf/2508.05004))
- R‑FEW: Guided Self-Evolving LLMs with Minimal Human Supervision ([`arXiv:2512.02472`](https://arxiv.org/pdf/2512.02472))

---

### 0) Executive summary
- **Problem**: Synthetic math data generation often collapses to low-entropy templates, duplicates, or “cheap difficulty” (verbosity/format), and does not reliably target what a specific base model needs to learn next.
- **Method 1**: Build a **concept-graph explorator** that proposes structured “idea specs” (concept bundles + relations), but make it *intelligent* by learning which bundles are cohesive and high-yield. Use a strong teacher to verify well-posedness + correctness, enforce semantic deduplication, and then apply **ZPD filtering** so the final dataset is both novel and learnable.
- **Deliverable**: A teacher-verified dataset whose **learning efficiency** (solver gain per accepted sample / per teacher-dollar) exceeds vanilla GRIP-style sampling, plus strong evidence that collapse is reduced (coverage metrics).
- **Why this matters**: It isolates the missing piece in GRIP: *student-aware, yield-aware exploration* rather than only large-scale concept recombination.

---

### 1) Problem definition (what exactly we are solving)
We want to train a base model \(S\) into a **better mathematical reasoner** using synthetic data, without relying on large human-labeled corpora.

We frame this as optimizing a pipeline that produces a training set \(D_{\text{train}}\) from a limited seed set \(D_{\text{seed}}\) and a limited teacher budget.

#### 1.1 Formal objects
- **Task/problem**: \(q \in \mathcal{Q}\) (a math question statement, with an intended answer type).
- **Solution**: \(a \in \mathcal{A}(q)\) (final answer + optionally a reasoning trace).
- **Teacher verifier**: \(T(q,a) \to \{\text{valid}, \text{invalid}\}\) plus metadata (difficulty estimate, ambiguity flags, equivalence checks).
- **Solver/base model**: \(S_\phi(a \mid q)\) (student we fine-tune).
- **Generator model**: \(G_\psi(q,a \mid \text{spec})\) (produces candidate problems + solutions from an explorator spec).

#### 1.2 Target property: “good synthetic math data”
An accepted synthetic sample \((q,a)\) must satisfy:
- **Validity / well-posedness**: the problem statement is unambiguous under the intended conventions; there is a coherent target answer type.
- **Correctness**: the provided solution \(a\) is correct (teacher-verified; symbolic checks when available).
- **Non-triviality**: avoids degenerate/trivial tasks that do not train reasoning (e.g., “compute 2+2”).
- **Novelty**: not a near-duplicate of previously accepted or seed tasks (semantic, not just lexical).
- **Learnability**: sits in the student’s **zone of proximal development** (ZPD): neither too easy nor impossible for the current solver.

We explicitly separate “novelty” from “difficulty.” A task can be novel but useless; we want **novel + learnable + valid**.

---

### 2) Motivation (why existing approaches are insufficient)
#### 2.1 Why prompt/seed-based generation collapses
Unconstrained prompting optimizes a single proxy (e.g., “be difficult”), which tends to produce:
- **Mode collapse** (repeated templates).
- **Proxy hacking** (length inflation masquerading as difficulty).
- **Hallucination drift** (invalid/ill-posed problems).

#### 2.2 What GRIP solves and what it misses
GRIP’s key idea is scalable novelty via **concept graph recombination**: extract key concepts, build a KCRG graph, and generate tasks from novel concept combinations ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864)).

What GRIP does *not* directly optimize:
- **Student-specific learning value**: GRIP synthesizes broadly; it does not explicitly target the current solver’s ZPD or maximize learning progress per sample.
- **Yield efficiency**: as you go to farther implicit hops, coherence drops; GRIP relies on downstream filtering rather than learning a policy that avoids waste.
- **Semantic equivalence novelty**: GRIP uses overlap analyses and concept novelty; equivalence under mathematical transformations is not fully addressed without a stronger semantic dedup stage.

#### 2.3 Why we do not start with R‑Zero/R‑FEW co-evolution
R‑Zero/R‑FEW co-evolution loops introduce non-stationarity and confounding: the proposer and solver improve together, making it hard to attribute gains to exploration vs curriculum vs verification ([`arXiv:2508.05004`](https://arxiv.org/pdf/2508.05004), [`arXiv:2512.02472`](https://arxiv.org/pdf/2512.02472)).

Method 1 is intentionally **offline and identifiable** first:
- Keep generator + teacher fixed.
- Vary only the explorator policy / selection policy.
- Measure downstream solver improvement per budget.

---

### 3) Core hypothesis and novelty claim (Method 1)
#### 3.1 Hypothesis (testable)
Given fixed seed set \(D_{\text{seed}}\), fixed generator \(G\), and fixed teacher \(T\), an **intelligent explorator** that selects concept bundles using yield/cohesion modeling + coverage constraints will produce a dataset \(D_{\text{train}}\) that yields:
- higher solver improvement \(\Delta\)score per accepted sample,
- higher solver improvement per teacher token/call,
- higher structural diversity/coverage,
compared to vanilla GRIP-style sampling.

#### 3.2 Novelty vs GRIP
Our contribution is not “more synthetic data.” It is **student-aware, yield-aware exploration and selection**, turning GRIP’s concept graph from a static combinator into a *policy-guided curriculum designer*:
- **Exploration policy**: choose what to generate (concept bundle selection).
- **Selection policy**: choose what to train on (teacher validity + semantic dedup + ZPD).

---

### 4) System overview (components)
We decompose the system into two front-end components (as agreed), plus the missing back-end gates that make it scientifically stable.

#### 4.1 Components
- **(A) Concept schema + extractor**: maps each problem to a set of normalized “concepts.”
- **(B) KCRG graph builder** (GRIP): builds explicit + implicit relationships over concepts.
- **(C) Explorator**: proposes structured specs (concept bundles + relations + target attributes).
- **(D) Question generator \(G\)**: converts specs into concrete (q, a, metadata).
- **(E) Teacher/verifier \(T\)**: checks well-posedness + correctness; provides difficulty/ambiguity labels.
- **(F) Semantic dedup gate**: rejects near-duplicates (embedding retrieval + teacher equivalence check).
- **(G) Student-aware ZPD gate**: keeps problems in a mid-difficulty band for the current solver \(S\).
- **(H) Solver training pipeline**: fine-tunes the base solver on the accepted dataset.
- **(I) Evaluation harness**: metrics and ablations.

#### 4.2 Flow diagram (ASCII)
Data + control flow:

Seed problems D_seed
   │
   ├─► Concept extraction + normalization ─► Concept vocabulary V
   │                                         │
   │                                         └─► Build KCRG graph G=(V,E)
   │
   └──────────────────────────────────────────────────────────────────────┐
                                                                          │
Explorator policy π (baseline GRIP sampling OR intelligent policy)        │
   │  samples spec s = (concept bundle S, hop/relation, target attrs)     │
   ▼                                                                      │
Question generator Gψ: s ─► (q_candidate, a_candidate, meta)              │
   │
   ├─► Teacher/verifier T: validity/correctness/difficulty/ambiguity
   │        │
   │        ├─reject invalid/ambiguous
   │        ▼
   ├─► Semantic dedup gate (retrieval + teacher equivalence check)
   │        ├─reject near-duplicates
   │        ▼
   ├─► Student-aware ZPD gate (run solver S0 on q; keep mid band)
   │        ├─reject too easy / impossible
   │        ▼
   └─► Accepted dataset D_train
            │
            └─► Fine-tune solver S0 → S1
                     │
                     └─► Evaluate (benchmarks + held-out concept-combos)

Explorator update loop (still Method 1):
- In the **baseline** condition, π is fixed (GRIP-style sampling).
- In the **intelligent** condition, we log outcomes per bundle \(S\) (validity / dedup / ZPD pass/fail + reasons), fit/update the bundle scorer \((\hat{p}_{\text{valid}}(S), \hat{u}(S))\), and update π **between batches** using a constrained bandit policy. This is “training the explorator,” but **not** via online policy-gradient RL.

Optional (still Method 1): repeat once or twice with an updated solver for ZPD scoring (S0→S1→S2), while keeping the explorator updates supervised/bandit-style (no co-evolution RL).

---

### 5) Data model and concept graph (GRIP-compatible, but explicit)
#### 5.1 What is a “concept”?
We require a concept schema that is:
- **Extractable** from text reliably (via LLM extractor + normalization).
- **Useful** for recombination (concepts meaningfully constrain problem generation).
- **Stable** across paraphrases (robust to wording).

Recommended concept categories (store as typed nodes):
- **Theorem/lemma**: e.g., Pythagorean theorem, AM-GM, Cauchy-Schwarz.
- **Technique/proof method**: induction, invariants, extremal principle, contradiction.
- **Object type**: triangle, polynomial, graph, group, matrix, random variable.
- **Operation**: factoring, modular arithmetic, integration by parts.
- **Constraint motif**: “integer solutions,” “monotonicity,” “convexity,” “symmetry.”

In practice we represent a concept as a normalized string + type:
- `type: "theorem" | "technique" | "object" | "operation" | "constraint"`
- `name: canonical label`

#### 5.2 Concept extraction and normalization pipeline
- **Extractor model**: a strong instruction model (GRIP used Qwen2.5-32B-Instruct; we can replicate or choose a similar high-accuracy model) ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864)).
- **Normalization**:
  - map synonyms to a canonical label (dictionary + embedding clustering),
  - enforce type constraints (e.g., “Pythagorean theorem” must be `theorem`),
  - deduplicate near-identical concept strings.

We also run “reverse concept extraction” on generated problems to verify adherence, as GRIP does ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864)).

#### 5.3 KCRG (Key Concepts Relationship Graph)
Let \(V\) be concepts. Construct:
- **Explicit edges** \(E_{\text{exp}}\): connect concepts co-occurring in the same seed example.
- **Implicit edges** \(E_{\text{imp}}\): connect concepts within 2-hop or 3-hop proximity (as in GRIP), or via graph-based similarity communities ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864)).

We store counts and strengths:
- co-occurrence count, PMI, edge weights.
- node frequency and hubness (degree).

This graph defines the *search space* for the explorator.

---

### 6) The explorator (baseline vs intelligent)
The explorator outputs a **spec** \(s\) that constrains the generator.

#### 6.1 Spec format
A spec is a structured tuple:
- **Concept bundle** \(S=\{c_1,\dots,c_k\}\), typically \(k \in \{3,4,5\}\).
- **Relation type**: explicit edge, 2-hop implicit, 3-hop implicit, community mix.
- **Target domain tag** (optional): algebra / geometry / combinatorics / etc.
- **Target answer type**: numeric / expression / proof / classification.
- **Target difficulty band**: ZPD target (e.g., expected solver success 30–70%).
- **Optional template/proof-style tag**: e.g., “use invariants,” “case split,” etc.

This keeps the explorator as a planner, not a text generator.

#### 6.2 Baseline explorator = GRIP-style sampling
Baseline policy \(\pi_{\text{GRIP}}\):
- Sample a node or small seed set.
- Expand along explicit/implicit edges to form \(S\).
- Optionally bias by graph hop class (explicit vs 2-hop vs 3-hop).
This is largely “random-but-structured” sampling as in GRIP ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864)).

#### 6.3 Intelligent explorator (our upgrade)
We replace uniform sampling with a policy that optimizes yield and curriculum value.

##### 6.3.1 Two predicted quantities
For a candidate bundle \(S\), learn predictors:
- **Cohesion / feasibility**:
  - \(p_{\text{valid}}(S) \approx \Pr[\text{generator + teacher can produce a well-posed, correct sample using all concepts in } S]\).
- **Instructional value (proxy)**:
  - \(u(S) \approx \mathbb{E}[\text{ZPD-accept} \mid S]\) or \(\mathbb{E}[\Delta\text{eval} \mid \text{train on samples from }S]\) (learning progress, if we can afford it).

##### 6.3.2 Features for bundle scoring (no black magic)
We want a scorer that is measurable and hard to game:
- **Graph features**: subgraph density of \(S\), average hop length, edge weights, hubness.
- **Concept type pattern**: mix of theorem/object/operation/constraint (avoid all-same).
- **Frequency features**: penalize bundles made of only the most common nodes (collapse risk).
- **Novelty/coverage features**: distance from previously used bundles; count of unseen pairs within \(S\).

##### 6.3.3 Learning the scorer
We collect bundle outcomes from generation attempts:
- Label \(y_{\text{valid}}=1\) if teacher accepts at least one sample for \(S\).
- Label \(y_{\text{zpd}}=1\) if at least one accepted sample also passes ZPD filter.

Train a lightweight model (logistic regression / gradient boosted trees / small neural net) to predict \(p_{\text{valid}}\) and \(u\).

##### 6.3.4 Selection policy: bandit + coverage constraints
We then choose bundles by a constrained policy:
- **Exploit**: sample high predicted \(u(S)\) to maximize useful data per teacher call.
- **Explore**: sample uncertain \(S\) to expand graph coverage (avoid premature collapse).
- **Coverage constraint**: enforce quotas over:
  - hop class (explicit vs 2-hop vs 3-hop),
  - domain tags,
  - concept types,
  - rarity bands (long-tail concepts get guaranteed probability mass).

This prevents the “intelligent” policy from degenerating into always picking safe, frequent bundles.

#### 6.4 Cold start (bootstrapping the explorator so it doesn’t generate nonsense)
Even though we say “intelligent explorator,” it is **not initialized from a blank slate over arbitrary concept bundles**. It is cold-started from the **seed-derived concept graph** and a staged sampling curriculum:

- **Stage 0 (high-cohesion warm start)**:
  - restrict to bundles that are directly grounded in seed examples:
    - sample \(S\) by taking 1 seed problem’s concept set (or a subset) and optionally swapping in **one** closely connected neighbor (explicit edge).
  - restrict bundle size to small \(k\) (e.g., 3–4).
  - require at least one `object` concept and at least one `operation/constraint` concept (avoid incoherent “all-theorem” bundles).

- **Stage 1 (learn feasibility fast)**:
  - run a modest number of generation attempts, log outcomes, and train \(\hat{p}_{\text{valid}}(S)\) to separate “cohesive” vs “garbage” bundles.
  - only after \(\hat{p}_{\text{valid}}\) becomes calibrated do we expand the search space.

- **Stage 2 (controlled expansion to implicit hops)**:
  - gradually introduce 2-hop and then 3-hop bundles, but with hard constraints such as:
    - minimum induced subgraph density / minimum total edge weight in the bundle,
    - “bridge” requirement: at least one node connects to ≥2 others within the bundle,
    - cap on the number of rare/long-tail nodes per bundle (to avoid incoherent leaps).

- **Stage 3 (coverage-enforced exploration)**:
  - once validity/yield is stable, increase exploration mass into under-covered concepts/hop classes using quotas, while keeping a safety fraction in Stage 0/1 distributions so yield doesn’t collapse.

This staged approach ensures the explorator is “from scratch” only in the sense that it has no pretrained policy—**it is still anchored to a seed-induced concept universe and grows outward safely**, using teacher feedback to learn which combinations are viable.

---

### 7) Question generator \(G_\psi\) (fixed in Exp v0)
To isolate the explorator, we keep \(G\) fixed across conditions.

#### 7.1 Required outputs
Given spec \(s\), generator outputs:
- **Problem statement** \(q\) (clear, complete).
- **Solution** \(a\) (final answer + optionally a reasoning trace).
- **Metadata**:
  - extracted concepts (generator’s own attempt; later cross-checked),
  - intended answer format,
  - any assumptions used.

#### 7.2 Generator model choice
Two reasonable regimes:
- **Open-source generator**: e.g., Qwen2.5-32B-Instruct or similar, for cost/scale.
- **Teacher-as-generator** (stronger but expensive): teacher generates and verifies in one loop.

For a clean experiment: pick one generator and keep it constant.

#### 7.3 Prompting protocol (spec-driven, not seed-driven)
We avoid feeding seed questions. The generator prompt contains:
- the concept bundle \(S\) and relation type,
- constraints on incorporating all concepts non-trivially,
- output format with explicit final answer,
- instruction to include minimal ambiguity.

---

### 8) Teacher/verifier \(T\) (strong oracle; central to validity)
We assume access to a strong teacher model (closed or open).

#### 8.1 Teacher tasks (must be explicit)
For each candidate \((q,a)\), teacher returns:
- **Well-posedness**: pass/fail + reason (missing constraints, ambiguous definitions, multiple answers, ill-defined).
- **Correctness**: pass/fail + (optionally) corrected final answer.
- **Difficulty estimate**: ordinal or calibrated score (e.g., 1–5) and/or predicted solver success band.
- **Ambiguity flags**: units, domain assumptions, reliance on unstated conventions.

For equivalence checks, teacher also answers:
- “Is this problem essentially equivalent to one of these k retrieved prior problems?”

R‑FEW explicitly uses a judge-style LLM evaluation for math equivalence checks ([`arXiv:2512.02472`](https://arxiv.org/pdf/2512.02472)); we adopt the same spirit but expand to well-posedness and semantic dedup.

#### 8.2 Multi-teacher option (robustness)
To reduce single-judge bias:
- use teacher ensemble voting for borderline cases,
- or use one strong teacher + one symbolic checker when applicable (sympy, numeric checkers) for final answers.

---

### 9) Semantic deduplication and novelty gating
This is required to prevent “novelty by rewording.”

#### 9.1 Layered novelty gates (from our discussion)
- **Gate A (lexical)**: n-gram overlap / fuzzy matching against seed + accepted pool (fast).
- **Gate B (embedding retrieval)**: nearest-neighbor search over a sentence embedding of the problem; reject if similarity exceeds threshold.
- **Gate C (teacher equivalence)**: for the top-k nearest neighbors, teacher judges equivalence up to trivial transformations (rename variables, reorder terms).

We track novelty statistics at each gate.

#### 9.2 Concept-combination novelty (GRIP-style)
We also track:
- unseen concept pairs/triples within \(S\),
- hop distance novelty (explicit vs implicit),
as GRIP’s novelty driver ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864)).

---

### 10) Student-aware ZPD filtering (the key missing ingredient vs GRIP)
We add a *student-specific* learnability constraint.

#### 10.1 Estimating difficulty for the current solver
Given candidate problem \(q\), estimate student success probability:
- sample multiple solver rollouts \(a_1,\dots,a_m \sim S_\phi(\cdot\mid q)\),
- score correctness via teacher/judge (or exact match where possible),
- define \(\hat{p}_{\text{succ}}(q)=\frac{1}{m}\sum \mathbf{1}[\text{correct}]\).

#### 10.2 ZPD acceptance band
Accept if:
- \(p_{\min} \le \hat{p}_{\text{succ}}(q) \le p_{\max}\),
with typical values \(p_{\min}=0.3\), \(p_{\max}=0.7\) (tunable).

R‑FEW uses a mid-uncertainty curriculum principle for stable self-evolution ([`arXiv:2512.02472`](https://arxiv.org/pdf/2512.02472)); we apply it offline as a *filter*.

#### 10.3 Why this prevents collapse
Without ZPD, generation can drift into:
- trivial tasks (high acceptance but low learning),
- impossible tasks (high novelty but no learning),
- or proxy-hacked difficulty.
ZPD forces a “productive frontier” for the student.

---

### 11) Solver training (offline fine-tuning)
#### 11.1 Training data formats
Each accepted sample can be stored as:
- **(q, final answer)** for strict answer supervision,
- **(q, reasoning trace, final answer)** for process supervision (if teacher accepts trace),
- **(q, multiple solutions)** if we want diversity in reasoning.

#### 11.2 Training objective (loss)
Primary: supervised fine-tuning (SFT) with token-level cross-entropy:
\[
\mathcal{L}_{\text{SFT}}(\phi) = -\mathbb{E}_{(q,a)\sim D_{\text{train}}}\left[\log S_\phi(a\mid q)\right].
\]

Optional additions (if we store structured fields):
- **Format loss**: ensure final answer formatting is consistent (R‑FEW uses explicit format correctness weighting in training curves) ([`arXiv:2512.02472`](https://arxiv.org/pdf/2512.02472)).
- **Answer-only vs CoT weighting**: downweight chain-of-thought tokens if we fear overfitting style; keep final answer tokens higher weight.

We do **not** start with RL for Method 1 v0, to keep stability and interpretability.

#### 11.3 Model choices
We choose one base solver for Exp v0 and hold it fixed across ablations. Examples:
- Qwen-family 7B–8B base model (R‑FEW reports results with Qwen3-8B-Base) ([`arXiv:2512.02472`](https://arxiv.org/pdf/2512.02472)).
- LLaMA-family 8B as an alternative.

The proposal is model-agnostic; what matters is consistent use across conditions.

---

### 12) Experimental design (Exp v0 / v1)
We design the experiment to isolate the effect of the explorator + selection, with tight controls.

#### 12.1 Fixed resources across conditions
For fair comparison, fix:
- seed set \(D_{\text{seed}}\),
- generator \(G\),
- teacher \(T\),
- total teacher calls / tokens,
- total number of candidate generations,
- solver training compute (steps, batch size).

#### 12.2 Conditions (the main ablation grid)
- **C0 (baseline)**: GRIP-style sampling \(\pi_{\text{GRIP}}\) + teacher validity + simple dedup (lexical).
- **C1**: C0 + semantic dedup (embedding + teacher equivalence).
- **C2**: C1 + ZPD filter (student-aware learnability).
- **C3 (full method)**: Intelligent explorator (yield/cohesion scoring + coverage constraints) + semantic dedup + ZPD.

This isolates which ingredient gives what gain.

#### 12.3 Optional iteration (still Method 1)
We can do 1–2 rounds:
- Round 0: build dataset using solver \(S_0\) for ZPD scoring; train to \(S_1\).
- Round 1: regenerate a new batch; ZPD-score with \(S_1\); train to \(S_2\).

Crucially: we do **not** update the explorator via policy-gradient RL; only via supervised yield modeling and constrained sampling.

---

### 13) Evaluation (metrics that cannot be gamed)
We evaluate both the **dataset** and the **trained solver**.

#### 13.1 Dataset quality metrics
- **Validity rate**: fraction of generated candidates accepted as well-posed by teacher.
- **Correctness rate**: fraction with teacher-verified correct solutions.
- **Ambiguity rate**: fraction flagged as ambiguous (should be low).
- **Yield per budget**: accepted samples per teacher-call and per teacher-token.

#### 13.2 Novelty/diversity metrics
We measure novelty at multiple layers:
- **Lexical novelty**: n-gram overlap vs seed and benchmark (GRIP-style decontamination) ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864)).
- **Embedding novelty**: nearest-neighbor similarity distribution.
- **Semantic novelty**: teacher equivalence rejection rate.
- **Concept novelty**:
  - unseen concept pairs/triples rate,
  - coverage entropy over concept nodes,
  - coverage entropy over concept-type patterns,
  - hop-class distribution (explicit vs 2-hop vs 3-hop).

Entropy collapse detection:
- Track coverage entropy over time/batches; collapse = entropy decreasing sharply while yield remains high.

#### 13.3 Learnability / curriculum metrics
- **ZPD pass rate**: fraction of teacher-accepted samples that pass ZPD band for \(S_0\).
- **Difficulty histogram**: distribution of \(\hat{p}_{\text{succ}}\) across accepted samples.
- **Learning progress proxy**: improvement on a held-out dev set after training on each batch (if feasible).

#### 13.4 Solver performance metrics
Evaluate \(S_0\) vs \(S_1\) (and optionally \(S_2\)) on:
- **Standard math reasoning benchmarks** (choose a consistent suite; ensure decontamination).
- **Held-out concept-combination test**:
  - tag benchmark items with concept extractor,
  - hold out specific concept pairs/triples during synthesis,
  - test generalization on those held-out combinations.
- **Robustness slices**:
  - by domain (algebra/combinatorics/geometry),
  - by proof/solution style tags (if we annotate).

Core scalar we care about:
- **Gain per accepted sample**: \(\Delta\)benchmark / |D_train|.
- **Gain per teacher budget**: \(\Delta\)benchmark / teacher_tokens.

---

### 14) Threats to validity + mitigations
- **Teacher bias / teacher leakage**:
  - mitigation: keep teacher fixed across conditions; optionally use multiple teachers for robustness; ensure teacher is only verifier, not seed provider.
- **Concept extractor noise**:
  - mitigation: normalization, human spot checks, reverse extraction adherence checks (GRIP-style) ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864)).
- **Novelty metric gaming**:
  - mitigation: semantic dedup with teacher equivalence checks, not just lexical metrics.
- **Benchmark contamination**:
  - mitigation: decontamination analysis (n-gram overlap) as GRIP reports; hold out concept combinations explicitly ([`arXiv:2412.08864`](https://arxiv.org/pdf/2412.08864)).
- **Compute confounds**:
  - mitigation: matched compute and teacher budget across conditions.

---

### 15) Expected outcomes and go/no-go criteria
We proceed to Method 2 (online co-evolution) only if Method 1 demonstrates:
- **(i) Higher yield**: more accepted + ZPD-eligible samples per teacher budget than baseline GRIP sampling.
- **(ii) Higher learning efficiency**: larger solver gains per accepted sample and per teacher-token.
- **(iii) Less collapse**: sustained or increasing coverage entropy over concept space across batches.
- **(iv) Real generalization**: improvements on held-out concept-combination tests, not only on in-distribution slices.

If Method 1 fails, the failure will usually diagnose into:
- concept schema too noisy/coarse,
- teacher verification too weak/inconsistent,
- generator unable to realize complex bundles coherently,
- ZPD band miscalibrated.
Those are actionable before attempting Method 2.

---

### 16) Concrete “components needed” checklist (implementation-facing, but no code)
- **Seed corpus**: a small set of verified math problems + solutions (e.g., MATH training subset).
- **Concept extractor**: strong instruction model + normalization rules.
- **Graph store**: KCRG with edge weights, hop neighborhoods, coverage counters.
- **Explorator**:
  - baseline sampler (GRIP-style),
  - intelligent scorer (validity/yield predictors),
  - constrained bandit sampler (coverage quotas).
- **Question generator**: fixed LLM + spec-driven prompt.
- **Teacher/verifier**:
  - well-posedness rubric,
  - correctness judge,
  - difficulty scoring,
  - equivalence judge for dedup.
- **Dedup system**: embedding index + retrieval + teacher equivalence confirmation.
- **ZPD evaluator**: multi-rollout student scoring + judge.
- **Training harness**: SFT fine-tuning + evaluation suite + decontamination tooling.

---

### 17) Summary of what is “new” in this experiment vs GRIP
Method 1 is explicitly about:
- turning concept recombination into a **policy** (intelligent explorator),
- producing data that is **student-optimal** (ZPD),
- enforcing **semantic novelty** (teacher equivalence dedup),
and measuring success as **learning efficiency per budget**, not dataset size.

