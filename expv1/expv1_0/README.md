### expv1_0 — Method 1 (offline) pipeline

This folder contains the **first runnable implementation** of the Method‑1 experiment described in:
- `tree/euclid/expv1/expv1_formal_planner/expv1_0.md`
- `tree/euclid/expv1/expv1_formal_planner/research_proposal_v0.md`

Goal (ExpV1_0):
- Build a synthetic dataset using **GRIP-style concept graph exploration**
- Keep **concept extraction + graph building** separate from **question generation + solver filtering**
- Add **dedup** and **solver-aware ZPD/self-consistency filtering**

This implementation is intentionally modular:
- Concept extraction, generator, teacher, solver inference, and dedup can be swapped between backends (local HF, vLLM, API).

#### Quickstart (high level)
1) Configure models and budgets in `config.yaml`

2) Run **Pipeline A** (concept extraction + KCRG artifacts):

```bash
python -m tree.euclid.expv1.expv1_0.run_build_concepts --config tree/euclid/expv1/expv1_0/config.yaml
```

3) Run **Pipeline B** (generation + solver filtering):

```bash
python -m tree.euclid.expv1.expv1_0.run_generate_dataset --config tree/euclid/expv1/expv1_0/config.yaml
```

4) Or run **end-to-end** (A then B):

```bash
python -m tree.euclid.expv1.expv1_0.run_build_dataset --config tree/euclid/expv1/expv1_0/config.yaml
```

5) Run SFT training (optional; heavy):

```bash
python -m tree.euclid.expv1.expv1_0.run_sft --config tree/euclid/expv1/expv1_0/config.yaml
```

6) Run evaluation:

```bash
python -m tree.euclid.expv1.expv1_0.run_eval --config tree/euclid/expv1/expv1_0/config.yaml
```

#### Notes
- This repo already contains an `expv0/exp0_0` baseline and the `R-Zero/` codebase. ExpV1_0 does **not** run co-evolution RL; it focuses on offline dataset construction + SFT for identifiability.

