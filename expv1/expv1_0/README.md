### expv1_0 — Method 1 (offline) pipeline

This folder contains the **first runnable implementation** of the Method‑1 experiment described in:
- `tree/euclid/expv1/expv1_formal_planner/expv1_0.md`
- `tree/euclid/expv1/expv1_formal_planner/research_proposal_v0.md`

Goal (ExpV1_0):
- Build a **teacher-verified** synthetic dataset using **GRIP-style concept graph exploration**
- Add **semantic/embedding dedup** and **student-aware ZPD filtering**
- Fine-tune a base solver (offline SFT) and evaluate learning efficiency.

This implementation is intentionally modular:
- Concept extraction, generator, teacher, solver inference, and dedup can be swapped between backends (local HF, vLLM, API).

#### Quickstart (high level)
1) Configure models and budgets in `config.yaml`
2) Run dataset build:

```bash
python -m tree.euclid.expv1.expv1_0.run_build_dataset --config tree/euclid/expv1/expv1_0/config.yaml
```

3) Run SFT training (optional; heavy):

```bash
python -m tree.euclid.expv1.expv1_0.run_sft --config tree/euclid/expv1/expv1_0/config.yaml
```

4) Run evaluation:

```bash
python -m tree.euclid.expv1.expv1_0.run_eval --config tree/euclid/expv1/expv1_0/config.yaml
```

#### Notes
- This repo already contains an `expv0/exp0_0` baseline and the `R-Zero/` codebase. ExpV1_0 does **not** run co-evolution RL; it focuses on offline dataset construction + SFT for identifiability.

