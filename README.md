# DLS-SUC: PyTorch model for lysine succinylation (Ksucc) site prediction

DLS-SUC is a deep learning model and inference codebase for predicting lysine succinylation (Ksucc) sites on protein sequences.
This repository provides a minimal pipeline to load features, run single/ensemble inference, and export metrics.

---

## Requirements

It is recommended to use a clean Conda/venv environment.

- torch 2.0.0+cu118 (or a version that matches your CUDA)
- pytorch-lightning 2.0.8
- pytorch-metric-learning 1.7.3
- pandas 2.1.1
- numpy >= 1.23
- pyyaml
- scikit-learn
- multimethod 1.9.1

Install:

```bash
pip install -r requirements.txt
```

---
## Quick Start with `testcode.py`

1. Environment: set up dependencies as above.
2. Data & features: place the provided CSV/FASTA/feature files under `data/suc/` (default paths expected by the code).
3. Checkpoints: put your trained checkpoints under `ckpt/dls-suc/` (multiple `.ckpt` files enable soft-voting ensemble).
4. Optional config: edit `config.yaml` (e.g., `name`, `monitor`, `mode`).

Notes:

- `testcode.py` automatically collects checkpoints in `ckpt/dls-suc/`, runs inference according to a selection strategy, and saves results to `result/`.
- To change strategy/ensemble size, edit the call at the end of `testcode.py`:

Outputs:

- Predictions CSV: `result/<name>result/soft_from_ckpts_ensemble_result.csv`
- Metrics CSV: `result/<name>result/soft_from_ckpts_metrics.csv`

Troubleshooting:
- Checkpoints not found: ensure `.ckpt` files exist under `ckpt/dls-suc/`.
- Environment: install a PyTorch build matching your CUDA; after renaming files, clear project `__pycache__/` to avoid stale bytecode.
---
