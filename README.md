# Amazon ML Challenge 2026

Competition workspace for the Amazon ML Challenge 2026. Treat the problem, data, target, metric, and constraints as unknown until Amazon releases the official materials; this framework does not assume the 2025 task.

```text
Problem understanding
        ↓
EDA
        ↓
Validation design
        ↓
Fast baseline
        ↓
Error analysis
        ↓
Targeted experiments
        ↓
Strong models
        ↓
Ensemble
        ↓
Submission validation
```

## First 6 Hours

### Hour 0–1

Understand the target, official metric, train/test schema, prediction unit, constraints, and required submission format. Choose validation to reflect the problem structure; do not blindly use random KFold.

### Hour 1–3

Build the fastest reasonable baseline and establish a trustworthy validation score.

### Hour 3–6

Investigate missingness, duplicates, leakage, train/test shift, target distribution, feature importance, subgroup errors, and alternative feature representations.

## 72-hour strategy

**Day 1:** Explore multiple approaches, establish reliable CV, and identify where the signal comes from.

**Day 2:** Stop broad exploration, optimize the strongest approaches, perform error analysis, and improve features.

**Day 3:** Ensemble complementary models, verify inference, generate the final submission, and prepare the approach document.

## Quick start

1. Put the released files in `data/` and edit `configs/baseline.yaml` after reviewing the official task.
2. Start the first-look notebook with `python -m notebook notebooks/00_first_look.ipynb` (or `jupyter notebook notebooks/00_first_look.ipynb`).
3. Choose a validation strategy that matches the data, set the target and submission columns, and implement `competition_metric` in `src/metrics.py`.
4. Run the baseline with `python -m src.train --config configs/baseline.yaml`; generate predictions with `python -m src.inference --config configs/baseline.yaml`.

The starter trainer covers ordinary tabular regression and classification with LightGBM. Adapt validation, metric, features, and model once the official problem is known. Text, image, and multimodal pipelines are extension points, not prebuilt assumptions.
