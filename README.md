# Amazon ML Challenge 2026 — Business Entity Resolution

This repository is a workspace for matching business records across three independent sources. Source 1 is the reference set; for every Source 1 record, predict zero or more matching Source 2 and Source 3 IDs.

## Challenge data

The released data is already present at:

```text
6ab10eb3b23ba_student_resource/student_resource/dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are TSV and must be read with a tab delimiter. The sources have `entity_id`, `business_name`, `business_address`, and `country`; labels map each Source 1 ID to a comma-separated set of Source 2/3 IDs. Training countries are US and India; test also includes France, so country handling must remain open-set.

## Project architecture

The generic tabular starter has been superseded by an entity-resolution pipeline. The intended flow is:

```text
TSV data + labels
       ↓
Load, validate schema, normalize text (retain raw fields)
       ↓
Candidate generation across S1×S2 and S1×S3
       ↓
Pairwise name/address/country feature construction
       ↓
Pair scorer + calibrated decision policy
       ↓
Per-S1 predictions and candidate audit files
       ↓
Macro F0.5 validation + submission format validator
```

The candidate set is the exact final set passed to the scorer and must include every emitted match. The matcher predicts links independently; it must allow multiple S2/S3 matches per S1 and empty predictions for singletons. Keep country comparisons as string equality/features rather than a closed categorical vocabulary.

See [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the staged implementation and validation plan. Current reusable pieces include the macro-F0.5 evaluator in `src/metrics.py`; `src/train.py` and `src/inference.py` are still generic tabular starters and are not yet the ER pipeline.

## Outputs and validation

Final artifacts go in `outputs/` (or the required submission package's `output/`):

- `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`
- `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`

Run the supplied validator from the extracted resource directory:

```powershell
python utils/validate_submission.py `
  --matching ..\..\outputs\matching_results.tsv `
  --candidate ..\..\outputs\candidate_pairs.tsv `
  --test-dir dataset\test
```

Use a validation split over Source 1 entities, construct train/validation candidate pairs without using held-out labels during fitting, and tune thresholds against macro F0.5. Do not use external identity lookup, geocoding, or business data; the pipeline must rely only on the provided challenge files.

## Repository map

```text
configs/       ER data paths and experiment settings
data/          optional local working data; official supplied data stays in the extracted resource folder
docs/          architecture and implementation plan
experiments/   experiment log
models/        locally trained artifacts
notebooks/     exploratory analysis
outputs/       generated prediction and candidate files
src/           implementation modules
submissions/   submission hygiene notes
```
