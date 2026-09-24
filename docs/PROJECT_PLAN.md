# Entity Resolution Architecture and Implementation Plan

## Objective and constraints

For each Source 1 business, recover all matching records in Sources 2 and 3. A source 1 entity may have no matches or several matches. The score is the macro-average of per-entity F0.5, where correctly predicting an empty set for a singleton scores 1.0. The solution must emit one row per test Source 1 entity in both output TSVs, and every predicted match must be present in that row's candidate set.

Only the supplied training and test data may be used. Do not query external data sources or APIs. Test country includes France, unseen during training; country must be handled as an open-set string field.

## Current repository assessment

- Official data is available in `6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}`.
- Each of six source files is a small TSV. Ground truth uses comma-separated match IDs.
- `src/metrics.py` already implements macro F0.5 and parses ID-list labels.
- `src/train.py`, `src/inference.py`, `src/data.py`, and `configs/baseline.yaml` assume a single ordinary CSV classification/regression table. They do not represent multiple sources, candidate pairs, multi-match labels, or the required output format.
- The supplied `utils/validate_submission.py` checks output structure and IDs. It does not measure predictive quality.
- The actual data sizes and distributions still need to be captured during EDA; the Windows session does not currently expose `python` on PATH, so no runtime-based statistics were available during this architecture pass.

## Target architecture

Implement the ER pipeline as small modules under `src/`:

| Module | Responsibility |
| --- | --- |
| `data.py` | Read TSVs explicitly, validate required columns/unique IDs, retain source identity, load labels and form per-S1 target sets. |
| `normalize.py` | Unicode and punctuation normalization, case folding, whitespace cleanup, token views, digit/postcode extraction; preserve raw fields and avoid country-specific assumptions that fail for France. |
| `blocking.py` | Generate the union of candidates using complementary name and address retrieval rules, separately for S1×S2 and S1×S3; output deterministic candidate IDs. |
| `features.py` | Produce pair features: character/token similarity, token overlap, edit ratios, exact/near-exact normalized fields, digit/postcode overlap, missingness and country agreement. |
| `train.py` | Create labeled candidate pairs from training data, add hard negatives, fit pairwise binary scorer, save model and feature metadata. |
| `validation.py` | Split by Source 1 entity, measure blocking recall, score final ID sets with macro F0.5, and select decision thresholds using validation only. |
| `inference.py` | Run blocking and scoring for test S1 against S2/S3, apply the decision policy, and write both required TSVs. |
| `metrics.py` | Keep the existing official macro-F0.5 implementation as the shared evaluation contract. |
| `submission.py` | Enforce row coverage, ID membership, uniqueness, and matched-subset-of-candidates invariants before calling the supplied validator. |

Pipeline order: load → validate → normalize → block → pair-feature extraction → fit/score → per-entity thresholding → output → validate. Store intermediate artifacts only when useful for reproducibility; candidate-pair output is the final scorer input, not an earlier broad block.

## Staged work plan

### 1. Data audit and baseline definition

- Inspect record counts per source, country distribution, missing fields, duplicate normalized names/addresses, label cardinality, and singleton rate.
- Verify each ground-truth ID exists in the matching source and quantify how often entities have multiple matches in one or both sources.
- Establish deterministic Source-1 holdout validation. Keep all S2/S3 rows available as the candidate universe, but do not fit thresholds or model parameters using held-out S1 labels.
- Record findings in an EDA note and log every run in `experiments/README.md`.

### 2. Build a high-recall candidate generator

- Start with normalized exact name/address keys and token or character n-gram retrieval.
- Add complementary retrieval paths such as rare name tokens, address digits/postcodes, and approximate TF-IDF similarity; union and deduplicate them.
- Apply country agreement as a retrieval preference or model feature, not an exclusion gate; retain safe fallback candidates for missing or inconsistent country values.
- Measure candidate recall separately for S2 and S3, plus reduction ratio and candidate count distribution. Tune blocking toward high recall while keeping pair scoring tractable.

### 3. Train and calibrate pair scoring

- Create positives from ground-truth links that survive blocking. Sample negatives from candidates, emphasizing hard negatives with similar names or addresses.
- Engineer name, address, numeric-token, country-agreement, and missingness comparisons. Avoid raw record IDs as predictive features.
- Establish a simple interpretable similarity-threshold baseline, then compare a licensed tree-based binary classifier (for example LightGBM or CatBoost, subject to installed package/license checks).
- Calibrate thresholds independently for S2 and S3 if validation supports it. Optimize the actual per-S1 macro F0.5, including singleton predictions and multi-link recall.
- Report blocking recall ceiling alongside end-to-end score so model errors are separated from missed candidates.

### 4. Error analysis and robustness

- Review false merges, missed matches, and predicted singletons by country, source, missingness, name/address similarity, and match cardinality.
- Add targeted blocking/features only when validation error analysis supports them.
- Confirm France test records flow through the same open-set logic and are never dropped due to unknown country labels.
- Compare experiments on the same fixed validation split; retain a known-best artifact and document configuration/score/runtime.

### 5. Final inference and package

- Refit on all labeled training entities after selecting the validation policy.
- Generate candidate and final TSVs for every test S1 ID, preserving exact headers and empty strings for no-match rows.
- Run the provided submission validator and internal invariants; confirm predicted IDs are from test S2/S3 and are subsets of candidates.
- Assemble the required zip layout with source, pinned dependencies, run instructions, outputs, and a completed methodology document describing measured results (do not fill in unmeasured claims).

## Definition of done

- Reproducible command rebuilds both TSVs from supplied files using only packaged code and dependencies.
- Validation uses macro F0.5 per Source 1 entity, singleton-aware, with thresholds selected without test labels.
- Candidate recall, end-to-end validation score, candidate volume, and known error patterns are documented.
- Submission validator passes; each test Source 1 entity occurs exactly once; every final match is a valid candidate and a test S2/S3 ID.
- Documentation states model and dependency licenses and confirms no external identity lookup was used.
