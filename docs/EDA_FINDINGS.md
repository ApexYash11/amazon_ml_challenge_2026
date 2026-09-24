# Initial EDA Findings and Recommended Approach

EDA was run locally on the supplied TSVs with the streaming scripts in `tools/`. No external services or data were used. Counts below are exact full-file row counts. Pair-level text statistics are from the first 5,000 ground-truth rows (17,250 known links), so use them as a diagnostic sample rather than a final validation estimate.

Reproduce from the repository root:

```powershell
node tools/eda_stream.mjs 6ab10eb3b23ba_student_resource/student_resource/dataset
node tools/eda_link_sample.mjs 6ab10eb3b23ba_student_resource/student_resource/dataset 5000
```

## Scale and schema

| File | Records |
| --- | ---: |
| Train Source 1 | 2,206,821 |
| Train Source 2 | 5,034,616 |
| Train Source 3 | 5,285,603 |
| Train labels | 2,206,821 |
| Test Source 1 | 1,732,544 |
| Test Source 2 | 4,887,273 |
| Test Source 3 | 5,082,316 |

Source TSVs have the expected four columns: `entity_id`, `business_name`, `business_address`, and `country`. Ground truth has `source1_entity_id` and `matched_entity_ids`. All Source 1 rows have nonempty names, addresses, and countries. Source 2/3 have no empty names or countries, but addresses are missing on 168,967/5,034,616 train S2 records (3.36%), 175,916/5,285,603 train S3 records (3.33%), 129,408/4,887,273 test S2 records (2.65%), and 136,098/5,082,316 test S3 records (2.68%).

The median normalized name lengths are 23–24 characters and median address lengths are 35–48 characters, depending on source/split. Names are short enough for character n-gram and token retrieval; addresses are longer and need token, digit, and character signals. A first-100k-row sample shows repeated normalized names in every source (10,124 S1, 2,377 S2, and 2,154 S3 repeats); exact name blocks alone will contain ambiguous common names.

## Label structure

- 123,247 of 2,206,821 training Source 1 entities have no matches: **5.59% singletons**.
- The remaining 2,083,574 entities have at least one link.
- Ground truth contains 3,693,619 S2 links and 3,944,746 S3 links (7,638,365 total).
- Link count per Source 1 ranges from 0 to 11. Cardinalities 2–5 are common; the task is genuinely many-to-many and must not impose a one-to-one constraint.
- Every link in the full label file has an S2- or S3-prefixed ID; no invalid prefixes were observed.

## Country shift

Training contains US and India only. Test includes France: 259,452/1,732,544 S1 records (14.98%), 703,378/4,887,273 S2 records (14.39%), and 731,615/5,082,316 S3 records (14.39%). Keep the country as an open string feature. The sampled positive links had no cross-country pairs (0/17,250); country equality looks like a valuable candidate partition, but validate that finding across held-out data before making it a hard gate. Do not treat France as unknown/missing or drop it.

## Linked-pair diagnostics

For the first 5,000 training label rows, all 5,000 S1 records and all 17,250 referenced target records were found in their source files:

| Signal among true links | Count | Share |
| --- | ---: | ---: |
| Normalized name exact | 3,650 | 21.2% |
| Normalized address exact | 1,384 | 8.0% |
| At least one shared name token (token length ≥2) | 14,718 | 85.3% |
| At least one shared address token | 16,505 | 95.7% |
| Shared token in both name and address | 13,974 | 81.0% |
| Shared address digit run of length ≥4 | 4,759 | 27.6% |
| Same country | 17,250 | 100% |

True-link token Jaccard median was 0.667 for names and 0.636 for addresses. Name Jaccard's 10th percentile was 0, so name-only token blocking would miss a material tail. Address-only blocking also misses some links; candidate generation should union independent name and address retrieval paths, with character n-gram fallback for typo/transliteration variants.

## Recommended approach

1. **Split validation by S1 IDs**, stratifying by singleton status and (where practical) match cardinality/country. Keep all true IDs for held-out rows out of training and threshold fitting. Measure candidate recall and final macro F0.5 on the same fixed holdout.
2. **Partition by exact country label** as the first retrieval scope only after the full training holdout confirms the sampled zero cross-country rate. This naturally supports France without a closed vocabulary. Keep a fallback review for missing or conflicting country values.
3. **Build sparse, capped candidate retrieval** separately for S1×S2 and S1×S3: normalized exact keys; rare-token inverted lists for names and addresses; address digit/postcode keys; and character n-gram TF-IDF top-k as a complementary path. Cap very frequent postings and use top-k retrieval to control candidates. Measure blocking recall separately by target source and by low-similarity tail.
4. **Train a pairwise binary scorer** on known links that survive blocking and hard negatives drawn from retrieved near-neighbors. Initial feature set: name/address token Jaccard and cosine; character n-gram cosine/edit similarity; exact normalized-field flags; address digit overlap; country equality; and missingness. Avoid row IDs as predictive features.
5. **Predict each pair independently**, then tune separate S2/S3 thresholds (or calibrated probabilities) against per-S1 macro F0.5. This permits multiple matches and accounts for singleton credit. Compare a similarity threshold baseline with a licensed CPU-friendly tree model such as LightGBM/CatBoost after confirming dependency availability.
6. **Engineer for the data volume.** Training sources 2 and 3 total about 10.3M records; test sources 2 and 3 total about 10.0M. Full Cartesian comparisons against S1 would require roughly 22.8 trillion train and 17.3 trillion test pairs. Candidate lookup therefore needs inverted sparse indices or another bounded-memory retrieval design, partitioned by country/source, with compact IDs and streamed output. Do not materialize the Cartesian product or all pair features in RAM.

## Known limits and next measurements

- Pair text results are a sequential prefix sample, not a randomized, stratified, or held-out estimate.
- The first full pass did not calculate global unique-name/address cardinality or full positive-pair country agreement; verify these on the actual validation split.
- No predictive validation score exists yet. Thresholds, candidate top-k, blocker recall, hard-negative ratio, and final model choice must be selected through the fixed S1 holdout.
- Python is not installed in the current Windows environment (`py` reports no installed Python). The EDA itself ran under Node; Python training/inference and submission validation still require a Python environment before implementation can be verified.
