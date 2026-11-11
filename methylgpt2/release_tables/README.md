# Release tables

Three tables define the MethylGPT 2 vocabulary and the split of its training corpus. Both CpG tables have one row per vocabulary CpG (49,156 rows, in the same order).

## methylgpt2_vocabulary_selection_2026-10-09.csv.gz

| Column | Meaning |
|---|---|
| `cpg` | Illumina probe ID |
| `ewas_atlas_association_records` | Number of association records for the CpG in the EWAS Atlas association table |
| `selected_by` | Criterion that brought the CpG into the vocabulary: five or more EWAS Atlas association records (28,947 CpGs), or membership of the 21,700 CpGs with fewest missing values in the corpus (20,209 further CpGs) |
| `on_27k`, `on_450k`, `on_epic_v1`, `on_epic_v2` | The CpG is on the 27K (22,287 CpGs), 450K (49,020), EPIC v1 (47,962) or EPIC v2 (45,668) array |
| `observed_fraction_in_corpus` | Fraction of corpus profiles in which the CpG is observed |
| `scenario_27k_shared_with_epic_v1`, `scenario_450k_shared_with_epic_v1` | The CpG belongs to the 27K-array (22,231 CpGs) or 450K-array (47,826 CpGs) set used in the missing-CpG scenarios, which keep the CpGs that each array shares with EPIC v1 |

## methylgpt2_vocabulary_annotation_2026-11-11.csv

| Column | Meaning |
|---|---|
| `cpg` | Illumina probe ID |
| `chr`, `pos_hg19` | Chromosome and position (hg19) |
| `island_relation` | Island, Shore, Shelf or OpenSea |
| `enhancer` | The CpG lies in an annotated enhancer |

## methylgpt2_profile_split_2026-10-03.csv.gz

| Column | Meaning |
|---|---|
| `profile_id` | Sample identifier in its source (GEO GSM, TCGA or TARGET barcode, ArrayExpress or EWAS Data Hub ID) |
| `dataset_group` | Dataset group; datasets that share samples form one group, and groups are assigned to splits whole |
| `split` | `train` (145,700 profiles, 1,941 groups), `dev` (development, 3,922 profiles, 15 groups), `test` (14,185 profiles, 116 groups), `excluded_downstream` (the 5,203 AltumAge validation and test profiles whose datasets fell in the training split, excluded from all training) or `excluded_qc` (110 profiles with more than 1% of values outside [0, 1]) |
