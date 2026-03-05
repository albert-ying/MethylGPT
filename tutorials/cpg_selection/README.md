# Tutorial: CpG Site Selection

This tutorial explains the criteria used to select the 49,156 CpG sites used in MethylGPT pretraining.

## Files

- **`cpg_selection.ipynb`** -- Interactive notebook analyzing CpG selection criteria

## Selection Criteria

MethylGPT uses CpG sites that meet all three criteria:

1. **EWAS association with >= 5 traits** -- Sites with broad biological relevance
2. **Present in > 95% of datasets** -- Ensures data availability across training samples
3. **Variance > 0.01 across samples** -- Excludes invariant sites

## Data Requirements

The notebook requires the EWAS Atlas associations file:
- Download from [EWAS Atlas](https://ngdc.cncb.ac.cn/ewas/atlas) (Downloads -> Association -> TSV file)
- Save as `EWAS_Atlas_associations.tsv` in the notebook directory

## Output

- Threshold comparison table with genome coverage metrics
- Trait frequency distribution plots
- Memory estimate table for different CpG set sizes
- Filtered probe ID list for custom configurations
