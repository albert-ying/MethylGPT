# Tutorial: CpG Methylation Imputation

This tutorial demonstrates how to use MethylGPT's masked language model (MLM) head to recover missing CpG methylation values.

## Overview

MethylGPT is pre-trained with a masked prediction objective: during training, a fraction of CpG values are masked and the model learns to predict them from surrounding context. This mechanism can be used at inference time to impute missing or held-out methylation values.

## Files

- **`imputation.ipynb`** -- Interactive notebook demonstrating the full imputation pipeline
- **`README.md`** -- This file

## Quick Start

```bash
# 1. Download model and data (from tutorials/ root)
bash download_data.sh

# 2. Run the notebook
jupyter notebook imputation.ipynb
```

## What the Notebook Covers

1. **Load pretrained model** -- Load a MethylGPT checkpoint and CpG vocabulary
2. **Prepare data** -- Load methylation parquet files and create a DataLoader
3. **Mask and predict** -- Mask 15% of CpG values, run model forward pass, collect predictions
4. **Evaluate quality** -- Compare predictions vs ground truth (MAE, RMSE, R-squared, Pearson r)
5. **Visualize results** -- Scatter plot, residual distribution, MAE by methylation level

## How Imputation Works

### Target CpG in Training List

If your target CpG site exists in the pre-trained CpG list (`probe_ids_type3.csv`):

1. Set those CpG positions to the `mask_value` (-1)
2. Run a forward pass through the model
3. The `mlm_output` tensor contains predicted values for all positions, including masked ones

### Target CpG Not in Training List

If the target CpG was not included during pre-training, fine-tuning is required:

1. Add the new CpG site to the model vocabulary
2. Fine-tune with data containing this CpG site
3. Then follow the masking approach above

## Requirements

- Pre-trained MethylGPT model checkpoint
- Processed methylation data in parquet format
- CpG probe ID list (`probe_ids_type3.csv`)
- GPU recommended for faster inference
