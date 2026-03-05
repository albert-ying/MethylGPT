# Tutorial: Disease Prediction

This tutorial demonstrates how to use MethylGPT embeddings for disease risk prediction and survival analysis.

## Files

- **`disease_prediction.ipynb`** -- Interactive notebook with two demo applications

## Overview

### Demo 1: Age Prediction from Embeddings

Uses the AltumAge dataset to demonstrate:
- Extracting embeddings from methylation data
- Training a Ridge regression model on embeddings
- Evaluating with Pearson correlation and MAE

### Demo 2: Survival Analysis (Template)

Provides a reusable pipeline for survival prediction:
- `survival_ridge_pipeline()` -- Ridge CV with C-index evaluation
- `evaluate_multiple_diseases()` -- Batch evaluation across disease endpoints
- `plot_cindex_bar()` -- Bar chart of C-index scores
- `plot_kaplan_meier()` -- Kaplan-Meier survival curves

## Requirements

- Pretrained MethylGPT model and data (run `bash download_data.sh` from tutorials/)
- AltumAge data (from `../finetuning_age_prediction/data/`)
- `lifelines` package for survival analysis: `pip install lifelines`
