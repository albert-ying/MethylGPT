# Tutorial: Fine-tuning MethylGPT for Age Prediction

This folder contains scripts for fine-tuning a pretrained MethylGPT model for age prediction tasks.

## Files

- **`finetuning_age_datasets.py`**
  - Handles data loading and preprocessing
  - Defines dataset classes and data transformations

- **`finetuning_age_models.py`**
  - Contains model architectures and configurations
  - Implements fine-tuning logic for age prediction

- **`finetuning_age_main.py`**
  - Main script for executing the fine-tuning process
  - Orchestrates training workflow and hyperparameter settings

- **`fintuning_age_metrics.py`**
  - Implements evaluation metrics for age prediction
  - Provides functions for model performance assessment

- **`train_methyGPT_altumage_dataset.yml`**
  - Configuration file for Altum Age dataset
  - Contains dataset-specific parameters and settings

- **`train_methyGPT_blood_dataset.yml`**
  - Configuration file for Blood methylation dataset
  - Specifies dataset-specific parameters and settings

## Usage

1. **Prepare Environment**
   - Ensure all required dependencies are installed
   - Configure dataset paths in the appropriate YAML file

2. **Execute Fine-tuning**
   ```bash
   python3 finetuning_age_main.py
   