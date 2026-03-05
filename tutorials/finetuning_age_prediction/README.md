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

- **`finetuning_age_metrics.py`**
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
   ```

# Finetuning Age Prediction Demo

This demo showcases fine-tuning a methylGPT model for age prediction.

## Models

The models can be downloaded from the following Google Drive links:

*   **Tiny:** [https://drive.google.com/drive/folders/1kWdmkkVQpU17uzUC6-wpNR_4UEdxGx6k](https://drive.google.com/drive/folders/1kWdmkkVQpU17uzUC6-wpNR_4UEdxGx6k)
*   **Small:** [https://drive.google.com/drive/folders/14M4wdS83el9PAgh9TdfjSCeEcDPbz34f](https://drive.google.com/drive/folders/14M4wdS83el9PAgh9TdfjSCeEcDPbz34f)
*   **Medium:** [https://drive.google.com/drive/folders/1lt8SF9MvoytPN3DeaxIss_ED9zNpf_Le](https://drive.google.com/drive/folders/1lt8SF9MvoytPN3DeaxIss_ED9zNpf_Le)

For this demo, please download and use the **tiny** model.

## Running the AltumAge Finetuning Demo

This demo uses the AltumAge dataset for fine-tuning. The main script to run is `run_finetuning_altumage.sh`.

**Before running, ensure the following are correctly set up:**

1.  **Download Data and Pretrained Model:**
    *   Run the `download_files.sh` script to download the "tiny" pretrained model and the AltumAge dataset.
    ```bash
    cd tutorials/finetuning_age_prediction/
    chmod +x download_files.sh
    ./download_files.sh
    ```
    *   This will create:
        *   `pretrained_models/tiny/` directory with `args.json` and the model weights file (e.g., `tiny-best_model_epoch10.pt`).
        *   `data/altumage_metadata/` directory with `altumage_train.parquet`, `altumage_valid.parquet`, and `altumage_test.parquet`.
    *   You can use `inspect_data.py` to check the downloaded data:
    ```bash
    python inspect_data.py
    ```

2.  **Verify Probe ID File:**
    *   Ensure `probe_ids_type3.csv` is present in the current directory (`tutorials/finetuning_age_prediction/`). The `download_files.sh` script should also handle this.

3.  **Configure `train_methyGPT_altumage.yml`:**
    *   Open `tutorials/finetuning_age_prediction/train_methyGPT_altumage.yml`.
    *   Verify/update `pretrained_model_args_path`: Should point to the `args.json` of your downloaded tiny model (e.g., `"pretrained_models/tiny/args.json"`).
    *   Verify/update `pretrained_file`: Should point to the actual model weights file (e.g., `\"pretrained_models/tiny/tiny-best_model_epoch10.pt\"`). **Make sure this filename matches your downloaded model.**
    *   Verify/update `data_dir`: Should point to the directory containing the altumage parquet files (e.g., `"data/altumage_metadata"`). This can also be overridden by the bash script.
    *   Verify/update `train_file_name`, `valid_file_name`, `test_file_name` if your parquet files have different names.
    *   Adjust other parameters like `max_epochs`, `pretrained_lr`, `head_lr`, `train_batch_size`, `wandb` (for Weights & Biases logging) as needed.
    *   Verify/update `probe_id_file_path`: Should point to the probe ID file (e.g., `\"./probe_ids_type3.csv\"`). This can also be overridden by the bash script.

4.  **Configure `run_finetuning_altumage.sh`:**
    *   Open `tutorials/finetuning_age_prediction/run_finetuning_altumage.sh`.
    *   This script uses default values. Modify these variables at the top of the script if needed:
        *   `CONFIG_FILE` (default: `"train_methyGPT_altumage.yml"`)
        *   `DEFAULT_GPUS` (e.g., `"0"` for GPU 0, `"1"` for GPU 1. The script will set `CUDA_VISIBLE_DEVICES` based on this.)
        *   `DATA_DIR` (default: `"./data/altumage_metadata/"`)
        *   `OUTPUT_DIR` (default: `"./output_altumage/"`)
        *   `PROBE_ID_FILE` (default: `"./probe_ids_type3.csv"`)

5.  **Set up Conda Environment:**
    *   Ensure your conda environment (e.g., `methygpt`) with all necessary dependencies (PyTorch, PyTorch Lightning, pandas, etc.) is active.
    *   The `run_finetuning_altumage.sh` script has commented-out lines for activating conda. Uncomment and adjust if needed.

**To run the fine-tuning:**

```bash
cd tutorials/finetuning_age_prediction/
chmod +x run_finetuning_altumage.sh
./run_finetuning_altumage.sh
```

This will start the fine-tuning process using the configurations you've set.
Checkpoints will be saved in the path specified by `weights_save_path` in the YAML file.
If `wandb: True` is set in the YAML, logs will also be sent to Weights & Biases.
