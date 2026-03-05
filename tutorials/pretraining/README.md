# Tutorial: Pretraining MethylGPT

This tutorial demonstrates how to pretrain a MethylGPT model from scratch on DNA methylation data.

## Files

- **`run_pretraining.sh`** -- Main script that orchestrates preprocessing and training
- **`pretraining.py`** -- Training loop implementation
- **`preprocessing_CpGs.py`** -- Data preprocessing (raw CSV to parquet)
- **`config_example.json`** -- Example training configuration
- **`probe_ids_type3.csv`** -- CpG probe ID list (49,156 sites)
- **`utils.py`** -- Utility functions for data loading

## Quick Start

```bash
# 1. Navigate to the pretraining directory
cd tutorials/pretraining

# 2. Download example data
bash download_files.sh

# 3. Review and update run_pretraining.sh variables (marked with # TOCHANGE)

# 4. Run pretraining
bash run_pretraining.sh
```

## Configuration

### `run_pretraining.sh` Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUN_PREPROCESSING` | Run preprocessing step | `true` |
| `RAW_DATA_DIR` | Directory with raw CSV files | `data_example` |
| `PROBE_ID_REF` | Probe ID reference file | `probe_ids_type3.csv` |
| `CONFIG_FILE` | Training config JSON | `config_example.json` |
| `DEVICE` | CUDA device | `cuda:0` |
| `SAVENAME_PREFIX` | Output directory prefix | `pretraining_test` |

### `config_example.json` Parameters

Key training parameters include:
- `layer_size`: Transformer hidden dimension (64 for tiny, 128 for medium, 256 for large)
- `nhead`: Number of attention heads
- `nlayers`: Number of transformer layers
- `dropout`: Dropout rate
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `mask_ratio`: Fraction of CpG values to mask during training

## Pipeline

1. **Preprocessing** (if `RUN_PREPROCESSING=true`):
   - Reads raw methylation CSV files
   - Filters CpG sites by probe ID list
   - Generates QC metadata (`QCed_samples_type3.csv`)
   - Saves preprocessed parquet files

2. **Training**:
   - Loads preprocessed parquet data
   - Initializes MethylGPT transformer model
   - Trains with masked value prediction objective
   - Saves model checkpoints and logs

## Output

- Model checkpoints (`.pt` files) in timestamped output directory
- Training configuration (`args.json`)
- Vocabulary file (`vocab.json`)
- Training logs (optionally via Weights & Biases)

## Requirements

- GPU with sufficient memory (recommended: 24+ GB for default config)
- Processed methylation data in CSV format
- See `requirements.txt` in the project root for Python dependencies
