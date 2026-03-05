# MethylGPT Tutorials

Step-by-step guides for common MethylGPT workflows. All notebooks include a Colab setup cell and work both locally and on Google Colab.

**Tested on Python 3.9, 3.10, and 3.11.**

## Quick Setup

Download all shared data (pretrained models, probe IDs, sample parquet files) with a single command:

```bash
cd tutorials
bash download_data.sh
```

This fetches everything into `tutorials/shared_data/` and creates symlinks in each tutorial directory. The script is idempotent -- safe to run multiple times. It requires `gdown` (`pip install gdown`) and `wget`.

On **Google Colab**, each notebook auto-downloads its required files in the first cells, so no manual setup is needed.

## Tutorials

| Tutorial | Description | Format | Data Required |
|---|---|---|---|
| [Quickstart](quickstart/) | Get started in 5 minutes | Notebook | Pretrained model + Parquet data |
| [Embedding Extraction](get_embeddings/) | Extract sample-level embeddings from a pretrained model | Notebook + script | Pretrained checkpoint + Parquet data |
| [Embedding Analysis](embedding_analysis/) | UMAP visualization by tissue/age/sex, clustering, silhouette scores | Notebook | Embeddings + metadata (optional) |
| [Age Prediction](finetuning_age_prediction/) | Finetune MethylGPT for biological age prediction | Notebook + script | AltumAge dataset |
| [Disease Prediction](disease_prediction/) | Disease risk prediction using embeddings + Ridge regression | Notebook | Embeddings + metadata |
| [Imputation](imputation/) | Recover missing CpG methylation values | Notebook | Pretrained checkpoint + data with missing values |
| [Pretraining](pretraining/) | Train MethylGPT from scratch on methylation data | Script | Preprocessed Parquet files + probe IDs CSV |
| [CpG Selection](cpg_selection/) | Analyze CpG feature importance and selection | Notebook | Pretrained checkpoint |

## Data Downloads

- **Preprocessed dataset (type3):** [Download](https://www.dropbox.com/scl/fi/bbs6sxlkpbx11rhyvdfto/processed_type3_parquet_shuffled.tar.gz?rlkey=s73utmumq6xldmv3y6kh9bz75&st=8pslwy2a&dl=0)
- **CpG probe IDs (type3):** [Download](https://www.dropbox.com/scl/fi/2n6bx7j8v0aon0kwfsghp/probe_ids_type3.csv?rlkey=ly133xlce1xxjiku6tiski6qq&st=pig4e41h&dl=0)
- **Pretrained models:** See [main README](../README.md#pretrained-models)

## Setup

```bash
# Install MethylGPT with tutorial dependencies
pip install methylgpt[tutorials]

# Or from requirements.txt
pip install -r requirements.txt
pip install umap-learn matplotlib seaborn jupyter aquarel
```

## Shared Data Files

Run `bash download_data.sh` to download all shared files to `tutorials/shared_data/`. The script automatically creates symlinks in each tutorial directory. Key shared files:

- `probe_ids_type3.csv` -- CpG probe ID vocabulary (49,156 sites)
- `pretrained_models/tiny/` -- methylgpt-base (~3M params)
- `pretrained_models/medium/` -- methylgpt-medium (~25M params)
- `sample_data/processed_type3_parquet_shuffled/` -- Preprocessed methylation data
