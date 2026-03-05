# MethylGPT Inference Guide

This guide covers loading a pretrained MethylGPT model and using it for inference tasks: embedding extraction, imputation, and downstream predictions.

## Loading a Pretrained Model

### 1. Prepare vocabulary and config

```python
import torch
import json
from pathlib import Path
from methylgpt import MethylGPTModel, MethylVocab

# Load vocabulary from CpG probe IDs
vocab = MethylVocab(
    probe_id_dir="data/probe_ids_type3.csv",
    pad_token="<pad>",
    special_tokens=["<pad>", "<cls>", "<eoc>"],
    save_dir=None,
)

# Load model configuration from args.json
model_dir = Path("pretrained_models/methylgpt-medium")
with open(model_dir / "args.json", "r") as f:
    config = json.load(f)

# Point to the checkpoint file
config["load_model"] = True
config["pretrained_file"] = str(model_dir / "best_model.pt")

# Load model
model = MethylGPTModel.from_pretrained(config, vocab)
model.eval()
```

### 2. Choose device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optional: use half precision for faster inference (GPU only)
if device.type == "cuda":
    model = model.half()
```

## Extracting Cell-Level Embeddings

Cell (sample) embeddings are dense vectors summarizing the full methylation profile of a sample. Use them for clustering, visualization, classification, and regression.

### Using the high-level API

```python
from methylgpt import create_dataloader
from methylgpt.inference import extract_embeddings

# Create data loader from Parquet files
parquet_files = ["data/samples_part1.parquet", "data/samples_part2.parquet"]
data_loader = create_dataloader(parquet_files, batch_size=32)

# Extract embeddings
embeddings, sample_ids = extract_embeddings(model, data_loader, device=str(device))
print(f"Embeddings shape: {embeddings.shape}")  # (n_samples, embedding_dim)
```

### Using the low-level API

```python
import numpy as np

with torch.no_grad():
    for batch in data_loader:
        batch_data = model.prepare_data(batch)
        gene_ids = batch_data["gene_ids"].to(device)
        values = batch_data["values"].to(device)

        cell_embs = model.get_cell_embeddings(gene_ids, values)
        print(f"Batch embeddings: {cell_embs.shape}")  # (batch_size, embedding_dim)
```

## Extracting CpG-Level Embeddings

CpG embeddings are learned representations for each CpG site. Use them for CpG similarity analysis, clustering, and annotation.

```python
from methylgpt.inference import extract_cpg_embeddings

cpg_embeddings = extract_cpg_embeddings(model)
print(f"CpG embeddings shape: {cpg_embeddings.shape}")  # (n_tokens, embedding_dim)

# Skip special tokens (first 3: <pad>, <cls>, <eoc>)
cpg_only = cpg_embeddings[3:]  # (n_cpgs, embedding_dim)
```

## Imputation (Missing Value Recovery)

MethylGPT can predict missing CpG values using its masked language model head.

```python
from methylgpt.model.methyl_pretraining import random_mask_value

with torch.no_grad():
    for batch in data_loader:
        batch_data = model.prepare_data(batch)
        gene_ids = batch_data["gene_ids"].to(device)
        values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)

        src_key_padding_mask = gene_ids.eq(vocab[vocab.pad_token]).to(device)
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
        )
        predictions = output_dict["mlm_output"]  # Predicted values for masked positions
```

## Data Format

### Parquet files

MethylGPT expects data in Parquet format with two columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Sample identifier (e.g., GSM accession) |
| `data` | list[float] | Methylation beta values for all CpG sites |

The `data` column must contain a list of float values matching the order of CpG sites in `probe_ids_type3.csv`. Missing values should be `NaN`.

### Probe IDs CSV

The `probe_ids_type3.csv` file maps CpG site positions to Illumina probe IDs:

```csv
illumina_probe_id
cg00000029
cg00000108
...
```

## GPU vs CPU

| Setting | GPU (recommended) | CPU |
|---------|-------------------|-----|
| Speed | ~6 sec/batch | ~60 sec/batch |
| Half precision | Supported (`model.half()`) | Not supported |
| Memory | ≥8 GB VRAM | ≥16 GB RAM |

## Memory Requirements

| Model | Parameters | GPU VRAM (fp16) | GPU VRAM (fp32) |
|-------|-----------|-----------------|-----------------|
| methylGPT-base (64-dim) | 3M | ~2 GB | ~4 GB |
| methylGPT-medium (128-dim) | 7M | ~4 GB | ~8 GB |
| methylGPT-large (256-dim) | 15M | ~6 GB | ~12 GB |

For large datasets, use `max_batches` parameter in `extract_embeddings()` to limit memory usage, or reduce `batch_size` in the data loader.

## Tips

- **Batch size**: Start with 32. Reduce if you encounter CUDA OOM errors.
- **Number of workers**: `create_dataloader` auto-detects optimal workers. For single-file datasets, only 1 worker is used.
- **Half precision**: Use `model.half()` on GPU for 2x speed and 50% memory reduction.
- **Mask ratio**: Set `config["mask_ratio"] = 0` for embedding extraction (no masking). Use `0.15`–`0.3` for imputation tasks.
