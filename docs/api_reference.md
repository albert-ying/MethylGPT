# MethylGPT API Reference

## Top-Level Imports

```python
from methylgpt import (
    MethylGPTModel,
    MethylVocab,
    CustomDataset,
    StandardizedCpGDataset,
    split_files,
    create_dataloader,
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
    random_mask_value,
)
```

---

## Core Model

### `MethylGPTModel`

`methylgpt.model.methyl_model.MethylGPTModel`

Transformer model for DNA methylation analysis. Extends scGPT's `TransformerModel`.

#### `__init__(config, vocab)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `dict` | Model configuration (see below) |
| `vocab` | `MethylVocab` | Vocabulary instance |

**Config keys:**

| Key | Type | Description |
|-----|------|-------------|
| `layer_size` | `int` | Transformer hidden dimension (64, 128, or 256) |
| `nhead` | `int` | Number of attention heads |
| `nlayers` | `int` | Number of transformer layers |
| `dropout` | `float` | Dropout rate |
| `fast_transformer` | `bool` | Use flash attention |
| `pre_norm` | `bool` | Use pre-layer normalization |

#### `from_pretrained(config, vocab)` → `MethylGPTModel`

Class method. Loads a pretrained model from checkpoint.

Additional config keys required:
- `load_model` (`bool`): Whether to load weights
- `pretrained_file` (`str`): Path to `.pt` checkpoint file

#### `get_cell_embeddings(gene_ids, values)` → `Tensor`

Extract cell-level embeddings.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `gene_ids` | `Tensor` | `(B, L)` | CpG token IDs |
| `values` | `Tensor` | `(B, L)` | Methylation values |
| **Returns** | `Tensor` | `(B, D)` | Cell embeddings |

#### `prepare_data(batch)` → `dict`

Tokenize and mask a batch for pretraining/inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch` | `dict` | Dict with `"data"` key containing `(B, n_cpgs)` tensor |

Returns dict with keys: `"gene_ids"`, `"values"`, `"target_values"`.

---

## Vocabulary

### `MethylVocab`

`methylgpt.model.methyl_vocab.MethylVocab`

#### `__init__(probe_id_dir, pad_token, special_tokens, save_dir)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `probe_id_dir` | `str` | Path to CSV with `illumina_probe_id` column |
| `pad_token` | `str` | Padding token (typically `"<pad>"`) |
| `special_tokens` | `list[str]` | Special tokens list `["<pad>", "<cls>", "<eoc>"]` |
| `save_dir` | `str \| None` | Directory to save vocab JSON, or `None` |

**Attributes:**
- `vocab`: `torchtext.vocab.Vocab` — token-to-index mapping
- `CpG_list`: `list[str]` — list of CpG probe IDs
- `CpG_ids`: `np.ndarray` — token indices for CpG sites
- `pad_token`: `str` — padding token string

---

## Inference

### `extract_embeddings`

`methylgpt.inference.extract_embeddings`

```python
extract_embeddings(model, data_loader, device="cuda", max_batches=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `MethylGPTModel` | — | Pretrained model |
| `data_loader` | `DataLoader` | — | Data loader from `create_dataloader` |
| `device` | `str` | `"cuda"` | Compute device |
| `max_batches` | `int \| None` | `None` | Limit number of batches |

**Returns:** `(embeddings: np.ndarray, sample_ids: list[str])`

### `extract_cpg_embeddings`

`methylgpt.inference.extract_cpg_embeddings`

```python
extract_cpg_embeddings(model)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `MethylGPTModel` | Model with loaded weights |

**Returns:** `np.ndarray` of shape `(n_tokens, embedding_dim)`

---

## Datasets

### `CustomDataset`

`methylgpt.model.methyl_datasets.CustomDataset`

Streaming `IterableDataset` for Parquet files with columns `id` and `data`.

### `StandardizedCpGDataset`

`methylgpt.model.methyl_datasets.StandardizedCpGDataset`

Like `CustomDataset` but applies per-CpG z-score normalization during iteration.

### `create_dataloader`

```python
create_dataloader(parquet_chunk_files, batch_size, num_workers=None, max_workers=8)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parquet_chunk_files` | `list[str]` | — | Paths to Parquet files |
| `batch_size` | `int` | — | Batch size |
| `num_workers` | `int \| None` | `None` | Auto-detected if None |
| `max_workers` | `int` | `8` | Maximum number of workers |

**Returns:** `DataLoader`

### `split_files`

```python
split_files(files, valid_ratio)
```

Split file list into train/validation sets.

---

## Loss Functions

### `masked_mse_loss(input, target, mask)` → `Tensor`

MSE loss computed only on masked positions.

### `criterion_neg_log_bernoulli(input, target, mask)` → `Tensor`

Negative log-Bernoulli loss on masked positions.

### `masked_relative_error(input, target, mask)` → `Tensor`

Relative error on masked positions.

---

## Pretraining Utilities

### `random_mask_value`

```python
random_mask_value(values, mask_ratio=0.15, mask_value=-1, pad_value=-2)
```

Randomly mask a fraction of input values for pretraining.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | `Tensor` | — | Input values `(B, L)` |
| `mask_ratio` | `float` | `0.15` | Fraction of values to mask |
| `mask_value` | `float` | `-1` | Replacement value for masked positions |
| `pad_value` | `float` | `0` | Padding value (not masked) |

**Returns:** `Tensor` with masked values.

---

## Finetuning

### `methylgpt.finetuning`

#### Model Heads

- **`RegressionHead`**: 3-layer MLP for regression tasks (e.g., age prediction)
- **`ClassificationHead`**: Deep SELU MLP for classification tasks
- **`EmbeddingReductionHead`**: Linear reduction + classification
- **`FintuneModel`** (`pl.LightningModule`): Combines encoder + task head with differential learning rates

#### Dataset Classes

- **`CollatableVocab`**: Wraps probe IDs + tokenization config
- **`TokenizedDataset`**: Applies tokenization + masking in collater
- **`RawDataset`**: Simple dataset without in-batch tokenization

#### Metrics

- **`elasticnet_metric()`**: R², RMSE, MAE, Pearson, Spearman correlations
- **`regression_metric()`**: Aggregate validation outputs → metrics dict
- **`disease_metric()`**: Multi-label classification metrics (micro/macro/weighted)

---

## Utilities

### Logging

```python
from methylgpt.utils.logging import setup_logger, add_console_handler

logger = setup_logger("my_logger", "output.log")
add_console_handler(logger)
```

### Plotting

```python
from methylgpt.utils.plot_embeddings import plot_umap_categorical, plot_umap_numerical

plot_umap_categorical("tissue", embedding_df, save_as="umap_tissue.png")
plot_umap_numerical("age", embedding_df, save_as="umap_age.png")
```
