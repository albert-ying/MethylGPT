# Troubleshooting

Common issues and solutions when using MethylGPT.

## Installation Issues

### `flash_attn` import error

**Error:**
```
ImportError: No module named 'flash_attn'
```

**Solution:** Flash attention is optional. MethylGPT works without it. If you want to install it:
```bash
pip install flash-attn --no-build-isolation
```

Requires CUDA toolkit and a compatible GPU (Ampere or newer: A100, H100, RTX 30xx/40xx). If installation fails, set `fast_transformer: False` in your config.

### `torchtext` version mismatch

**Error:**
```
ModuleNotFoundError: No module named 'torchtext'
```
or
```
ImportError: cannot import name 'Vocab' from 'torchtext.vocab'
```

**Solution:** `torchtext` must match your PyTorch version:

| PyTorch | torchtext |
|---------|-----------|
| 2.0.x | 0.15.x |
| 2.1.x | 0.16.x |
| 2.2.x | 0.17.x |
| 2.3.x | 0.18.x |

Install the matching version:
```bash
pip install torchtext==0.16.0  # for torch 2.1.0
```

### `IProgress` warning

**Warning:**
```
TqdmWarning: IProgress not found. Please update jupyter and ipywidgets.
```

**Solution:** This is harmless — progress bars will still work. To suppress:
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Python version incompatibility

**Error:** Various import errors or `SyntaxError` on Python 3.12+.

**Solution:** MethylGPT requires Python 3.9, 3.10, or 3.11. Check your version:
```bash
python --version
```

Create a compatible environment:
```bash
conda create -n methylgpt python=3.11
conda activate methylgpt
pip install methylgpt
```

## Runtime Issues

### CUDA out of memory (OOM)

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions (try in order):**

1. **Reduce batch size:**
   ```python
   data_loader = create_dataloader(parquet_files, batch_size=8)  # default is 32
   ```

2. **Use half precision:**
   ```python
   model = model.half()
   ```

3. **Use a smaller model** (base=64-dim instead of large=256-dim)

4. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

5. **Process fewer batches:**
   ```python
   embeddings, ids = extract_embeddings(model, loader, max_batches=100)
   ```

### Missing data files

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/probe_ids_type3.csv'
```

**Solution:** Download the required files:

- **Probe IDs:** [Download](https://www.dropbox.com/scl/fi/2n6bx7j8v0aon0kwfsghp/probe_ids_type3.csv?rlkey=ly133xlce1xxjiku6tiski6qq&st=pig4e41h&dl=0)
- **Preprocessed dataset:** [Download](https://www.dropbox.com/scl/fi/bbs6sxlkpbx11rhyvdfto/processed_type3_parquet_shuffled.tar.gz?rlkey=s73utmumq6xldmv3y6kh9bz75&st=8pslwy2a&dl=0)
- **Pretrained models:** See [README](../README.md#pretrained-models)

Place files in the expected directory structure:
```
MethylGPT/
├── data/
│   └── probe_ids_type3.csv
├── pretrained_models/
│   └── methylgpt-medium/
│       ├── args.json
│       └── best_model.pt
```

### Too many dataloader workers

**Warning:**
```
Too many dataloader workers: 24 (max is dataset.n_shards=1). Stopping 23 dataloader workers.
```

**Solution:** This is handled automatically. When using a single Parquet file, only 1 worker is needed. The warning is informational and can be ignored.

### Model loading — partial weight match

**Warning:**
```
Warning: Skipped loading N incompatible params: {...}
```

**Solution:** This occurs when the checkpoint was trained with a different model configuration. Compatible parameters are loaded and incompatible ones are initialized randomly. This is expected when using a checkpoint from a different model size.

## Google Colab Issues

### GPU not available

**Problem:** `torch.cuda.is_available()` returns `False` on Colab.

**Solution:**
1. Go to **Runtime → Change runtime type**
2. Select **T4 GPU** (or any GPU option)
3. Click **Save** and restart the runtime

### Package installation hangs

**Problem:** `pip install methylgpt` takes too long on Colab.

**Solution:** Install with quiet mode and avoid unnecessary deps:
```python
!pip install -q methylgpt
```

### Kernel crashes during inference

**Problem:** Colab kernel dies during embedding extraction.

**Solution:** Colab free tier has limited GPU memory (~15 GB). Use:
```python
model = model.half()  # Use fp16
data_loader = create_dataloader(files, batch_size=8)  # Smaller batches
embeddings, ids = extract_embeddings(model, data_loader, max_batches=50)
```

## Getting Help

If you encounter an issue not covered here:

1. Check the [GitHub Issues](https://github.com/albert-ying/MethylGPT/issues)
2. Open a new issue with:
   - Python version (`python --version`)
   - Package versions (`pip list | grep -E "torch|methylgpt"`)
   - Full error traceback
   - Minimal code to reproduce
