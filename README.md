# MethylGPT

![schem](https://github.com/user-attachments/assets/a47c1d49-7bab-4556-921b-de01129b8e36)

**MethylGPT: a foundation model for the DNA methylome**

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2024.10.30.621013v2) &nbsp;
[![PyPI version](https://badge.fury.io/py/methylgpt.svg)](https://pypi.org/project/methylgpt/) &nbsp;
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](https://github.com/albert-ying/MethylGPT/blob/main/LICENSE)


## Overview

MethylGPT is a transformer-based foundation model pretrained on **226,555 human DNA methylation profiles** from **5,281 datasets**. It learns universal representations of the methylome that transfer to downstream tasks including:

- **Embedding extraction** -- dense sample-level representations for clustering, visualization, and classification
- **Age prediction** -- biological age estimation from methylation data
- **Disease prediction** -- disease risk stratification from methylation profiles
- **Imputation** -- recovery of missing CpG site values


## Installation

**Requirements:** Python 3.9, 3.10, or 3.11 | PyTorch >= 2.0

### From PyPI (recommended)

```bash
pip install methylgpt
```

### From requirements.txt (pinned versions)

```bash
git clone https://github.com/albert-ying/MethylGPT.git
cd MethylGPT
pip install -r requirements.txt
pip install -e .
```

### From conda (environment.yml)

```bash
git clone https://github.com/albert-ying/MethylGPT.git
cd MethylGPT
conda env create -f environment.yml
conda activate methylgpt
pip install -e .
```

### From source (for development)

```bash
git clone https://github.com/albert-ying/MethylGPT.git
cd MethylGPT
pip install -e ".[dev]"
```

### Optional: flash attention

For faster training and inference on supported GPUs:

```bash
pip install flash-attn --no-build-isolation
```


## Quick Start

```python
import torch
from methylgpt.model.methyl_model import MethylGPTModel
from methylgpt.model.methyl_vocab import MethylVocab

# Load vocabulary
vocab = MethylVocab(
    probe_id_dir="data/probe_ids_type3.csv",
    pad_token="<pad>",
    special_tokens=["<pad>", "<cls>", "<eoc>"],
    save_dir=None,
)

# Load pretrained model
config = {
    "layer_size": 128, "nhead": 4, "nlayers": 6,
    "dropout": 0.0, "fast_transformer": False, "pre_norm": False,
    "load_model": True,
    "pretrained_file": "pretrained_models/methylgpt-medium/best_model.pt",
}
model = MethylGPTModel.from_pretrained(config, vocab)
model.eval()

# Extract embeddings (gene_ids and values from your tokenized data)
with torch.no_grad():
    embeddings = model.get_cell_embeddings(gene_ids, values)
```

For complete working examples, see the [tutorials](#tutorials) below.


## Pretrained Models

| Model | Embedding Dim | Layers | Heads | Parameters | Download |
|---|---|---|---|---|---|
| **methylGPT-base** | 64 | 6 | 4 | 3M | [Download](https://drive.google.com/drive/folders/1kWdmkkVQpU17uzUC6-wpNR_4UEdxGx6k?usp=share_link) |
| **methylGPT-medium** | 128 | 6 | 4 | 7M | [Download](https://drive.google.com/drive/folders/14M4wdS83el9PAgh9TdfjSCeEcDPbz34f?usp=sharing) |
| **methylGPT-large** | 256 | 6 | 4 | 15M | [Download](https://drive.google.com/drive/folders/1lt8SF9MvoytPN3DeaxIss_ED9zNpf_Le?usp=share_link) |

- **base:** Lightweight experiments and quick prototyping.
- **medium:** Balanced performance for most applications (recommended).
- **large:** Comprehensive studies and maximum accuracy.


## Pretraining Data

The pretraining corpus comprises **226,555 human DNA methylation profiles** from **5,281 datasets** across [EWAS Data Hub](https://bigd.big.ac.cn/ewas/datahub) and [Clockbase](https://clockbase.org).

- **Preprocessed dataset (type3, default):** [Download](https://www.dropbox.com/scl/fi/bbs6sxlkpbx11rhyvdfto/processed_type3_parquet_shuffled.tar.gz?rlkey=s73utmumq6xldmv3y6kh9bz75&st=8pslwy2a&dl=0)
- **CpG probe IDs (type3, default):** [Download](https://www.dropbox.com/scl/fi/2n6bx7j8v0aon0kwfsghp/probe_ids_type3.csv?rlkey=ly133xlce1xxjiku6tiski6qq&st=pig4e41h&dl=0)

You can also load these data from a LaminDB instance: https://lamin.ai/laminlabs/methyldata

## Tutorials

| Tutorial | Description | Format |
|---|---|---|
| [Quickstart](tutorials/quickstart/) | Get started in 5 minutes | Notebook |
| [Embedding extraction](tutorials/get_embeddings/) | Extract cell-level embeddings from pretrained model | Notebook + script |
| [Embedding analysis](tutorials/embedding_analysis/) | UMAP visualization, clustering, silhouette scores | Notebook |
| [Age prediction](tutorials/finetuning_age_prediction/) | Finetune for biological age prediction | Notebook + script |
| [Disease prediction](tutorials/disease_prediction/) | Disease risk prediction with Ridge regression | Notebook |
| [Imputation](tutorials/imputation/) | Recover missing CpG values | Notebook |
| [Pretraining](tutorials/pretraining/) | Train MethylGPT from scratch on methylation data | Script |
| [CpG selection](tutorials/cpg_selection/) | Feature selection analysis | Notebook |

All notebooks include a Colab setup cell and work both locally and on Google Colab.

## Documentation

| Document | Description |
|---|---|
| [Inference Guide](docs/inference_guide.md) | Loading models, extracting embeddings, imputation, GPU/CPU tips |
| [API Reference](docs/api_reference.md) | Full public API with signatures and descriptions |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and solutions (flash-attn, CUDA OOM, torchtext, Colab) |


## Hardware Requirements

- **Inference / embedding extraction:** 1 GPU with >= 8 GB VRAM (or CPU, slower)
- **Finetuning:** 1 GPU with >= 16 GB VRAM recommended
- **Pretraining:** 1+ GPUs with >= 40 GB VRAM recommended (H100, A100)


## Contributing

We welcome contributions. Please submit a pull request for ideas or bug fixes. Open an issue for questions or problems.


## Acknowledgements

MethylGPT's backend architecture is based on [scGPT](https://github.com/bowang-lab/scGPT) (Wang Lab). We also thank:

- [flash-attention](https://github.com/HazyResearch/flash-attention)
- [scanpy](https://github.com/scverse/scanpy)
- [scvi-tools](https://github.com/scverse/scvi-tools)
- [PyTorch Lightning](https://lightning.ai/)


## Citing MethylGPT

```bibtex
@article{ying2024methylgpt,
  title={MethylGPT: a foundation model for the DNA methylome},
  author={Ying, Kejun and Song, Jinyeop and Cui, Haotian and Zhang, Yikun and Li, Siyuan and Chen, Xingyu and Liu, Hanna and Eames, Alec and McCartney, Daniel L and Marioni, Riccardo E and others},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

If you use the GEO metadata (derived from ClockBase), please also cite:

```bibtex
@article{ying2023clockbase,
  title={ClockBase: a comprehensive platform for biological age profiling in human and mouse},
  author={Ying, K. and Tyshkovskiy, A. and Trapp, A. and Liu, H. and Moqri, M. and Kerepesi, C. and Gladyshev, V.N.},
  journal={bioRxiv},
  year={2023},
  doi={10.1101/2023.02.28.530532}
}
```

## License

[Apache 2.0](LICENSE)
