# MethylGPT

This is the official codebase for **methylGPT : a foundation model for the DNA methylome**.


[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2024.10.30.621013v2) &nbsp;
[![PyPI version](https://badge.fury.io/py/scgpt.svg)](https://pypi.org/project/methylgpt/) &nbsp;
#[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)

**!UPDATE**: 
**[2025.02.10]** methylGPT is now available on PyPI
**[2024.12.10]** We made initial launching of the methylGPT codebase.
**[2025.11.04]** Manuscript available on arXiv


## Installation

Architecture Note : MethylGPT's backend architecture is largely based on [scGPT](https://github.com/bowang-lab/scGPT), developed by the Wang Lab. As such, our project inherits and follows similar dependencies and architectural patterns. We acknowledge and thank the scGPT team for their foundational work.

methylGPT works with Python >= 3.9.10  and R >=3.6.1. Please make sure you have the correct version of Python and R installed pre-installation.

methylGPT is available on PyPI. To install methylGPT, run the following command:

```bash
pip install methylgpt "flash-attn<1.0.5"  # optional, recommended
```

[Optional] We recommend using [wandb](https://wandb.ai/) for logging and visualization.

```bash
pip install wandb
```

For developing, we are using the [Poetry](https://python-poetry.org/) package manager. To install Poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).

```bash
$ git clone this-repo-url
$ cd MethylGPT_clean
$ poetry install
```

**Note**: The `flash-attn` dependency usually requires specific GPU and CUDA version. If you encounter any issues, please refer to the [flash-attn](https://github.com/HazyResearch/flash-attention/tree/main) repository for installation instructions. For now, May 2023, we recommend using CUDA 11.7 and flash-attn<1.0.5 due to various issues reported about installing new versions of flash-attn.


## Running pretraining

The primary pretraining code is implemented in `methylgpt.pretraining.py`. During training, model checkpoints are automatically saved to the `save/` directory at the end of each epoch.

For a detailed walkthrough of the pretraining process, refer to our step-by-step examples in the [pretraining tutorials](tutorials/pretraining).

## # Pretrained methylGPT Models

This repository provides access to our suite of pretraining models for DNA methylation analysis. The major data sources for pretraining are derived from a comprehensive collection of human DNA methylation profiles.

### Major Data Sources for Pretraining

We collected a total of **226,555 human DNA methylation profiles** aggregated from **5,281 datasets** through two complementary resources: EWAS Data Hub and Clockbase.

| Data Sources                         | Datasets (Combined) | DNA Methylation Profiles | Description                                                                                  | Links                                                                                     |
|--------------------------------------|---------------------|--------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| EWAS Data Hub & Clockbase (Combined) | 5,281               | 226,555                  | Aggregated high-quality human DNA methylation profiles curated for pretraining purposes.     | [EWAS Data Hub](https://bigd.big.ac.cn/ewas/datahub) • [Clockbase](https://clockbase.org)     |

### Available Pretraining Models

Our current suite of pretraining models includes the following architectures:

| Model             | Hyperparameters                     | Parameters |
|-------------------|-------------------------------------|------------|
| methylGPT-tiny    | emb-dim: 64, layers: 6, heads: 4     | 3M         |
| methylGPT-small   | emb-dim: 128, layers: 6, heads: 4    | 7M         |
| methylGPT-normal  | emb-dim: 256, layers: 6, heads: 4    | 15M        |

### Usage

- **Recommended model:** We suggest using the `methylGPT-normal` model for most applications unless computational constraints require a lighter model.
- **Checkpoint folders:** We don't provide checkpoints yet. #Each model checkpoint is provided along with a paired vocabulary file mapping gene names to IDs.

## Fine-tune methylGPT for age prediction

Please see our example code in [tutorials/finetuning_age_prediction](tutorials/finetuning_age_prediction/finetuning_age_main.py). 

## To-do-list

- [x] Upload the pretrained model checkpoint
- [x] Publish to pypi
- [ ] Provide the pretraining code with generative attention masking
- [ ] More tutorial examples for disease prediction
- [ ] Publish to huggingface model hub

## Contributing

We greatly welcome contributions to methylGPT. Please submit a pull request if you have any ideas or bug fixes. We also welcome any issues you encounter while using methylGPT.

## Acknowledgements

We sincerely thank the authors of following open-source projects:

- [flash-attention](https://github.com/HazyResearch/flash-attention)
- [scanpy](https://github.com/scverse/scanpy)
- [scvi-tools](https://github.com/scverse/scvi-tools)
- [scib](https://github.com/theislab/scib)
- [datasets](https://github.com/huggingface/datasets)
- [transformers](https://github.com/huggingface/transformers)
- [scGPT](https://github.com/bowang-lab/scGPT)


## Citing methylGPT

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
