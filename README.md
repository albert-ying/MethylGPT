# MethylGPT

![schem](https://github.com/user-attachments/assets/a47c1d49-7bab-4556-921b-de01129b8e36)


This is the official codebase for **methylGPT : a foundation model for the DNA methylome**.


[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2024.10.30.621013v2) &nbsp;
[![PyPI version](https://badge.fury.io/py/scgpt.svg)](https://pypi.org/project/methylgpt/) &nbsp;
[![License](https://img.shields.io/badge/license-Apache-blue)](https://github.com/albert-ying/MethylGPT/blob/main/LICENSE)

**!UPDATE**: 

**[2025.02.18]** We released our training dataset comprising 226,555 human DNA methylation profiles from 5,281 datasets - the largest training data set available for DNA methylation  

**[2025.02.10]** methylGPT is now available on PyPI

**[2024.12.10]** We made initial launching of the methylGPT codebase.

**[2024.11.04]** Manuscript available on biorXiv


## Installation

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
$ git clone https://github.com/albert-ying/MethylGPT
$ cd MethylGPT
$ poetry install
```

**Note**: The `flash-attn` dependency usually requires specific GPU and CUDA version. If you encounter any issues, please refer to the [flash-attn](https://github.com/HazyResearch/flash-attention/tree/main) repository for installation instructions. For now, May 2023, we recommend using CUDA 11.7 and flash-attn<1.0.5 due to various issues reported about installing new versions of flash-attn.


## Running pretraining

The primary pretraining code is implemented in `methylgpt.pretraining.py`. During training, model checkpoints are automatically saved to the `save/` directory at the end of each epoch.

For a detailed walkthrough of the pretraining process, refer to our step-by-step examples in the [pretraining tutorials](tutorials/pretraining).

## Pretrained methylGPT Models

This repository provides access to our suite of pretraining models for DNA methylation analysis. The major data sources for pretraining are derived from a comprehensive collection of human DNA methylation profiles.

### Major Data Sources for Pretraining

We collected a total of **226,555 human DNA methylation profiles** aggregated from **5,281 datasets** through two complementary resources: EWAS Data Hub and Clockbase.

| Data Sources                         | Datasets (Combined) | DNA Methylation Profiles | Description                                                                                  | Links                                                                                     |
|--------------------------------------|---------------------|--------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| EWAS Data Hub & Clockbase (Combined) | 5,281               | 226,555                  | Aggregated high-quality human DNA methylation profiles curated for pretraining purposes.     | [EWAS Data Hub](https://bigd.big.ac.cn/ewas/datahub) • [Clockbase](https://clockbase.org)     |


### Pretraining Dataset

MethylGPT leverages one of the largest DNA methylation corpora ever assembled, comprising **226,555 human DNA methylation profiles** meticulously collected from **5,281 datasets** across two complementary resources: EWAS Data Hub and Clockbase. This unprecedented scale enables our foundation model to capture the complex patterns and variations in the human methylome across diverse tissue types, conditions, and developmental stages. Our comprehensive preprocessing pipeline ensures high-quality input data, with pre-processed datasets readily available for download. This extensive pretraining corpus forms the backbone of MethylGPT's remarkable ability to generalize across methylation tasks and generate biologically meaningful predictions, even in low-data settings.

- **Preprocessed dataset type3 (default)**: [Download here](https://www.dropbox.com/scl/fi/bbs6sxlkpbx11rhyvdfto/processed_type3_parquet_shuffled.tar.gz?rlkey=s73utmumq6xldmv3y6kh9bz75&st=8pslwy2a&dl=0)  
- **Preprocessed dataset CpG IDs type3 (default)**: [Download here](https://www.dropbox.com/scl/fi/2n6bx7j8v0aon0kwfsghp/probe_ids_type3.csv?rlkey=ly133xlce1xxjiku6tiski6qq&st=pig4e41h&dl=0)

You can also load these data from a LaminDB instance: https://lamin.ai/laminlabs/methyldata

### Available Pretraining Models

Our current suite of pretraining models includes the following architectures available for download:

| Model                | Embedding Dimension | Layers | Heads | Parameters | Download Link                                                                                           |
|----------------------|---------------------|--------|-------|------------|---------------------------------------------------------------------------------------------------------|
| **methylGPT-base**   | 64                  | 6      | 4     | 3M         | [Download base](https://drive.google.com/drive/folders/1kWdmkkVQpU17uzUC6-wpNR_4UEdxGx6k?usp=share_link)   |
| **methylGPT-medium**  | 128                 | 6      | 4     | 7M         | [Download medium](https://drive.google.com/drive/folders/14M4wdS83el9PAgh9TdfjSCeEcDPbz34f?usp=sharing)     |
| **methylGPT-large** | 256                 | 6      | 4     | 15M        | [Download large](https://drive.google.com/drive/folders/1lt8SF9MvoytPN3DeaxIss_ED9zNpf_Le?usp=share_link) |

Choose the appropriate model size based on your computational resources and intended use:

- **base:** Ideal for lightweight experiments and quick prototyping.
- **medium:** Suitable for moderate computational resources and intermediate analyses.
- **large:** Recommended for comprehensive studies and robust analytical tasks.


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

MethylGPT's backend architecture is largely based on [scGPT](https://github.com/bowang-lab/scGPT), developed by the Wang Lab. As such, our project inherits and follows similar dependencies and architectural patterns. We acknowledge and thank the scGPT team for their foundational work.

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

The GEO metadata file is derived from the work from the ClockBase. If you use that metadata, please also cite our ClockBase paper.

```bibtex
@article{ying2023clockbase,
  title={ClockBase: a comprehensive platform for biological age profiling in human and mouse},
  author={Ying, K. and Tyshkovskiy, A. and Trapp, A. and Liu, H. and Moqri, M. and Kerepesi, C. and Gladyshev, V.N.},
  journal={bioRxiv},
  year={2023},
  doi={10.1101/2023.02.28.530532}
}
```
