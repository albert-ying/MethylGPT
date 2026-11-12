# MethylGPT

**MethylGPT: a foundation model for the DNA methylome**

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2024.10.30.621013v2) &nbsp;
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](https://github.com/albert-ying/MethylGPT/blob/main/LICENSE)

MethylGPT is a foundation model that learns the co-methylation structure of the human DNA methylome without labels from 169,010 public DNA methylation profiles (145,700 for training) and reads any subset of its 49,156 CpGs, from 27K, 450K, EPIC v1 or EPIC v2 arrays. Trained further on age, sex, tissue and 25 published clock outputs, it returns imputed values, latent scores, sample embeddings, age, sex, tissue and the distilled clock outputs.

## Code and weights

[`methylgpt2/`](methylgpt2/) holds the `methylgpt2` package (version 1.1.2) with its tutorial and test, the analysis code of the paper, and the vocabulary and split tables; its README explains how to install and run the package. The weights (bundle `methylgpt2-1.1`) are attached to the release [methylgpt2-v1.1.2](https://github.com/albert-ying/MethylGPT/releases/tag/methylgpt2-v1.1.2).

## Data

Training data are available from GEO, the [EWAS Data Hub](https://bigd.big.ac.cn/ewas/datahub) and [ClockBase](https://clockbase.org); `methylgpt2/release_tables/` gives each profile's identifier, dataset group and split. The corpus was assembled from these preprocessed files:

- Preprocessed methylation matrices (type3): [Download](https://www.dropbox.com/scl/fi/bbs6sxlkpbx11rhyvdfto/processed_type3_parquet_shuffled.tar.gz?rlkey=s73utmumq6xldmv3y6kh9bz75&st=8pslwy2a&dl=0)
- CpG probe IDs (type3): [Download](https://www.dropbox.com/scl/fi/2n6bx7j8v0aon0kwfsghp/probe_ids_type3.csv?rlkey=ly133xlce1xxjiku6tiski6qq&st=pig4e41h&dl=0)

The same data can be loaded from the LaminDB instance https://lamin.ai/laminlabs/methyldata.

## Earlier version

The code of the model described in the 2024 preprint, the `methylgpt` package on PyPI, is kept at two tags: [methylgpt-original-2025-08-05](https://github.com/albert-ying/MethylGPT/tree/methylgpt-original-2025-08-05), the repository as of 5 August 2025, and [methylgpt-original-2026-03-05](https://github.com/albert-ying/MethylGPT/tree/methylgpt-original-2026-03-05), its last version, with tutorials, documentation and links to the pretrained models.

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
