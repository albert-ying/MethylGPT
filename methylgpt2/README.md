# MethylGPT 2

This folder holds the rebuilt MethylGPT model: the `methylgpt2` inference package with its tutorial and test, the analysis and plotting code behind the results of the paper, and the tables that define the model's CpG vocabulary and the dataset split of the training corpus. The results correspond to package version 1.1.2 and weight bundle `methylgpt2-1.1`.

| Path | Contents |
|---|---|
| `package/` | The `methylgpt2` package (imputation, sample embeddings, latent scores, age, sex, tissue and 25 distilled clock outputs), its README, the tutorial notebook with an executed copy, and the fresh-environment test with its log |
| `*.py`, `*.sh` | Training, evaluation and analysis scripts as run for the paper |
| `figures/` | Plotting scripts of the figures |
| `diagnostics_2026-09-30/` | Reconstruction baselines loaded by the reconstruction evaluations |
| `release_tables/` | The CpG vocabulary with its selection criteria and annotation, the dataset-level split of the 169,120 corpus profiles, and a column dictionary (`release_tables/README.md`) |
| `upstream/` | `artifacts/external/GSE194154/scripts/03_build_matrix.py`, which builds the RRBS beta-value matrix of GSE194154 on the vocabulary CpGs |

## Install and run

```bash
cd methylgpt2/package
pip install -e .                 # numpy, pandas, torch
pip install -e ".[examples]"     # plus jupyter and matplotlib for the tutorial
```

The weights are attached to the release [methylgpt2-v1.1.2](https://github.com/albert-ying/MethylGPT/releases/tag/methylgpt2-v1.1.2) of this repository. Download its eight files (`config.json`, `vocab.txt`, `core.npz`, `weights.pt`, `weights_run2.pt`, `weights_run3.pt`, `weights_age_stage.pt` and `checksums.sha256`) into one folder named `methylgpt2-1.1` and check them with `shasum -a 256 -c checksums.sha256` inside that folder; `package/methylgpt2-1.1.checksums.sha256` holds the same checksums.

```python
import pandas as pd
from methylgpt2 import MethylGPT

model = MethylGPT.from_pretrained("methylgpt2-1.1")    # path to the bundle folder; GPU if available
betas = pd.read_csv("betas.csv", index_col=0)          # samples x CpG IDs, beta values in [0, 1]
pred = model.predict(betas)                            # age, sex, tissue and the distilled clock outputs
```

`package/README.md` lists every call and which checkpoint produced which result in the paper. `package/tests/run_quickstart.sh <bundle> <new venv>` installs the package with CPU-only PyTorch in a new environment, runs the tutorial on six public profiles of GSE54848 and compares the predicted ages with those of the analysis code (largest difference 0.002 years; the test fails above 0.01 years; `package/tests/quickstart_fresh_env_2026-10-09.log` is the log of that run). Run on a GPU, `package/tests/check_tutorial_agreement.py` gives a largest difference of 0.004 years.

The analysis scripts import the package (install it first) and contain the data and output paths of the server on which they were run; change these to your copies of the data. Scripts that evaluate the originally submitted model (`*original_model*`, `si_reconstruction_diagnostics_2026-10-18.py`, `diagnostics_2026-09-30/`) load that model and its code from the server; the code of that model is kept at the tags `methylgpt-original-2025-08-05` and `methylgpt-original-2026-03-05` of this repository.

## Data

Training data are available from GEO, the EWAS Data Hub and ClockBase; `release_tables/methylgpt2_profile_split_2026-10-03.csv.gz` gives each profile's identifier, dataset group and split. GSE308517, GSE54848 and GSE194154 are available from GEO. Generation Scotland data are available on application to the Generation Scotland access committee; the analysis code for Generation Scotland is included, its data are not.

## License

Apache 2.0, as for the rest of this repository (`LICENSE`).
