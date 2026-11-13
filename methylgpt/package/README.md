# methylgpt

Inference package for MethylGPT, a trained model of human DNA methylation profiles (49,156 CpGs).

## Install

```bash
pip install -e .                 # numpy, pandas, torch
pip install -e ".[examples]"     # plus jupyter and matplotlib for the notebook
```

Download the files of the model bundle `methylgpt-1.1` from the release
[v1.1.2](https://github.com/albert-ying/MethylGPT/releases/tag/v1.1.2) into one folder named
`methylgpt-1.1`: config.json, vocab.txt, core.npz, weights.pt, weights_run2.pt and weights_run3.pt for the age
ensemble, weights_age_stage.pt for the disease embedding of the paper, and checksums.sha256
(`shasum -a 256 -c checksums.sha256` inside the folder checks them).

In the paper, the factor model (`core.npz`) is the probabilistic input layer, and its factor scores are the latent
scores.

## Use

```python
import pandas as pd
from methylgpt import MethylGPT

model = MethylGPT.from_pretrained("methylgpt-1.1")       # GPU if available, otherwise CPU
betas = pd.read_csv("betas.csv", index_col=0)            # samples x CpG IDs, beta values in [0, 1]

pred = model.predict(betas)       # age (years), p_male, tissue, 25 distilled clock outputs
runs = model.predict_age_runs(betas)  # age (years) from each of the three training runs
full = model.impute(betas)        # all 49,156 vocabulary CpGs; observed values unchanged, the rest from the factor model
emb = model.embed(betas)          # 2,048-dimensional sample embedding (input to the heads)
fs = model.factor_scores(betas)   # 1,536 factor scores; use these for clustering and visualization
```

Predicted age (`predict()['age']`, `predict_age()`) is the mean of the ages predicted by three training
runs of the model (seeds 20261001-20261003) that share the same factor model core; `predict_age_runs()`
returns each run's prediction. Sex, tissue, the clock outputs and embeddings come from the first run; factor
scores, reconstruction and imputation come from the factor model, which the three runs share.

### Which checkpoint produced which result in the paper

| Feature in the paper | Checkpoint | Call |
|---|---|---|
| Age (all age results) | mean of the three runs, each after the clock-distillation stage (`weights.pt`, `weights_run2.pt`, `weights_run3.pt`) | `model.predict_age(betas)` |
| Sex, tissue and the 25 distilled clock outputs (including the distilled-clock disease model) | run 1 after the clock-distillation stage (`weights.pt`) | `model.predict(betas)`, `model.predict_clocks(betas)` |
| Embedding of the disease models (development folds, test and validation subsets) | run 1 at the end of the age, sex and tissue stage, before clock distillation (`weights_age_stage.pt`) | `model.embed(betas, stage="age_stage")` |
| Factor scores and reconstruction | factor model (`core.npz`), no encoder | `model.factor_scores(betas)`; `model.reconstruct(betas)` returns the reconstruction at every CpG; `model.impute(betas)` keeps observed values and fills the rest |
| Residuals of the residual disease model | factor model (`core.npz`), no encoder | `model.residuals(betas)`: observed value minus `reconstruct`, NaN where unobserved |
| All features of the residual disease model, in the column order of the fitted models | the three rows above | `model.disease_features(betas)` |

`model.embed(betas)` without `stage` returns the embedding of the released encoder (run 1 after clock
distillation), which is the input of the prediction heads. The disease models in the paper were fitted on
the earlier `age_stage` embedding, so use `stage="age_stage"` to reproduce them:

```python
E = model.embed(betas, stage="age_stage")   # the embedding used for the disease models in the paper
R = model.residuals(betas)                  # observed beta minus the factor-model reconstruction; NaN if unobserved
F = model.disease_features(betas)           # [embedding (2,048), 25 standardized clock outputs, 49,156 residuals]
```

`impute()` returns observed values unchanged, so subtracting it from the input gives zero at observed CpGs;
use `residuals()` (or `reconstruct()`) for the factor-model residuals. The Cox models of the paper standardize
each feature with the mean and standard deviation of the fitting participants, set unobserved residuals to the
fitting mean, and add age and sex as unpenalized covariates.

Inputs may contain any subset of CpGs from 27K, 450K, EPIC v1 or EPIC v2 arrays. Columns outside the
vocabulary are ignored and missing vocabulary CpGs are left unobserved; do not mean-impute before calling
the model. M-values must be converted to beta values first.

The distilled clock outputs (GrimAge2 and its components, DunedinPACE, PhenoAge, Horvath, Hannum,
DNAmTL, McCartney BMI, smoking and alcohol scores, and others) approximate the published clocks computed
from full arrays; they are not replacements for clocks measured on complete data.

To apply an existing linear clock to profiles that lack some of its CpGs, fill them with
`model.impute(betas)` first (the factor model's reconstruction; `method="transformer"` adds the encoder's correction,
which is not more accurate). In the MethylGPT evaluation, an elastic-net age model applied this way had a
median absolute error of 3.39 years with 90% of CpGs removed (MethylGPT 3.25 years; 19.35 years when the
missing CpGs were filled with training means).

`examples/quickstart.ipynb` runs end to end on six public GSE54848 profiles.
The SHA-256 checksums of the files of bundle methylgpt-1.1 are in `methylgpt-1.1.checksums.sha256` (also inside the bundle as `checksums.sha256`).

Example data (in `examples/`, public):

- `GSE54848_example_betas.csv`: six 450K profiles of GEO series GSE54848 (GSM1345997, GSM1345998, GSM1345999,
  GSM1346021, GSM1346022 and GSM1346023; dermal fibroblasts at day 0 and iPSCs after three passages),
  restricted to the vocabulary CpGs, with the beta values of the GEO series matrix (48,999 of the 49,020
  vocabulary CpGs of the 450K array are in the series matrix; 25 to 60 values per profile are missing as deposited). To rebuild it, download
  the series matrix of GSE54848 from GEO and keep these six samples and the CpGs listed in the bundle's
  `vocab.txt`.
- `hm27_vocab_cpgs.txt`: the 22,231 vocabulary CpGs that the 27K array (GPL8490) shares with the EPIC array, the
  set used for the 27K scenarios of the paper (the 27K array carries 56 further vocabulary CpGs).
- `GSE54848_example_analysis_code_ages.csv`: the ages that the analysis code of the paper gives for the six
  profiles (mean of the three training runs, computed on GPU).

`tests/check_tutorial_agreement.py <bundle>` compares the package's ages for the six profiles with the
analysis-code ages and fails if any differs by more than 0.01 years (largest difference 0.004 years on GPU and
0.002 years on CPU with package 1.1.2).

To reproduce the disease embedding of the paper bit for bit on GPU, set
`torch.backends.cuda.matmul.allow_tf32 = True` before calling `model.embed(betas, stage="age_stage")`, as the
analysis code did; without it the values differ by at most 0.04 (mean 0.0006).
`tests/run_quickstart.sh` creates a fresh virtual environment, installs the package and executes the
notebook.
