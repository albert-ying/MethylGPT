"""Write examples/quickstart.ipynb (run once; the notebook is executed by tests/run_quickstart.sh)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# MethylGPT quickstart\n\n"
       "Predict age and 25 full-array clock outputs, convert a 27K-array profile to the full 49,156-CpG vocabulary, "
       "and compute sample embeddings. The example file holds six public 450K profiles from GSE54848 "
       "(dermal fibroblasts and iPSCs after three passages); none of them were used to train MethylGPT.\n\n"
       "Set `BUNDLE` to the unpacked model bundle (`methylgpt2-1.1`). Predicted age is the mean of three training runs. "
       "A GPU is used when available; CPU also works."),
    code("import os\nimport numpy as np\nimport pandas as pd\nfrom methylgpt2 import MethylGPT\n\n"
         "BUNDLE = os.environ.get('METHYLGPT_BUNDLE', 'methylgpt2-1.1')\n"
         "model = MethylGPT.from_pretrained(BUNDLE)\n"
         "print(model.device, len(model.vocab), 'vocabulary CpGs')"),
    md("## Input\nRows are samples, columns are CpG probe IDs, values are beta values in [0, 1]. "
       "Missing CpGs can be absent or NaN; no imputation is needed beforehand."),
    code("betas = pd.read_csv('GSE54848_example_betas.csv', index_col=0)\nbetas.iloc[:, :5]"),
    md("## Age, sex, tissue and distilled clocks"),
    code("pred = model.predict(betas)\npred[['vocab_cpgs_observed', 'age', 'p_male', 'tissue', 'GrimAge2', 'DunedinPACE', 'PhenoAge']].round(2)"),
    md("## Array conversion: keep only CpGs on the 27K array, impute the rest\n"
       "The 27K array covers about 45% of the vocabulary. Here the other CpGs are removed and then predicted, "
       "and the predictions are compared with the measured values."),
    code("on27k = set(pd.read_csv('hm27_vocab_cpgs.txt', header=None)[0])\n"
         "reduced = betas[[c for c in betas.columns if c in on27k]]\n"
         "full = model.impute(reduced)\n"
         "held = [c for c in betas.columns if c not in on27k]\n"
         "err = (full[held] - betas[held]).abs()\n"
         "print('kept', reduced.shape[1], 'CpGs; predicted', len(held), 'measured CpGs; mean absolute error',\n"
         "      round(float(np.nanmean(err.to_numpy())), 4))"),
    code("pd.DataFrame({'age, all CpGs': pred.age, 'age, 27K CpGs only': model.predict_age(reduced)}).round(1)"),
    md("## Reconstruction and residuals\n`reconstruct()` returns the factor model's reconstruction of every vocabulary CpG, including the measured ones; "
       "`residuals()` returns the measured value minus this reconstruction (NaN where a CpG is not measured). "
       "`impute()`, used above, keeps measured values and fills only the others."),
    code("rec = model.reconstruct(betas)\nres = model.residuals(betas)\n"
         "print(rec.shape, res.shape)\n"
         "print('mean absolute residual at measured CpGs:', round(float(np.nanmean(np.abs(res.to_numpy()))), 4))\n"
         "print('unmeasured CpGs in the first profile:', int(res.iloc[0].isna().sum()))"),
    md("## Sample embeddings\nA 2,048-dimensional representation (standardised factor scores plus pooled transformer latents) "
       "that is the input of the prediction heads; use it for downstream models, and `factor_scores()` (below) for clustering."),
    code("emb = model.embed(betas)\nprint(emb.shape)\nnp.round(np.corrcoef(emb.to_numpy())[:3, :3], 3)"),
    md("## Factor scores and the features of the paper's disease models\n"
       "`factor_scores()` gives the 1,536 factor scores (use these for clustering and nearest-neighbour search). "
       "`embed()` returns the embedding of the released encoder; the disease models of the paper were fitted on the embedding of an earlier "
       "checkpoint, returned by `embed(stage='age_stage')`, and `disease_features()` returns their complete feature set "
       "(that embedding, the 25 clock outputs in standardized units and the 49,156 factor-model residuals)."),
    code("fs = model.factor_scores(betas)\nE = model.embed(betas, stage='age_stage')\nF = model.disease_features(betas)\n"
         "print(fs.shape, E.shape, F.shape)\n"
         "print('correlation of the two embeddings, first profile:', round(float(np.corrcoef(E.iloc[0], emb.iloc[0])[0, 1]), 3))"),
]
nbf.write(nb, "quickstart.ipynb")
