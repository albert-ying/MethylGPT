#!/usr/bin/env python3
"""Package ages against the analysis-code ages for the six tutorial profiles.

Usage: python tests/check_tutorial_agreement.py <bundle-dir> [tolerance in years, default 0.01]
Reads examples/GSE54848_example_betas.csv (six public 450K profiles of GSE54848) and
examples/GSE54848_example_analysis_code_ages.csv (ages from the analysis code of the paper, GPU), prints both
with their differences and exits with status 1 if the largest absolute difference exceeds the tolerance.
"""
import sys
from pathlib import Path

import pandas as pd

from methylgpt2 import MethylGPT, __version__

here = Path(__file__).resolve().parents[1]
bundle = sys.argv[1]
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
betas = pd.read_csv(here / "examples/GSE54848_example_betas.csv", index_col=0)
ref = pd.read_csv(here / "examples/GSE54848_example_analysis_code_ages.csv").set_index("sample").analysis_code_age
model = MethylGPT.from_pretrained(bundle)
age = model.predict_age(betas)
tab = pd.DataFrame({"package_age": age, "analysis_code_age": ref.reindex(age.index)})
tab["difference"] = tab.package_age - tab.analysis_code_age
worst = float(tab.difference.abs().max())
print(f"methylgpt2 {__version__} on {model.device}; tolerance {tol} years")
print(tab.round(4).to_string())
print(f"largest absolute difference: {worst:.4f} years -> {'PASS' if worst <= tol else 'FAIL'}")
sys.exit(0 if worst <= tol else 1)
