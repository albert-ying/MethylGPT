#!/usr/bin/env python3
"""Large-gain analysis E: covariate model with BMI and smoking in Generation Scotland.

Written on 24 Oct 2026 before running (large-gain analysis plan of 23 Oct 2026, analysis E). This
analysis is reported whatever the rule says.

Pre-specified design (as in the plan; runs only if the variables exist)
  Participants, endpoints and outer folds: the locked nested protocol of gs_clinical_covariates_2026-10-03.py (9,866
  development participants, 8 endpoints, 5 outer folds). Covariate model: unpenalized Cox model of age, sex, BMI and
  smoking (pack-years, or ever/current smoking if pack-years are absent), standardized with fitting-fold statistics and
  missing values filled with fitting-fold medians, fitted on the four fitting folds and scored on the held-out fold.
  Methylation risks (MethylGPT combined model; raw-CpG models; GrimAge2 + DunedinPACE) are added as in that script (the
  equal-weight sum of within-fold standardized risks), for complete profiles and for 27K-array CpGs (out-of-fold risks of
  the hm27 scenario in artifacts/gs_cox/scenario_combined_2026-10-01 and private_clock_scenario_risks_2026-10-01). C-index
  pooled within outer folds; gains above the covariate model with 1,000 paired bootstrap draws of participants within
  folds and Benjamini-Hochberg adjustment, as in gs_clinical_covariates_2026-10-03.py.
  Primary metric: mean over the eight endpoints of the C-index gain above the covariate model (dC). Comparators: the
  raw-CpG models and GrimAge2 + DunedinPACE; best comparator = the one with the largest dC.
  Decision rule (plan, "Decision rule", discrimination): (dC_ours - dC_best) / dC_best >= 10% and the 95% interval of
  dC_ours - dC_best excludes zero. Main-figure candidacy only for the 27K result; otherwise Supplementary Information.

Data availability check (this script, --stage check)
  Before any model is fitted, the header of every Generation Scotland release file on Chronos is searched for BMI and
  smoking variables (case-insensitive pattern below): meta.csv and 2024-06-03_phenotypes.csv of the GS project folder,
  the member list of gs.tar.1.gz and the headers of its tabular members (age.csv, 2024-06-03_phenotypes_lastfollow202310.csv;
  the batch files are M-value matrices whose header holds sample IDs only, checked for the first batch member), the
  extracted copies in /Entropy/kying/gs_release/gs/, and the frame-cache label files of the
  locked protocol. If no BMI or no smoking variable is found, the analysis is not run and the outputs state
  "Cannot proceed - required data file not found: [files searched]". DNA methylation surrogates (the McCartney BMI and
  smoking scores, GrimAge2's pack-years component) are not substitutes for measured covariates and are not used.
Amendment 24 Oct 2026 (before any result): the first check run stopped because pigz is not installed on Chronos; the
  archive is now decompressed with gzip -dc. Nothing else changed.
Outputs (aggregate only): artifacts/large_gain_2026-10-24/E/{E_data_availability.csv, E_rule_outcome.csv, README.md}
Run: python3 large_gain_E_bmi_smoking_covariates_2026-10-24.py --stage check
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from methylgpt_lib import ARTIFACTS, log

OUT = ARTIFACTS / "large_gain_2026-10-24/E"
GSDIR = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT")
TAR = GSDIR / "gs.tar.1.gz"
EXTRACTED = Path("/Entropy/kying/gs_release/gs")
CACHE = GSDIR / "task_a_private_training_20260629/_frame_cache"
BMI = re.compile(r"bmi|body.?mass|weight|height", re.I)
SMOKE = re.compile(r"smok|pack.?y|cigar|tobacco", re.I)


def header(path):
    with open(path) as f:
        return [c.strip().strip('"') for c in f.readline().rstrip("\n").split(",")]


def tar_stream(member, nbytes=1 << 16):
    """First bytes of one tar member (streamed; nothing is written to disk)."""
    cmd = f"gzip -dc '{TAR}' | tar -xf - -O '{member}' | head -c {nbytes}"
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True).stdout


def row(path, kind, cols, note=""):
    b = [c for c in cols if BMI.search(c)]
    s = [c for c in cols if SMOKE.search(c)]
    return {"file": str(path), "kind": kind, "n_columns_or_members": len(cols), "bmi_matches": ";".join(b),
            "smoking_matches": ";".join(s), "bmi_found": bool(b), "smoking_found": bool(s), "note": note}


def stage_check(a):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for p in (GSDIR / "meta.csv", GSDIR / "2024-06-03_phenotypes.csv"):
        rows.append(row(p, "csv header", header(p)))
    lst = subprocess.run(["bash", "-o", "pipefail", "-c", f"gzip -dc '{TAR}' | tar -tf -"], capture_output=True, text=True)
    members = [m for m in lst.stdout.split("\n") if m]
    if lst.returncode != 0 or not members:
        raise SystemExit(f"could not list {TAR}: {lst.stderr[-500:]}")
    rows.append(row(TAR, "tar member list", members, note="members: " + "; ".join(m for m in members if not Path(m).name.startswith("."))))
    for m in members:
        name = Path(m).name
        if name.startswith(".") or m.endswith("/"):
            continue
        if m.endswith(".csv"):
            first = tar_stream(m).decode(errors="replace").split("\n")[0]
            rows.append(row(f"{TAR}::{m}", "csv header (tar member)", [c.strip().strip('"') for c in first.split(",")]))
    first_batch = next(m for m in members if m.startswith("gs/batches/split_file_") and m.endswith(".csv.gz"))
    raw = subprocess.run(["bash", "-c", f"gzip -dc '{TAR}' | tar -xf - -O '{first_batch}' | zcat | head -1"],
                         capture_output=True).stdout.decode(errors="replace").strip()
    fields = [c.strip().strip('"') for c in raw.split(",")]
    sample_like = sum(bool(re.fullmatch(r"\d+_R\d+C\d+", c)) for c in fields)
    rows.append({**row(f"{TAR}::{first_batch}", "methylation batch header", []), "n_columns_or_members": len(fields),
                 "note": f"{sample_like} of {len(fields)} header fields are array sample IDs (M-value matrix); other batches share the format"})
    for p in sorted(EXTRACTED.glob("*.csv")):
        if p.name in ("age.csv", "2024-06-03_phenotypes_lastfollow202310.csv"):
            rows.append(row(p, "csv header (extracted copy of tar member)", header(p)))
    for p in sorted(CACHE.glob("COMMON_SAMPLE_SPLIT_*_all_gs_batches_all_probes_*.labels.parquet")):
        rows.append(row(p, "parquet columns", list(pd.read_parquet(p).columns)))
    av = pd.DataFrame(rows)
    av.to_csv(OUT / "E_data_availability.csv", index=False)
    bmi_ok, smk_ok = bool(av.bmi_found.any()), bool(av.smoking_found.any())
    log(f"searched {len(av)} files or member lists: BMI found {bmi_ok}, smoking found {smk_ok} ({time.time() - t0:.0f} s)")
    searched = ", ".join(sorted(set(av.file.str.split("::").str[0])))
    msg = f"Cannot proceed - required data file not found: {searched}"
    if bmi_ok and smk_ok:
        raise SystemExit("BMI and smoking variables were found; the fitting stage must be written and committed before running")
    missing = " and ".join(v for v, ok in (("BMI", bmi_ok), ("smoking", smk_ok)) if not ok)
    rule = []
    for s in ("complete profiles", "27K-array CpGs"):
        rule.append({"analysis": "E", "setting": s, "role": "main-figure candidate if it passes" if s.startswith("27K") else "Supplementary Information",
                     "primary_metric": "mean C-index gain above the age, sex, BMI and smoking covariate model (dC)",
                     "methylgpt_value": None, "best_comparator": None, "best_comparator_value": None, "relative_gain": None,
                     "difference_ci95_low": None, "difference_ci95_high": None, "outcome": f"not run: {missing} not available",
                     "pass": None, "status": msg})
    pd.DataFrame(rule).to_csv(OUT / "E_rule_outcome.csv", index=False)
    lines = ["# E. Covariate model with BMI and smoking (Generation Scotland)", "",
             f"Source script: `{Path(__file__).name}` (stage check). The pre-specified design and rule are in its docstring.", "",
             f"{msg}", "",
             ("Neither BMI nor smoking (pack-years, ever or current smoking)" if not (bmi_ok or smk_ok) else f"No {missing} variable")
             + " is recorded in any Generation Scotland release file on Chronos, so no covariate model with BMI and "
             "smoking was fitted and no C-index gain was computed. Methylation-based surrogates (McCartney BMI and smoking scores, "
             "GrimAge2 pack-years) were not used as substitutes. The Methods of the manuscript already state that body mass index "
             "was not available in the release used.", "",
             "| File | Contents and columns | Rows | Stage |", "|---|---|---:|---|",
             f"| `E_data_availability.csv` | One row per file or member list searched: file, kind, n_columns_or_members, bmi_matches, "
             f"smoking_matches, bmi_found, smoking_found, note (search pattern BMI `{BMI.pattern}`, smoking `{SMOKE.pattern}`, "
             f"case-insensitive, on column names and member names only) | {len(av)} | check |",
             f"| `E_rule_outcome.csv` | Rule-outcome table (not computed): analysis, setting, role, primary_metric, methylgpt_value, "
             f"best_comparator, best_comparator_value, relative_gain, difference_ci95_low, difference_ci95_high, outcome, pass, status "
             f"| {len(rule)} | check |"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    log(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["check"], required=True)
    a = ap.parse_args()
    stage_check(a)


if __name__ == "__main__":
    main()
