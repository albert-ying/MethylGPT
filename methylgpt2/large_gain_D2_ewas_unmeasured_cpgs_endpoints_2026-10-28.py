#!/usr/bin/env python3
"""Large-gain analysis D2: disease EWAS at CpGs a reduced array does not measure, over the other seven incident endpoints
of the locked Generation Scotland protocol (extension of analysis D).

Written on 28 Oct 2026 from the analysis brief for D2 and the large-gain analysis plan of 23 Oct 2026 (Decision
rule; analysis D), BEFORE anything that produces a result of D2 is run. Every result is reported, whichever method it
favours. Analysis D (large_gain_D_ewas_unmeasured_cpgs_2026-10-24.py; artifacts/large_gain_2026-10-24/D) ran this design
for incident type 2 diabetes only (27K setting: Spearman 0.666 for MethylGPT against 0.311 for nearest neighbours). D2
asks whether that holds for the other endpoints of Fig. 6.

What is D's, unchanged
  D's script is imported (importlib) and its functions are used as they are: load_gs (participants, matrix, type 2
  diabetes and its cross-checks), load_sets (D's frozen CpG sets, artifacts/large_gain_2026-10-24/D/D_cpg_sets.csv.gz),
  CoxBatch (estimator), compare and zfill (primary metric and the undefined-z rule), hits, recall_precision and bh
  (secondary hits), accuracy (imputation accuracy), pct (percentile interval), boot_p (bootstrap P), sha256_file. The
  orchestration of D's fit stage (point estimates, estimator checks, bootstrap loop) is repeated line by line for each
  endpoint, because D's stage function writes to D's own directory. Run setting as D's reported run: 12,000 CpGs per
  batched Cox fit and NUMPY_MADVISE_HUGEPAGE=0 (speed and memory only).

Data and inclusion
  Participants: D's 9,866 GS development participants (training subset of the locked incident-disease protocol; frame-
  cache matrix COMMON_SAMPLE_SPLIT_full18859_common_task_a_incident_positive_duration_20260629_train_..._matrix.npz; beta
  values, 49,156 vocabulary columns; 43,899 observed in every participant). Nobody is excluded.
  Endpoints (the eight of the locked protocol used in Fig. 6):
    primary set (7): Autoimmune_Diseases, Cancers, Cardiovascular_Diseases, Kidney_Diseases, Neurological_Diseases,
                     Respiratory_Diseases, date_of_death (death from any cause).
    check (1):       Endocrine_and_Metabolic_Diseases (incident type 2 diabetes, D's endpoint). It is recomputed and must
                     reproduce D's primary values (27K setting: rho 0.6663619916 for MethylGPT and 0.3112852187 for nearest
                     neighbours, D_point_estimates.json) within 1e-6; otherwise the fit stage stops. It is not part of the
                     primary metric; it enters the secondary eight-endpoint mean.
  Event (0/1), follow-up (duration, days from the baseline month to first diagnosis, death or 1 October 2023; for
  date_of_death to death or 1 October 2023), age (years) and sex (1 = female) of each endpoint come from the locked
  endpoint table D used, artifacts/gs_cox/mgpt2_combined_2026-10-01/private_nested_predictions.parquet, rows of that
  endpoint with age_adjustment "linear", aligned to the frame-cache participant order exactly as D's load_gs does.
  At-risk set (locked protocol, artifacts/gs_cox/gs_endpoint_definitions_common_split_2026-06-29_audit.json, label mode
  incident_positive_duration): for every endpoint, participants with a prevalent or same-month event or a non-positive
  duration were excluded, and the common analysis set (16,444) is the intersection of the remaining participants across
  all eight endpoints; the 9,866 development participants are its training split. Each endpoint's at-risk set is
  therefore all 9,866 development participants.
  Checks at the freeze (the stage stops if any fails): for each endpoint, D's checks (the table rows are exactly the
  frame-cache participants, unique; age and sex equal to the frame-cache labels; events 0/1; durations > 0); event and
  duration equal to the protocol's split label table (revision_server_20260622/.../regenerated_survival_split_label_tables/
  survial_<endpoint>_full18859_common_task_a_incident_positive_duration_20260629_split_labels.parquet, split "train");
  age and sex identical across endpoints; type 2 diabetes arrays identical to D's load_gs; participant SHA-256 equal to
  D's frozen analysis set; D's CpG-set counts equal to D's frozen table.
Freezing of the endpoint table
  --stage freeze writes D2_frozen_endpoints.csv (one row per endpoint: role, at-risk rule, participant, event and
  censored counts, sex counts, age and follow-up summaries, SHA-256 of the sorted participant IDs and of the event and
  duration vectors; no identifiers) and D2_frozen_endpoints.json (inputs, checksums, checks). Both are committed before
  any fit.

Settings, target CpGs and versions: D's. hm27 (PRIMARY): keep the vocabulary CpGs of the 27K array (21,577 observed in
  GS); targets = the 22,322 GS-observed vocabulary CpGs not kept. hm450 (secondary): keep "hm450" (43,767); 132 targets.
  Versions of the target CpGs: full (measured EPIC values, the reference), MethylGPT (released bundle methylgpt2-1.1,
  MethylGPT.impute, method "factor"), mean filling (training-split CpG means, core mu; constant, z undefined), nearest
  neighbours (k = 10 against the 145,700 corpus training-split profiles, the reconstruction benchmark's kNN).
Imputed values
  D's private imputations (artifacts/large_gain_2026-10-24_private/D/imputed_{hm27,hm450}.npz; participant level, never
  committed) are reused after --stage verify checks them against D's records: the release bundle checksums
  (package/methylgpt2-1.1.checksums.sha256); participant SHA-256 and target indices stored in each file equal to D's
  frozen sets; shapes (9,866 x targets) and finite values; mean-filling values equal to the bundle's core mu at the
  targets; the fraction of MethylGPT values identical to the measured values equal to D_impute_record.json; and every
  imputation-accuracy metric recomputed with D's accuracy() equal to D_imputation_accuracy.csv (relative tolerance
  1e-12). File SHA-256 and modification times are recorded. If any check fails, the analysis stops; recomputation with
  D's impute code would then be added in a dated amendment committed before its output is used.

EWAS estimator (D's CoxBatch, unchanged): per CpG, Cox proportional hazards model of the endpoint on the CpG value, age
  and sex (Breslow ties), CpG, age and sex standardized over the participants (weighted in bootstrap draws), batched
  Newton-Raphson in float64 on the GPU from 0 with step halving, stop when the largest step < 1e-9 or after 50
  iterations; not converged if the last step > 1e-6 or the estimate is not finite (z undefined); CpGs with SD < 1e-8 have
  undefined z. beta = log HR per SD, se from the inverse observed information, Wald z, two-sided normal p.
  Checks per endpoint (D's, reported): statsmodels PHReg(ties="breslow") on D's 30 check CpGs (25 drawn with
  default_rng(20261024) from the 27K targets plus the endpoint's 5 largest full-data |z| among them); frequency weights
  against row duplication for one draw (default_rng(20261025)).

Primary metric
  rho_e(v) = Spearman correlation over the hm27 target CpGs between the per-CpG z of version v and the full-data z, for
  endpoint e, with D's undefined-z rule (undefined z set to 0 in every version; a version whose z are all undefined, i.e.
  mean filling, has correlation 0 by rule). PRIMARY: rho_bar(v) = mean of rho_e(v) over the seven primary endpoints.
Comparators: mean filling (rho_bar = 0 by rule) and nearest neighbours. Best comparator as in D: the comparator with the
  higher rho_bar, so nearest neighbours unless its rho_bar is <= 0 (if it is exactly 0, nearest neighbours).
Decision rule (plan, "Decision rule"; agreement metrics use the relative improvement):
  relative gain = (rho_bar_MethylGPT - rho_bar_best) / rho_bar_best.
  PASS iff the relative gain is >= 0.10 and the lower end of the 95% interval of rho_bar_MethylGPT - rho_bar_best is > 0;
  otherwise FAIL. If rho_bar_best <= 0 the relative gain is undefined and the rule is reported as not applicable (fail).
Interval: D's participant bootstrap: 1,000 draws; numpy default_rng(20261024); draw b uses the b-th call of
  rng.integers(0, 9866, 9866) on the frame-cache participant order (bincount = frequency weights), which is D's draw b.
  The same resample is shared by every endpoint, setting and version in a draw ("h" + a 16-hex SHA-256 prefix of each draw's
  weight vector is stored per endpoint and checked to be identical across endpoints). In each draw the full-data,
  MethylGPT and nearest-neighbour EWAS are refitted at the target CpGs for every endpoint, rho_e recomputed, and the
  difference of the endpoint means formed within the draw. Imputed values are not recomputed (as D). Percentile 95%
  intervals; bootstrap P = min(1, 2(k + 1)/(B + 1)) (D's boot_p).

Secondary (all reported)
  (a) Each endpoint separately (all eight): rho_e for MethylGPT and nearest neighbours with intervals, difference with
      interval and bootstrap P, relative gain with interval, and the rule applied to that endpoint (descriptive).
  (b) The mean over all eight endpoints including type 2 diabetes (rho_bar_8), with the same rule quantities.
  (c) hm450: the same quantities (seven-endpoint mean, eight-endpoint mean, each endpoint).
  (d) Recall of full-data hits per endpoint and setting (D's secondary a): hits are BH q < 0.05 over the setting's target
      CpGs; recall = fraction of full-data hits that are hits of the imputed version with the same sign (mean filling 0;
      undefined without full-data hits); also hits, precision, recall difference, with bootstrap intervals (D's caveat
      applies: hit-based bootstrap intervals are not sampling intervals of the point estimates).
  (e) Pearson correlation of z per endpoint and version (D's secondary b), with intervals.
  (f) Imputation accuracy table as D (D's accuracy() on the reused imputations; equal to D's table by the verify check).
  (g) Full-data EWAS over all 43,899 GS CpGs per endpoint: number of BH q < 0.05 hits (descriptive, D's secondary e).
  (h) Type 2 diabetes reproduction check: D2 against D for rho, Pearson r and recall per setting and version, per-CpG z
      (max abs difference against D's per-CpG table, written to 6 significant digits), and draw-level rho against D's
      bootstrap draws.
Seeds: bootstrap 20261024; check CpGs default_rng(20261024); duplication check default_rng(20261025). Resamples: 1,000.

Stages
  --stage freeze   (CPU) frozen endpoint table; commit before any fit.
  --stage verify   (CPU) verification of D's private imputations and the imputation-accuracy table.
  --stage fit      (GPU) spawns one worker per device (--devices cuda:0,cuda:1); each worker takes a fixed contiguous
                   share of the endpoint order (type 2 diabetes first) and, per endpoint, writes the point estimates and
                   checks, then the 1,000 bootstrap draws with a checkpoint every 10 draws. Done-markers per endpoint;
                   a rerun resumes from the markers and the checkpoint (stored weight hashes are re-verified). The split
                   of endpoints over devices changes speed only.
  --stage analyze  (CPU) assembles the per-CpG and draw tables, primary and secondary results, rule outcome, README and a
                   privacy check (no participant identifier or participant-level value in any output file).
Outputs (aggregate and per-CpG summary statistics only): artifacts/large_gain_2026-10-28/D2/ with README.md.
Participant-level and working files (checkpoints, logs): artifacts/large_gain_2026-10-28_private/D2/ (git-ignored, never
committed, stay on Chronos); D's imputations are read in place from artifacts/large_gain_2026-10-24_private/D/.
Run: python3 large_gain_D2_ewas_unmeasured_cpgs_endpoints_2026-10-28.py --stage {freeze,verify,fit,analyze}
AMENDMENT 28 Oct 2026, execution only, before any fit: the first freeze run stopped at the privacy check of the frozen
  table, because the header test matched the substring "participant_id" in the column sha256_sorted_participant_ids (a
  SHA-256 of the sorted IDs, D's column). The token and barcode scans found nothing. The header test now flags only columns
  named exactly dnam_id, participant_id, sample_id, id or ids. No data, estimator, metric or rule changes; the freeze stage
  is rerun.
AMENDMENT 30 Oct 2026, execution only, after the fit stage and before any analysis output existed: the analysis stage stopped
at its first read (BadGzipFile). atomic_csv writes to <name>.tmp, so pandas inferred no compression and the working
files per_cpg_*.csv.gz and full_*.csv.gz are plain CSV under a .gz name. The analysis stage now reads them with
compression=None; no fit, estimate or design changes, and the public output tables are written compressed as designed.

AMENDMENT 28 Oct 2026, documentation only, while the fit stage ran and before the analysis stage: the README gains notes
  computed from the outputs (type 2 diabetes reproduction, endpoints without full-data hits, estimator checks, run
  settings, amendments) and a sentence that only the primary row carries the decision. No analysis change.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, log, require, write_json

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("large_gain_D", HERE / "large_gain_D_ewas_unmeasured_cpgs_2026-10-24.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

OUT = ARTIFACTS / "large_gain_2026-10-28/D2"
PRIV = ARTIFACTS / "large_gain_2026-10-28_private/D2"
WORK = PRIV / "work"
LOGS = PRIV / "logs"
D_OUT, D_PRIV = D.OUT, D.PRIV
AUDIT = ARTIFACTS / "gs_cox/gs_endpoint_definitions_common_split_2026-06-29_audit.json"
LABELS = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_server_20260622/server_results/task_a/results/"
              "regenerated_survival_split_label_tables")
PRIMARY_ENDPOINTS = ("Autoimmune_Diseases", "Cancers", "Cardiovascular_Diseases", "Kidney_Diseases", "Neurological_Diseases",
                     "Respiratory_Diseases", "date_of_death")
CHECK_ENDPOINT = "Endocrine_and_Metabolic_Diseases"
ALL_ENDPOINTS = (CHECK_ENDPOINT,) + PRIMARY_ENDPOINTS
# execution order over devices (type 2 diabetes first so the reproduction check runs early); speed only
FIT_ORDER = (CHECK_ENDPOINT, "Autoimmune_Diseases", "Cancers", "Cardiovascular_Diseases",
             "Kidney_Diseases", "Neurological_Diseases", "Respiratory_Diseases", "date_of_death")
SETTINGS, PRIMARY, SEED, N_BOOT, FDR = D.SETTINGS, D.PRIMARY, D.SEED, D.N_BOOT, D.FDR
REPRO_TOL = 1e-6
CKPT_EVERY = 10
AT_RISK_RULE = ("all 9,866 development participants: the locked protocol (label mode incident_positive_duration) excluded, per "
                "endpoint, prevalent or same-month events and non-positive durations, and kept the intersection over all eight "
                "endpoints (16,444; development = its 9,866 training participants), so every development participant is at "
                "risk for every endpoint")
RULE = ("PASS iff (rho_bar_MethylGPT - rho_bar_best) / rho_bar_best >= 0.10 and the lower end of the 95% bootstrap interval "
        "of rho_bar_MethylGPT - rho_bar_best is > 0; rho_bar = mean over the seven primary endpoints of the Spearman "
        "correlation of per-CpG Cox z with the full-data z at the hm27 target CpGs")
ID_PATTERN = re.compile(r"\d{12}_R\d{2}C\d{2}")
ID_COLUMNS = {"dnam_id", "participant_id", "sample_id", "id", "ids"}


def atomic_csv(df, path, **kw):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    write_json(tmp, obj)
    os.replace(tmp, path)


def sha_ids(ids):
    return hashlib.sha256("\n".join(np.sort(np.asarray(ids, str))).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
def load_endpoints(g):
    """Event, duration, age and sex of every endpoint, aligned to D's participant order, with D's checks per endpoint."""
    ids = g["ids"]
    lab = pd.read_parquet(require(Path(g["labels_path"])))
    if not np.array_equal(lab.dnam_id.astype(str).to_numpy(), ids):
        raise ValueError("frame-cache label order differs from load_gs")
    ep_all = pd.read_parquet(require(D.ENDPOINT_TABLE))
    out = {}
    for e in ALL_ENDPOINTS:
        ep = ep_all[(ep_all.age_adjustment == "linear") & (ep_all.endpoint == e)].copy()      # as D.load_gs
        ep["dnam_id"] = ep.dnam_id.astype(str)
        if len(ep) != len(ids) or ep.dnam_id.nunique() != len(ids) or set(ep.dnam_id) != set(ids):
            raise ValueError(f"{e}: endpoint table and frame-cache participants differ")
        ep = ep.set_index("dnam_id").loc[ids]
        checks = {"age_equal_to_frame_cache_labels": bool(np.array_equal(ep.age.to_numpy(float), lab.age.to_numpy(float))),
                  "sex_equal_to_frame_cache_labels (1 = F)": bool(np.array_equal(ep.sex.to_numpy(float),
                                                                                 (lab.sex.astype(str) == "F").to_numpy(float)))}
        ev, du = ep.event.to_numpy(float), ep.duration.to_numpy(float)
        if not np.isin(ev, (0, 1)).all() or (du <= 0).any():
            raise ValueError(f"{e}: events must be 0/1 and durations positive")
        sp = LABELS / f"survial_{e}_{D.SPLIT}_split_labels.parquet"
        sl = pd.read_parquet(require(sp), filters=[("split", "==", "train")])
        sl["dnam_id"] = sl.dnam_id.astype(str)
        if len(sl) != len(ids) or sl.dnam_id.nunique() != len(ids) or set(sl.dnam_id) != set(ids):
            raise ValueError(f"{e}: protocol split label table (train) participants differ")
        sl = sl.set_index("dnam_id").loc[ids]
        checks["event_equal_to_protocol_split_label_table"] = bool(np.array_equal(ev, sl.event.to_numpy(float)))
        checks["duration_equal_to_protocol_split_label_table"] = bool(np.array_equal(du, sl.duration.to_numpy(float)))
        if not all(checks.values()):
            raise ValueError(f"{e}: cross-check failed: {checks}")
        out[e] = {"event": ev, "duration": du, "age": ep.age.to_numpy(float), "sex": ep.sex.to_numpy(float), "checks": checks,
                  "split_label_table": str(sp)}
    t = out[CHECK_ENDPOINT]
    same = {k: bool(np.array_equal(t[k], g[k])) for k in ("event", "duration", "age", "sex")}
    if not all(same.values()):
        raise ValueError(f"type 2 diabetes arrays differ from D's load_gs: {same}")
    across = {k: all(np.array_equal(out[e][k], t[k]) for e in ALL_ENDPOINTS) for k in ("age", "sex")}
    if not all(across.values()):
        raise ValueError(f"age or sex differ across endpoints: {across}")
    return out, {"t2d_equal_to_D_load_gs": same, "age_sex_identical_across_endpoints": across}


def load_imputations(sets, sha):
    """D's private imputed values at the target CpGs (as D's stage_fit reads them)."""
    imp = {}
    for s in SETTINGS:
        z = np.load(require(D_PRIV / f"imputed_{s}.npz"))
        if str(z["sha256_sorted_participant_ids"]) != sha or not np.array_equal(z["target_idx"], np.flatnonzero(sets[f"target_{s}"])):
            raise ValueError(f"imputed file for {s} does not match the frozen sets")
        imp[s] = {k: z[k] for k in ("methylgpt", "nearest_neighbours", "mean_filling")}
    return imp


# ---------------------------------------------------------------------------------------------------------------------
# privacy check of output files
# ---------------------------------------------------------------------------------------------------------------------
def privacy_check(files, ids):
    import gzip
    idset = set(map(str, ids))
    rows = []
    for p in files:
        raw = gzip.open(p, "rt").read() if p.name.endswith(".gz") else p.read_text()
        toks = set(re.split(r"[,\s\"'{}\[\]:;|`()]+", raw))
        n_id = len(toks & idset)
        n_pat = len(ID_PATTERN.findall(raw))
        header = [c.strip().strip('"') for c in raw.split("\n", 1)[0].split(",")] if p.suffix != ".json" else []
        has_id_col = bool(set(header) & ID_COLUMNS)
        rows.append({"file": p.name, "bytes": p.stat().st_size, "n_participant_ids_found": n_id,
                     "n_barcode_pattern_matches": n_pat, "identifier_column_in_header": has_id_col})
    ok = all(r["n_participant_ids_found"] == 0 and r["n_barcode_pattern_matches"] == 0 and not r["identifier_column_in_header"]
             for r in rows)
    return {"files": rows, "passed": ok,
            "method": "every token of every output file compared with the 9,866 development participant IDs; regex search for the "
                      "GS barcode pattern (12 digits, _R##C##); header scanned for identifier columns. Tables are per CpG, per "
                      "endpoint or per bootstrap draw; none has one row per participant."}


# ---------------------------------------------------------------------------------------------------------------------
# stage 1: freeze
# ---------------------------------------------------------------------------------------------------------------------
def stage_freeze(a):
    OUT.mkdir(parents=True, exist_ok=True)
    g = D.load_gs()
    E, cross = load_endpoints(g)
    ids = g["ids"]
    sha = sha_ids(ids)
    dfz = pd.read_csv(require(D_OUT / "D_frozen_analysis_set.csv")).set_index("setting")
    sets = D.load_sets()
    dchk = {"participant_sha256_equal_to_D_frozen": bool(all(dfz.sha256_sorted_participant_ids == sha)),
            "n_participants_equal_to_D_frozen": bool(all(dfz.n_participants == len(ids)))}
    for s in SETTINGS:
        dchk[f"{s}_kept_and_target_counts_equal_to_D_frozen"] = bool(int(sets[f"kept_{s}"].sum()) == int(dfz.loc[s, "n_kept_cpgs"])
                                                                   and int(sets[f"target_{s}"].sum()) == int(dfz.loc[s, "n_target_cpgs"]))
    if not all(dchk.values()):
        raise ValueError(f"D's frozen set checks failed: {dchk}")
    audit = json.loads(require(AUDIT).read_text())
    order = np.argsort(ids)
    rows = []
    for e in ALL_ENDPOINTS:
        x = E[e]
        ev, du = x["event"], x["duration"]
        lsum = audit["label_summaries"][e]
        rows.append({"endpoint": e, "endpoint_label": lsum["endpoint_label"],
                     "role": "primary set" if e in PRIMARY_ENDPOINTS else "check (D's endpoint; secondary eight-endpoint mean only)",
                     "in_primary_mean": e in PRIMARY_ENDPOINTS,
                     "constituent_disease_columns": lsum["mapped_disease_column_count"],
                     "at_risk_rule": AT_RISK_RULE,
                     "n_participants": int(len(ids)), "n_events": int(ev.sum()), "n_censored": int((1 - ev).sum()),
                     "event_fraction": float(ev.mean()),
                     "n_female": int(x["sex"].sum()), "n_male": int((1 - x["sex"]).sum()),
                     "age_mean": float(x["age"].mean()), "age_sd": float(x["age"].std(ddof=1)),
                     "followup_days_median": float(np.median(du)), "followup_days_q25": float(np.percentile(du, 25)),
                     "followup_days_q75": float(np.percentile(du, 75)),
                     "n_excluded_prevalent_or_same_month_full_cohort": int(lsum["prevalent_or_same_month_count"]),
                     "n_target_cpgs_hm27": int(sets["target_hm27"].sum()), "n_target_cpgs_hm450": int(sets["target_hm450"].sum()),
                     "sha256_sorted_participant_ids": sha,
                     "sha256_event_duration_in_sorted_id_order": hashlib.sha256(
                         np.ascontiguousarray(np.column_stack([ev[order], du[order]])).tobytes()).hexdigest()})
        log(f"{e}: {rows[-1]['n_participants']} participants, {rows[-1]['n_events']} events ({rows[-1]['role']})")
    atomic_csv(pd.DataFrame(rows), OUT / "D2_frozen_endpoints.csv")
    atomic_json(OUT / "D2_frozen_endpoints.json", {
        "script": Path(__file__).name, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"matrix": g["matrix_path"], "matrix_mtime": time.ctime(Path(g["matrix_path"]).stat().st_mtime),
                   "labels": g["labels_path"], "endpoint_table": str(D.ENDPOINT_TABLE),
                   "endpoint_table_sha256": D.sha256_file(D.ENDPOINT_TABLE),
                   "endpoint_table_mtime": time.ctime(D.ENDPOINT_TABLE.stat().st_mtime),
                   "endpoint_definition_audit": str(AUDIT), "endpoint_definition_audit_sha256": D.sha256_file(AUDIT),
                   "split_label_tables": {e: E[e]["split_label_table"] for e in ALL_ENDPOINTS},
                   "split_label_tables_sha256": {e: D.sha256_file(E[e]["split_label_table"]) for e in ALL_ENDPOINTS},
                   "D_cpg_sets": str(D_OUT / "D_cpg_sets.csv.gz"), "D_cpg_sets_sha256": D.sha256_file(D_OUT / "D_cpg_sets.csv.gz"),
                   "D_frozen_analysis_set": str(D_OUT / "D_frozen_analysis_set.csv")},
        "at_risk_rule": AT_RISK_RULE, "incident_only_policy_of_the_protocol": audit["incident_only_policy"],
        "checks": {"per_endpoint": {e: E[e]["checks"] for e in ALL_ENDPOINTS}, **cross, "D_frozen": dchk,
                   "D_load_gs_checks": g["checks"]},
        "endpoints": rows})
    pc = privacy_check([OUT / "D2_frozen_endpoints.csv", OUT / "D2_frozen_endpoints.json"], ids)
    atomic_json(PRIV / "privacy_check_freeze.json", pc)
    if not pc["passed"]:
        raise ValueError("privacy check of the frozen table failed")
    log("frozen endpoint table written; privacy check passed")


# ---------------------------------------------------------------------------------------------------------------------
# stage 2: verification of D's private imputations (CPU)
# ---------------------------------------------------------------------------------------------------------------------
def stage_verify(a):
    t0 = time.time()
    want = dict(reversed(l.split()) for l in require(D.CHECKSUMS).read_text().splitlines() if l.strip())
    bad = sorted(f for f in want if D.sha256_file(D.BUNDLE / f) != want[f])
    if bad:
        sys.exit(f"release bundle checksum mismatch: {bad}")
    log("release bundle checksums verified")
    mu = np.load(require(D.BUNDLE / "core.npz"))["mu"].astype(np.float32)
    g = D.load_gs()
    X = g["X"]
    sha = sha_ids(g["ids"])
    sets = D.load_sets()
    dacc = pd.read_csv(require(D_OUT / "D_imputation_accuracy.csv"))
    drec = json.loads(require(D_OUT / "D_impute_record.json").read_text())
    dfit_mtime = (D_OUT / "D_fit_record.json").stat().st_mtime
    info = {"bundle_checksums_verified": sorted(want), "settings": {}}
    acc = []
    ok_all = True
    for s in SETTINGS:
        p = require(D_PRIV / f"imputed_{s}.npz")
        z = np.load(p)
        tgt = sets[f"target_{s}"].to_numpy(bool)
        nt = int(tgt.sum())
        mg, kn, mf = z["methylgpt"], z["nearest_neighbours"], z["mean_filling"]
        truth = np.ascontiguousarray(X[:, tgt])
        c = {"participant_sha256_equal_to_frozen": str(z["sha256_sorted_participant_ids"]) == sha,
             "target_idx_equal_to_D_cpg_sets": bool(np.array_equal(z["target_idx"], np.flatnonzero(tgt))),
             "shapes": bool(mg.shape == (len(X), nt) and kn.shape == (len(X), nt) and mf.shape == (nt,)),
             "finite": bool(np.isfinite(mg).all() and np.isfinite(kn).all() and np.isfinite(mf).all()),
             "methylgpt_within_0_1": bool((mg >= 0).all() and (mg <= 1).all()),
             "mean_filling_equal_to_bundle_core_mu": bool(np.array_equal(mf, mu[tgt])),
             "methylgpt_identical_to_measured_fraction_equal_to_D_record":
                 float((mg == truth).mean()) == float(drec["settings"][s]["methylgpt_identical_to_measured_fraction"]),
             "file_written_before_D_fit_record": bool(p.stat().st_mtime < dfit_mtime)}
        diffs = {}
        for name, pr in (("MethylGPT", mg), ("nearest neighbours", kn), ("mean filling", mf[None, :])):
            r = {"setting": s, "version": name, "n_participants": len(X), "n_target_cpgs": nt, **D.accuracy(pr, truth)}
            dref = dacc[(dacc.setting == s) & (dacc.version == name)].iloc[0]
            eq = True
            for m in ("mse", "mae", "within_cpg_r2_pooled", "median_per_cpg_pearson_r", "n_cpgs_with_defined_r", "n_values"):
                v, w = r[m], dref[m]
                if v is None or (isinstance(w, float) and np.isnan(w)):
                    same = (v is None or (isinstance(v, float) and np.isnan(v))) and pd.isna(w)
                else:
                    same = bool(np.isclose(float(v), float(w), rtol=1e-12, atol=0))
                eq &= same
                diffs[f"{name}:{m}"] = None if (v is None or pd.isna(w)) else float(abs(float(v) - float(w)))
            r["equal_to_D_imputation_accuracy"] = bool(eq)
            c[f"accuracy_{name}_equal_to_D"] = bool(eq)
            acc.append(r)
            log(f"{s} {name}: MSE {r['mse']:.6g}; equal to D's table: {eq}")
        info["settings"][s] = {"file": str(p), "file_sha256": D.sha256_file(p), "file_mtime": time.ctime(p.stat().st_mtime),
                               "checks": c, "abs_diff_to_D_accuracy": diffs}
        ok_all &= all(c.values())
        del z, mg, kn, truth
    info["all_checks_passed"] = bool(ok_all)
    info["elapsed_s"] = round(time.time() - t0, 1)
    pd.DataFrame(acc).to_csv(OUT / "D2_imputation_accuracy.csv", index=False)
    atomic_json(OUT / "D2_imputation_verification.json", info)
    if not ok_all:
        sys.exit("verification of D's private imputations failed; see D2_imputation_verification.json (stop; amendment needed)")
    log(f"D's private imputations verified ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------------------------------------------------------------
# stage 3: fits (GPU workers)
# ---------------------------------------------------------------------------------------------------------------------
def wait_for_gpu(torch, dev, min_free_gb):
    while True:
        free, total = torch.cuda.mem_get_info(dev)
        if free / 2**30 >= min_free_gb:
            log(f"GPU {dev}: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
            return
        log(f"GPU {dev}: only {free / 2**30:.1f} GiB free (< {min_free_gb}); waiting 60 s")
        time.sleep(60)


def point_stage(e, cox, x, X, sets, imp, a, dev):
    """D's point estimates and estimator checks for one endpoint (D.stage_fit, first part)."""
    import torch
    from scipy.stats import norm
    from statsmodels.duration.hazard_regression import PHReg   # as D (statsmodels.api fails to import with scipy 1.16)
    t0 = time.time()
    n = len(X)
    vocab = sets.cpg_id.astype(str).to_numpy()
    obs = sets.observed_in_gs.to_numpy(bool)
    w1 = cox.weights()
    obs_idx = np.flatnonzero(obs)
    Mfull = cox.sort(X[:, obs_idx])
    bf, sf, zf, cf, sdf = cox.fit_all(Mfull, w1, a.batch)
    del Mfull
    torch.cuda.empty_cache()
    log(f"{e}: full-data EWAS of {len(obs_idx)} CpGs in {time.time() - t0:.0f} s; not converged {int((~cf).sum())}")
    full = pd.DataFrame({"cpg_id": vocab[obs_idx], "beta_log_hr_per_sd": bf, "se": sf, "z": zf,
                         "p": np.where(np.isfinite(zf), 2 * norm.sf(np.abs(zf)), np.nan), "sd_beta_value": sdf, "converged": cf})
    full["q_bh_all_gs_cpgs"] = D.bh(full.p.to_numpy())
    for s in SETTINGS:
        full[f"target_{s}"] = sets.set_index("cpg_id").loc[full.cpg_id, f"target_{s}"].to_numpy(bool)
    pos = {c: i for i, c in enumerate(full.cpg_id)}
    checks = {}
    rng = np.random.default_rng(SEED)
    tg = np.flatnonzero(sets["target_hm27"].to_numpy(bool))
    pick = list(rng.choice(tg, 25, replace=False))
    top = full.loc[full.target_hm27].assign(a=lambda d: d.z.abs()).sort_values("a", ascending=False).cpg_id.head(5)
    pick += [int(np.flatnonzero(vocab == c)[0]) for c in top]
    dz, db, dse = [], [], []
    for j in pick:
        exog = np.column_stack([X[:, j], x["age"], x["sex"]]).astype(float)
        exog = (exog - exog.mean(0)) / exog.std(0, ddof=1)
        r = PHReg(x["duration"], exog, status=x["event"].astype(int), ties="breslow").fit()
        k = pos[vocab[j]]
        dz.append(abs(r.params[0] / r.bse[0] - zf[k])); db.append(abs(r.params[0] - bf[k])); dse.append(abs(r.bse[0] - sf[k]))
    checks["statsmodels_phreg_30_cpgs"] = {"max_abs_diff_z": float(np.nanmax(dz)), "max_abs_diff_beta": float(np.nanmax(db)),
                                           "max_abs_diff_se": float(np.nanmax(dse))}
    log(f"{e}: check against statsmodels PHReg (30 CpGs): max |dz| {np.nanmax(dz):.2e}")
    cnt = np.bincount(np.random.default_rng(SEED + 1).integers(0, n, n), minlength=n)
    cols = np.array(pick[:30])
    rep = np.repeat(np.arange(n), cnt)
    cox_dup = D.CoxBatch(x["duration"][rep], x["event"][rep], x["age"][rep], x["sex"][rep], dev)
    zw = cox.fit(cox.sort(X[:, cols]), cox.weights(cnt))[2]
    zd = cox_dup.fit(cox_dup.sort(X[:, cols][rep]), cox_dup.weights())[2]
    checks["weights_vs_row_duplication_30_cpgs_max_abs_diff_z"] = float(np.nanmax(np.abs(zw - zd)))
    log(f"{e}: check frequency weights against row duplication: max |dz| {checks['weights_vs_row_duplication_30_cpgs_max_abs_diff_z']:.2e}")
    del cox_dup
    point = {}
    for s in SETTINGS:
        tmask = sets[f"target_{s}"].to_numpy(bool)
        tid = vocab[tmask]
        fi = np.array([pos[c] for c in tid])
        df = pd.DataFrame({"endpoint": e, "cpg_id": tid, "setting": s})
        for col, v in (("beta", bf), ("se", sf), ("z", zf), ("sd_beta_value", sdf), ("converged", cf)):
            df[f"full_{col}"] = v[fi]
        df["full_p"] = np.where(np.isfinite(df.full_z), 2 * norm.sf(np.abs(df.full_z)), np.nan)
        df["full_q_bh_targets"] = D.bh(df.full_p.to_numpy())
        pt = {"n_target_cpgs": int(tmask.sum())}
        for name, key in (("MethylGPT", "methylgpt"), ("nearest neighbours", "nearest_neighbours")):
            Mg = cox.sort(imp[s][key])
            b_, s_, z_, c_, sd_ = cox.fit_all(Mg, w1, a.batch)
            del Mg
            tag = "methylgpt" if name == "MethylGPT" else "knn"
            df[f"{tag}_beta"], df[f"{tag}_se"], df[f"{tag}_z"], df[f"{tag}_sd_beta_value"], df[f"{tag}_converged"] = b_, s_, z_, sd_, c_
            df[f"{tag}_p"] = np.where(np.isfinite(z_), 2 * norm.sf(np.abs(z_)), np.nan)
            df[f"{tag}_q_bh_targets"] = D.bh(df[f"{tag}_p"].to_numpy())
            rho, r = D.compare(z_, df.full_z.to_numpy())
            pt[name] = {"spearman_rho": rho, "pearson_r": r, "n_not_converged": int((~c_).sum()),
                        **D.recall_precision(z_, df.full_z.to_numpy())}
        df["mean_filling_value"] = imp[s]["mean_filling"]
        df["mean_filling_z"] = np.nan
        pt["mean filling"] = {"spearman_rho": 0.0, "pearson_r": 0.0, "n_not_converged": int(tmask.sum()),
                              "note": "constant imputed values; z undefined at every CpG; correlation 0 and recall 0 by rule",
                              **{**D.recall_precision(np.full(tmask.sum(), np.nan), df.full_z.to_numpy()), "recall": 0.0}}
        pt["full_not_converged"] = int((~df.full_converged.astype(bool)).sum())
        point[s] = pt
        log(f"{e} {s}: rho MethylGPT {pt['MethylGPT']['spearman_rho']:.6f}, nearest neighbours {pt['nearest neighbours']['spearman_rho']:.6f}; "
            f"full-data hits {pt['MethylGPT']['n_hits_full']}")
        atomic_csv(df, WORK / f"per_cpg_{s}_{e}.csv.gz", float_format="%.6g")
    atomic_csv(full.assign(endpoint=e)[["endpoint"] + list(full.columns)], WORK / f"full_{e}.csv.gz", float_format="%.6g")
    rec = {"endpoint": e, "checks": checks, "point": point, "device": a.device, "batch": a.batch,
           "full_data_all_cpgs": {"n_cpgs": int(len(full)), "n_not_converged": int((~full.converged).sum()),
                                  "n_hits_bh_all_gs_cpgs": int((full.q_bh_all_gs_cpgs < FDR).sum())},
           "elapsed_s": round(time.time() - t0, 1)}
    if e == CHECK_ENDPOINT:
        dpe = json.loads(require(D_OUT / "D_point_estimates.json").read_text())["point"]
        repro = {}
        for s in SETTINGS:
            for name in ("MethylGPT", "nearest neighbours"):
                repro[f"{s}:{name}"] = {"D": dpe[s][name]["spearman_rho"], "D2": point[s][name]["spearman_rho"],
                                        "abs_diff": abs(dpe[s][name]["spearman_rho"] - point[s][name]["spearman_rho"])}
        rec["t2d_reproduction"] = repro
        rec["t2d_reproduction_passed_hm27"] = bool(all(repro[f"hm27:{v}"]["abs_diff"] <= REPRO_TOL for v in ("MethylGPT", "nearest neighbours")))
        atomic_json(WORK / f"point_{e}.json", rec)
        log(f"type 2 diabetes reproduction (hm27): " + "; ".join(f"{k} D {v['D']:.10f} D2 {v['D2']:.10f} |d| {v['abs_diff']:.2e}"
                                                             for k, v in repro.items()))
        if not rec["t2d_reproduction_passed_hm27"]:
            raise SystemExit(f"type 2 diabetes does not reproduce D's primary values within {REPRO_TOL}; stopping")
    else:
        atomic_json(WORK / f"point_{e}.json", rec)


def bootstrap_stage(e, cox, X, sets, imp, a):
    """D's participant bootstrap for one endpoint (D.stage_fit, second part), resumable."""
    import torch
    n = len(X)
    t0 = time.time()
    union = np.flatnonzero(np.logical_or.reduce([sets[f"target_{s}"].to_numpy(bool) for s in SETTINGS]))
    uni_pos = {j: i for i, j in enumerate(union)}
    sub = {s: np.array([uni_pos[j] for j in np.flatnonzero(sets[f"target_{s}"].to_numpy(bool))]) for s in SETTINGS}
    gpu = {s: {"MethylGPT": cox.sort(imp[s]["methylgpt"]), "nearest neighbours": cox.sort(imp[s]["nearest_neighbours"])}
           for s in SETTINGS}
    Mu = cox.sort(X[:, union])
    part = WORK / f"draws_{e}.csv"
    rows = pd.read_csv(part, dtype={"resample_sha256_16": str}).to_dict("records") if part.exists() else []
    stored = {}
    for r in rows:
        stored.setdefault(int(r["draw"]), set()).add(str(r["resample_sha256_16"]))
    k_done = len(stored)
    if sorted(stored) != list(range(k_done)) or any(len(rows_s) != k_done for rows_s in
                                                    [[r for r in rows if r["setting"] == s] for s in SETTINGS]):
        raise ValueError(f"{e}: checkpoint {part} is not a contiguous set of complete draws")
    if k_done:
        log(f"{e}: resuming bootstrap at draw {k_done}")
    rngb = np.random.default_rng(SEED)
    tb = time.time()
    n_new = 0
    for b in range(N_BOOT):
        cntb = np.bincount(rngb.integers(0, n, n), minlength=n)
        h = "h" + hashlib.sha256(cntb.astype(np.int64).tobytes()).hexdigest()[:16]     # prefix keeps it a string in CSV
        if b < k_done:
            if stored[b] != {h}:
                raise ValueError(f"{e}: stored resample hash of draw {b} differs from the regenerated one")
            continue
        w = cox.weights(cntb)
        zfu = cox.fit_all(Mu, w, a.batch)[2]
        for s in SETTINGS:
            zfs = zfu[sub[s]]
            row = {"endpoint": e, "setting": s, "draw": b, "resample_sha256_16": h, "full_not_converged": int((~np.isfinite(zfs)).sum())}
            for name in ("MethylGPT", "nearest neighbours"):
                zb = cox.fit_all(gpu[s][name], w, a.batch)[2]
                tag = "methylgpt" if name == "MethylGPT" else "knn"
                rho, r = D.compare(zb, zfs)
                rp = D.recall_precision(zb, zfs)
                row.update({f"{tag}_rho": rho, f"{tag}_pearson": r, f"{tag}_not_converged": int((~np.isfinite(zb)).sum()),
                            f"{tag}_recall": rp["recall"], f"{tag}_precision": rp["precision"], f"{tag}_n_hits": rp["n_hits_imputed"],
                            "n_hits_full": rp["n_hits_full"]})
            row["mean_filling_rho"] = 0.0
            rows.append(row)
        n_new += 1
        if (b + 1) % CKPT_EVERY == 0 or b == N_BOOT - 1:
            atomic_csv(pd.DataFrame(rows), part)
        if (b + 1) % 25 == 0 or n_new == 1:
            el = time.time() - tb
            last = [r for r in rows if r["setting"] == PRIMARY][-1]
            log(f"{e}: bootstrap draw {b + 1}/{N_BOOT} ({el / n_new:.1f} s per draw); hm27 rho MGPT {last['methylgpt_rho']:.3f} "
                f"kNN {last['knn_rho']:.3f}")
    del Mu, gpu
    torch.cuda.empty_cache()
    atomic_json(WORK / f"done_{e}.json", {"endpoint": e, "n_boot": N_BOOT, "seed": SEED, "device": a.device, "batch": a.batch,
                                          "draws_computed_this_run": n_new, "elapsed_s_this_run": round(time.time() - t0, 1),
                                          "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"{e}: bootstrap done ({n_new} draws this run, {time.time() - t0:.0f} s)")


def stage_fit_worker(a):
    import torch
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    WORK.mkdir(parents=True, exist_ok=True)
    endpoints = [e for e in a.endpoints.split(",") if e]
    if not set(endpoints) <= set(ALL_ENDPOINTS):
        raise ValueError(f"unknown endpoint in {endpoints}")
    ver = json.loads(require(OUT / "D2_imputation_verification.json").read_text())
    if not ver.get("all_checks_passed"):
        sys.exit("D's private imputations are not verified; run --stage verify")
    fr = pd.read_csv(require(OUT / "D2_frozen_endpoints.csv")).set_index("endpoint")
    g = D.load_gs()
    X = g["X"]
    sha = sha_ids(g["ids"])
    E, _ = load_endpoints(g)
    for e in endpoints:
        if int(E[e]["event"].sum()) != int(fr.loc[e, "n_events"]) or sha != fr.loc[e, "sha256_sorted_participant_ids"]:
            raise ValueError(f"{e}: data differ from the frozen endpoint table")
    sets = D.load_sets()
    imp = load_imputations(sets, sha)
    log(f"worker {a.device}: endpoints {endpoints}")
    for e in endpoints:
        if (WORK / f"done_{e}.json").exists():
            log(f"{e}: done-marker present; skipped")
            continue
        wait_for_gpu(torch, dev, a.min_free_gb)
        x = E[e]
        cox = D.CoxBatch(x["duration"], x["event"], x["age"], x["sex"], dev)
        if not (WORK / f"point_{e}.json").exists():
            point_stage(e, cox, x, X, sets, imp, a, dev)
        else:
            pe = json.loads((WORK / f"point_{e}.json").read_text())
            if e == CHECK_ENDPOINT and not pe.get("t2d_reproduction_passed_hm27"):
                raise SystemExit("type 2 diabetes reproduction failed earlier; stopping")
            log(f"{e}: point estimates present; skipped")
        bootstrap_stage(e, cox, X, sets, imp, a)
        del cox
        torch.cuda.empty_cache()
    log(f"worker {a.device} finished")


def stage_fit(a):
    """Spawn one worker per device over fixed contiguous shares of FIT_ORDER; stop all if one fails."""
    LOGS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    devs = [d for d in a.devices.split(",") if d]
    shares = np.array_split(np.array(FIT_ORDER), len(devs))
    procs = []
    for d, sh in zip(devs, shares):
        lp = LOGS / f"fit_{d.replace(':', '')}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "fit-worker", "--device", d, "--endpoints", ",".join(sh),
               "--batch", str(a.batch), "--min-free-gb", str(a.min_free_gb)]
        fh = open(lp, "a")
        procs.append((d, list(sh), subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE)), fh))
        log(f"started worker on {d} for {list(sh)} (log {lp})")
    failed = False
    while True:
        states = [(d, p.poll()) for d, _, p, _ in procs]
        if any(rc not in (None, 0) for _, rc in states):
            failed = True
            for d, _, p, _ in procs:
                if p.poll() is None:
                    log(f"terminating worker on {d} because another worker failed")
                    p.terminate()
            for _, _, p, _ in procs:
                p.wait()
            break
        if all(rc == 0 for _, rc in states):
            break
        time.sleep(30)
    for _, _, _, fh in procs:
        fh.close()
    for d, sh, p, _ in procs:
        log(f"worker {d} ({sh}) exit code {p.returncode}")
    if failed:
        sys.exit("a fit worker failed; see the logs")
    log("all fit workers finished")


# ---------------------------------------------------------------------------------------------------------------------
# stage 4: analysis and README
# ---------------------------------------------------------------------------------------------------------------------
def rule_quantities(rho_m, rho_k, dm, dk):
    """D's rule logic on point values rho_m, rho_k and draw vectors dm, dk (mean filling 0)."""
    comp = {"mean filling": 0.0, "nearest neighbours": rho_k}
    best = max(comp, key=comp.get) if rho_k != 0.0 else "nearest neighbours"
    rho_b = comp[best]
    db = dk if best == "nearest neighbours" else np.zeros_like(dk)
    diff = dm - db
    dci = D.pct(diff)
    if rho_b > 0:
        rel = (rho_m - rho_b) / rho_b
        rel_ci = D.pct((dm - db) / db)
        passed = bool(rel >= 0.10 and dci[0] is not None and dci[0] > 0)
        outcome = "pass" if passed else "fail"
    else:
        rel, rel_ci, passed, outcome = None, [None, None], False, "not applicable (rho_best <= 0)"
    return {"methylgpt_value": rho_m, "methylgpt_ci95_low": D.pct(dm)[0], "methylgpt_ci95_high": D.pct(dm)[1],
            "best_comparator": best, "best_comparator_value": rho_b, "best_comparator_ci95_low": D.pct(db)[0],
            "best_comparator_ci95_high": D.pct(db)[1],
            "other_comparator": "mean filling" if best != "mean filling" else "nearest neighbours",
            "other_comparator_value": 0.0 if best != "mean filling" else rho_k,
            "nearest_neighbours_value": rho_k, "nearest_neighbours_ci95_low": D.pct(dk)[0], "nearest_neighbours_ci95_high": D.pct(dk)[1],
            "difference": rho_m - rho_b, "difference_ci95_low": dci[0], "difference_ci95_high": dci[1],
            "bootstrap_p_difference": D.boot_p(diff), "relative_gain": rel, "relative_gain_ci95_low": rel_ci[0],
            "relative_gain_ci95_high": rel_ci[1], "outcome": outcome, "pass": passed, "n_bootstrap": int(len(dm))}


def stage_analyze(a):
    fr = pd.read_csv(require(OUT / "D2_frozen_endpoints.csv")).set_index("endpoint")
    for e in ALL_ENDPOINTS:
        require(WORK / f"done_{e}.json")
    pts = {e: json.loads(require(WORK / f"point_{e}.json").read_text()) for e in ALL_ENDPOINTS}
    dones = {e: json.loads(require(WORK / f"done_{e}.json").read_text()) for e in ALL_ENDPOINTS}
    # ---- assemble per-CpG, full-data and draw tables ------------------------------------------------------------------
    for s in SETTINGS:
        pc = pd.concat([pd.read_csv(require(WORK / f"per_cpg_{s}_{e}.csv.gz"), compression=None) for e in ALL_ENDPOINTS], ignore_index=True)
        pc.to_csv(OUT / f"D2_ewas_per_cpg_{s}.csv.gz", index=False, float_format="%.6g")
    fu = pd.concat([pd.read_csv(require(WORK / f"full_{e}.csv.gz"), compression=None) for e in ALL_ENDPOINTS], ignore_index=True)
    fu.to_csv(OUT / "D2_ewas_full_data_all_gs_cpgs.csv.gz", index=False, float_format="%.6g")
    dr = pd.concat([pd.read_csv(require(WORK / f"draws_{e}.csv"), dtype={"resample_sha256_16": str}) for e in ALL_ENDPOINTS],
                   ignore_index=True)
    for s in SETTINGS:
        dr[dr.setting == s].to_csv(OUT / f"D2_bootstrap_draws_{s}.csv", index=False)
    checks = {}
    hh = dr.groupby("draw").resample_sha256_16.nunique()
    checks["one_resample_per_draw_shared_by_all_endpoints_and_settings"] = bool((hh == 1).all() and len(hh) == N_BOOT)
    checks["draws_per_endpoint_and_setting"] = {f"{e}:{s}": int(((dr.endpoint == e) & (dr.setting == s)).sum())
                                                for e in ALL_ENDPOINTS for s in SETTINGS}
    if not checks["one_resample_per_draw_shared_by_all_endpoints_and_settings"] or \
            any(v != N_BOOT for v in checks["draws_per_endpoint_and_setting"].values()):
        raise ValueError(f"draw table check failed: {checks}")
    # ---- type 2 diabetes reproduction check --------------------------------------------------------------------------
    dpe = json.loads(require(D_OUT / "D_point_estimates.json").read_text())["point"]
    drule = pd.read_csv(require(D_OUT / "D_rule_outcome.csv")).set_index("setting")
    rep = []
    t2 = pts[CHECK_ENDPOINT]["point"]
    for s in SETTINGS:
        for name in ("MethylGPT", "nearest neighbours"):
            for m in ("spearman_rho", "pearson_r", "recall", "precision", "n_hits_full", "n_hits_imputed", "n_not_converged"):
                v1, v2 = dpe[s][name][m], t2[s][name][m]
                rep.append({"setting": s, "version": name, "quantity": m, "D": v1, "D2": v2,
                            "abs_diff": abs(v1 - v2) if (v1 is not None and v2 is not None) else None,
                            "tolerance": REPRO_TOL if m == "spearman_rho" else None,
                            "gating": bool(m == "spearman_rho" and s == PRIMARY),
                            "pass": bool(abs(v1 - v2) <= REPRO_TOL) if m == "spearman_rho" else None})
        dpc = pd.read_csv(require(D_OUT / f"D_ewas_per_cpg_{s}.csv.gz"))
        npc = pd.read_csv(require(WORK / f"per_cpg_{s}_{CHECK_ENDPOINT}.csv.gz"), compression=None)
        if not np.array_equal(dpc.cpg_id.to_numpy(str), npc.cpg_id.to_numpy(str)):
            raise ValueError("per-CpG CpG order differs from D's")
        for col in ("full_z", "methylgpt_z", "knn_z"):
            dd = np.abs(dpc[col].to_numpy(float) - npc[col].to_numpy(float))
            rep.append({"setting": s, "version": col.split("_")[0], "quantity": f"per-CpG {col}: max abs diff (both written to 6 "
                        "significant digits)", "D": None, "D2": None, "abs_diff": float(np.nanmax(dd)), "tolerance": None,
                        "gating": False, "pass": None})
        ddr = pd.read_csv(require(D_OUT / f"D_bootstrap_draws_{s}.csv")).set_index("draw")
        ndr = dr[(dr.endpoint == CHECK_ENDPOINT) & (dr.setting == s)].set_index("draw").loc[ddr.index]
        for col in ("methylgpt_rho", "knn_rho"):
            rep.append({"setting": s, "version": col.split("_")[0], "quantity": f"bootstrap draws {col}: max abs diff over 1,000 draws "
                        "(D's draws written to 6 significant digits)", "D": None, "D2": None,
                        "abs_diff": float(np.nanmax(np.abs(ddr[col].to_numpy(float) - ndr[col].to_numpy(float)))),
                        "tolerance": None, "gating": False, "pass": None})
    repdf = pd.DataFrame(rep)
    repdf.to_csv(OUT / "D2_t2d_reproduction_check.csv", index=False)
    checks["t2d_reproduction_hm27_passed"] = bool(repdf[repdf.gating]["pass"].all())
    # ---- per-endpoint quantities -------------------------------------------------------------------------------------
    per, mats = [], {}
    for s in SETTINGS:
        ds = dr[dr.setting == s]
        mats[s] = {k: ds.pivot(index="draw", columns="endpoint", values=k).loc[range(N_BOOT)] for k in ("methylgpt_rho", "knn_rho")}
        for e in ALL_ENDPOINTS:
            pt = pts[e]["point"][s]
            d = ds[ds.endpoint == e].set_index("draw").loc[range(N_BOOT)]
            q = rule_quantities(pt["MethylGPT"]["spearman_rho"], pt["nearest neighbours"]["spearman_rho"],
                                d.methylgpt_rho.to_numpy(float), d.knn_rho.to_numpy(float))
            row = {"analysis": "D2", "setting": s, "endpoint": e, "in_primary_mean": e in PRIMARY_ENDPOINTS,
                   "n_participants": int(fr.loc[e, "n_participants"]), "n_events": int(fr.loc[e, "n_events"]),
                   "n_target_cpgs": int(pt["n_target_cpgs"]), **q,
                   "rule_note": "rule applied to this endpoint alone (secondary, descriptive)"}
            for tag, name in (("methylgpt", "MethylGPT"), ("knn", "nearest neighbours")):
                p = pt[name]
                row[f"{tag}_pearson_r"] = p["pearson_r"]
                row[f"{tag}_pearson_ci95_low"], row[f"{tag}_pearson_ci95_high"] = D.pct(d[f"{tag}_pearson"])
                row[f"{tag}_recall"] = p["recall"]
                row[f"{tag}_recall_ci95_low"], row[f"{tag}_recall_ci95_high"] = D.pct(d[f"{tag}_recall"])
                row[f"{tag}_precision"] = p["precision"]
                row[f"{tag}_precision_ci95_low"], row[f"{tag}_precision_ci95_high"] = D.pct(d[f"{tag}_precision"])
                row[f"{tag}_n_hits"] = p["n_hits_imputed"]
                row[f"{tag}_n_hits_shared_same_sign"] = p["n_hits_shared_same_sign"]
                row[f"{tag}_n_not_converged"] = p["n_not_converged"]
                row[f"{tag}_rho_bootstrap_median"] = float(np.median(d[f"{tag}_rho"]))
            row["mean_filling_rho"], row["mean_filling_recall"] = 0.0, 0.0
            row["n_hits_full"] = pt["MethylGPT"]["n_hits_full"]
            row["n_hits_full_ci95_low"], row["n_hits_full_ci95_high"] = D.pct(d.n_hits_full)
            row["full_n_not_converged"] = pt["full_not_converged"]
            rd = (d.methylgpt_recall - d.knn_recall).to_numpy(float)
            row["recall_difference"] = (pt["MethylGPT"]["recall"] - pt["nearest neighbours"]["recall"]) if pt["MethylGPT"]["n_hits_full"] > 0 else None
            row["recall_difference_ci95_low"], row["recall_difference_ci95_high"] = D.pct(rd)
            row["recall_difference_bootstrap_p"] = D.boot_p(rd) if np.isfinite(rd).any() else None
            row["recall_difference_n_bootstrap_defined"] = int(np.isfinite(rd).sum())
            row["full_data_hits_bh_all_gs_cpgs"] = pts[e]["full_data_all_cpgs"]["n_hits_bh_all_gs_cpgs"]
            row["full_data_not_converged_all_gs_cpgs"] = pts[e]["full_data_all_cpgs"]["n_not_converged"]
            row["check_statsmodels_max_abs_diff_z"] = pts[e]["checks"]["statsmodels_phreg_30_cpgs"]["max_abs_diff_z"]
            row["check_weights_vs_duplication_max_abs_diff_z"] = pts[e]["checks"]["weights_vs_row_duplication_30_cpgs_max_abs_diff_z"]
            per.append(row)
    perdf = pd.DataFrame(per)
    perdf.to_csv(OUT / "D2_per_endpoint.csv", index=False)
    # ---- endpoint means (primary and secondary) ----------------------------------------------------------------------
    rule_rows = []
    for s in SETTINGS:
        for label, eps in (("seven primary endpoints", PRIMARY_ENDPOINTS), ("all eight endpoints (incl. type 2 diabetes)", ALL_ENDPOINTS)):
            rm = float(np.mean([pts[e]["point"][s]["MethylGPT"]["spearman_rho"] for e in eps]))
            rk = float(np.mean([pts[e]["point"][s]["nearest neighbours"]["spearman_rho"] for e in eps]))
            dm = mats[s]["methylgpt_rho"][list(eps)].mean(1, skipna=False).to_numpy(float)
            dk = mats[s]["knn_rho"][list(eps)].mean(1, skipna=False).to_numpy(float)
            q = rule_quantities(rm, rk, dm, dk)
            primary = (s == PRIMARY and len(eps) == 7)
            rule_rows.append({"analysis": "D2", "setting": s, "endpoint_set": label, "n_endpoints": len(eps),
                              "role": "PRIMARY" if primary else "secondary",
                              "primary_metric": "mean over endpoints of the Spearman rho of per-CpG Cox z (imputed vs full data) at "
                                                "the setting's target CpGs",
                              "n_participants": int(fr.n_participants.iloc[0]),
                              "n_events_total": int(sum(int(fr.loc[e, "n_events"]) for e in eps)),
                              "n_target_cpgs": int(pts[ALL_ENDPOINTS[0]]["point"][s]["n_target_cpgs"]), **q,
                              "n_endpoints_methylgpt_above_knn": int(sum(pts[e]["point"][s]["MethylGPT"]["spearman_rho"] >
                                                                         pts[e]["point"][s]["nearest neighbours"]["spearman_rho"] for e in eps)),
                              "interval": "percentile, 1,000 participant bootstrap draws (seed 20261024), one resample per draw shared "
                                          "by all endpoints and versions; difference of endpoint means within each draw",
                              "rule": RULE if primary else RULE.replace("seven primary endpoints", label).replace("hm27", s)})
    rt = pd.DataFrame(rule_rows)
    rt.to_csv(OUT / "D2_rule_outcome.csv", index=False)
    # ---- records ----------------------------------------------------------------------------------------------------
    write_json(OUT / "D2_point_estimates.json", {e: {"point": pts[e]["point"], "checks": pts[e]["checks"],
                                                     "full_data_all_cpgs": pts[e]["full_data_all_cpgs"]} for e in ALL_ENDPOINTS})
    write_json(OUT / "D2_fit_record.json", {"n_boot": N_BOOT, "seed": SEED, "checks": checks,
                                            "per_endpoint": {e: {"device": pts[e]["device"], "batch": pts[e]["batch"],
                                                                 "point_elapsed_s": pts[e]["elapsed_s"], "bootstrap": dones[e]}
                                                             for e in ALL_ENDPOINTS}})
    res = {"script": Path(__file__).name, "rule": RULE, "primary": rule_rows[0], "rule_rows": rule_rows, "checks": checks,
           "t2d_reproduction": rep}
    write_json(OUT / "D2_results.json", res)
    for r in rule_rows:
        log(f"{r['setting']} {r['endpoint_set']} ({r['role']}): rho_bar MethylGPT {r['methylgpt_value']:.4f} vs {r['best_comparator']} "
            f"{r['best_comparator_value']:.4f}; difference {r['difference']:.4f} [{r['difference_ci95_low']:.4f}, "
            f"{r['difference_ci95_high']:.4f}]; gain {r['relative_gain'] if r['relative_gain'] is None else round(r['relative_gain'], 4)}; {r['outcome']}")
    notes = readme_notes(pts, dones, repdf)
    write_readme(rt, perdf, notes)
    g_ids = pd.read_parquet(require(next(D.CACHE.glob(f"COMMON_SAMPLE_SPLIT_{D.SPLIT}_train_all_gs_batches_all_probes_*.labels.parquet"))),
                            columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "D2_privacy_check.json")
    pc = privacy_check(files, g_ids)
    write_json(OUT / "D2_privacy_check.json", pc)
    log(f"privacy check over {len(files)} files: {'passed' if pc['passed'] else 'FAILED'}")
    if not pc["passed"]:
        raise ValueError("privacy check failed; do not commit")
    write_readme(rt, perdf, notes)    # again, to list the privacy-check file


def readme_notes(pts, dones, repdf):
    """Data-driven README notes (documentation only)."""
    g = repdf[repdf.gating]
    rm = g[g.version == "MethylGPT"].abs_diff.iloc[0]
    rk = g[g.version == "nearest neighbours"].abs_diff.iloc[0]
    dmax = repdf[repdf.quantity.str.startswith("bootstrap draws")].abs_diff.max()
    zmax = repdf[repdf.quantity.str.startswith("per-CpG")].abs_diff.max()
    nohit = {s: [e for e in ALL_ENDPOINTS if pts[e]["point"][s]["MethylGPT"]["n_hits_full"] == 0] for s in SETTINGS}
    sm = max(pts[e]["checks"]["statsmodels_phreg_30_cpgs"]["max_abs_diff_z"] for e in ALL_ENDPOINTS)
    wd = max(pts[e]["checks"]["weights_vs_row_duplication_30_cpgs_max_abs_diff_z"] for e in ALL_ENDPOINTS)
    spd = {e: dones[e]["elapsed_s_this_run"] / max(dones[e]["draws_computed_this_run"], 1) for e in ALL_ENDPOINTS}
    return [
        f"- Type 2 diabetes reproduction (gating, hm27, tolerance 1e-6): |rho_D2 - rho_D| = {rm:.2e} for MethylGPT and {rk:.2e} for "
        f"nearest neighbours ({'passed' if bool(g['pass'].all()) else 'FAILED'}). Largest draw-level rho difference against D's "
        f"bootstrap draws {dmax:.2e} and largest per-CpG z difference {zmax:.2e} (D's tables are written to 6 significant digits).",
        "- Endpoints without full-data hits (BH q < 0.05 over the setting's targets), so recall is undefined there: "
        + "; ".join(f"{s}: {', '.join(nohit[s]) if nohit[s] else 'none'}" for s in SETTINGS) + ".",
        f"- Estimator checks over all endpoints: batched Cox z against statsmodels PHReg (Breslow) max |dz| {sm:.1e}; frequency "
        f"weights against row duplication max |dz| {wd:.1e}.",
        "- Run settings (speed and memory only): two GPU workers (" + "; ".join(f"{e} on {pts[e]['device']}" for e in ALL_ENDPOINTS)
        + f"), 12,000 CpGs per batched Cox fit, NUMPY_MADVISE_HUGEPAGE=0, in a transient systemd user unit; bootstrap "
        f"{min(spd.values()):.1f} to {max(spd.values()):.1f} s per draw and endpoint on the shared GPUs.",
        "- Amendment 28 Oct 2026 (execution only, before any fit): the privacy check's header test first matched the hashed column "
        "sha256_sorted_participant_ids; it now matches exact identifier column names. A second amendment (documentation only, "
        "before the analysis stage ran) added these notes.",
    ]


def write_readme(rt, perdf, notes=()):
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "README.md" and not p.name.endswith(".tmp"))

    def nrows(p):
        if p.suffix == ".json":
            return "JSON"
        return f"{len(pd.read_csv(p)):,}"

    desc = {
        "D2_frozen_endpoints.csv": ("Frozen endpoint table, one row per endpoint (role, at-risk rule, participant, event and censored counts, "
                                    "sex, age and follow-up summaries, SHA-256 of sorted participant IDs and of the event/duration "
                                    "vectors; no identifiers)", "--stage freeze"),
        "D2_frozen_endpoints.json": ("Same, with input paths, checksums, modification times and every freeze check", "--stage freeze"),
        "D2_imputation_verification.json": ("Verification of D's private imputations against D's records (bundle checksums, participant "
                                            "hash, target indices, mean filling = core mu, identical-value fraction, accuracy equal "
                                            "to D's table), file SHA-256 and times", "--stage verify"),
        "D2_imputation_accuracy.csv": ("Secondary (f): imputation accuracy at the target CpGs per setting and version (D's columns: mse, "
                                       "mae, within_cpg_r2_pooled, median_per_cpg_pearson_r, n_cpgs_with_defined_r, n_values) and "
                                       "equal_to_D_imputation_accuracy", "--stage verify"),
        "D2_ewas_full_data_all_gs_cpgs.csv.gz": ("Full-data EWAS at every GS CpG, per endpoint: endpoint, cpg_id, beta_log_hr_per_sd, se, z, "
                                                 "p, sd_beta_value, converged, q_bh_all_gs_cpgs (within endpoint), target_hm27, "
                                                 "target_hm450", "--stage fit"),
        "D2_ewas_per_cpg_hm27.csv.gz": ("Per endpoint and target CpG of the 27K setting (D's columns plus endpoint): full_*, methylgpt_* "
                                        "and knn_* (beta = log HR per SD, se, z, p, q_bh_targets within endpoint, sd_beta_value, "
                                        "converged), mean_filling_value, mean_filling_z (undefined)", "--stage fit"),
        "D2_ewas_per_cpg_hm450.csv.gz": ("Same for the 450K setting", "--stage fit"),
        "D2_bootstrap_draws_hm27.csv": ("One row per endpoint and participant-bootstrap draw (aggregate statistics only): resample hash, "
                                        "rho, Pearson r, recall, precision, hits and non-converged counts for MethylGPT and nearest "
                                        "neighbours, full-data hits", "--stage fit"),
        "D2_bootstrap_draws_hm450.csv": ("Same for the 450K setting", "--stage fit"),
        "D2_point_estimates.json": ("Point estimates per endpoint, setting and version, and the estimator checks per endpoint", "--stage fit"),
        "D2_fit_record.json": ("Seed, draws, devices, batch, timings and the draw-table checks", "--stage analyze"),
        "D2_t2d_reproduction_check.csv": ("Type 2 diabetes recomputed against D: rho (gating, tolerance 1e-6 at hm27), Pearson r, recall, "
                                          "precision, hits, non-converged counts, per-CpG z and draw-level rho differences",
                                          "--stage analyze"),
        "D2_per_endpoint.csv": ("Secondary (a, c, d, e, g): per setting and endpoint, rho for MethylGPT and nearest neighbours with "
                                "intervals, difference with interval and bootstrap P, relative gain with interval, rule outcome "
                                "for that endpoint alone (descriptive), Pearson r, recall, precision and hits with intervals, "
                                "recall difference, full-data hits over all GS CpGs, estimator checks", "--stage analyze"),
        "D2_rule_outcome.csv": ("Rule-outcome table: PRIMARY row (hm27, mean over the seven primary endpoints) and secondary rows "
                                "(hm27 eight-endpoint mean; hm450 seven- and eight-endpoint means): MethylGPT value, best comparator "
                                "and value, nearest-neighbour value, difference with 95% interval, relative gain, pass or fail",
                                "--stage analyze"),
        "D2_results.json": ("All results of the analysis stage", "--stage analyze"),
        "D2_privacy_check.json": ("Scan of every output file for participant identifiers (token match against the 9,866 IDs, "
                                  "barcode regex, identifier columns)", "--stage analyze"),
    }
    p0 = rt[rt.role == "PRIMARY"].iloc[0]
    lines = ["# D2. Disease EWAS at CpGs the reduced array does not measure, seven further GS endpoints", "",
             f"Source script: `{Path(__file__).name}` (stages freeze, verify, fit, analyze). Pre-specification, estimator, rule "
             "and seeds are in its docstring; D's functions are imported from `large_gain_D_ewas_unmeasured_cpgs_2026-10-24.py`.", "",
             "Inputs (on Chronos only): D's frame-cache matrix and labels, the locked endpoint table "
             "`artifacts/gs_cox/mgpt2_combined_2026-10-01/private_nested_predictions.parquet` (age_adjustment linear; eight endpoints), "
             "the endpoint audit `artifacts/gs_cox/gs_endpoint_definitions_common_split_2026-06-29_audit.json`, the protocol's split "
             "label tables (cross-check), D's frozen CpG sets (`../../large_gain_2026-10-24/D/D_cpg_sets.csv.gz`) and D's verified "
             "private imputations (`artifacts/large_gain_2026-10-24_private/D/`).", "",
             "Only aggregate tables and per-CpG summary statistics are here. Participant-level values (D's imputations, checkpoints, "
             "logs) stay on Chronos under `artifacts/large_gain_2026-10-24_private/D/` and `artifacts/large_gain_2026-10-28_private/D2/` "
             "and are not committed. `D2_privacy_check.json` records the identifier scan of every file here.", "",
             f"Primary (hm27, mean over the seven primary endpoints): MethylGPT {p0.methylgpt_value:.4f}, {p0.best_comparator} "
             f"{p0.best_comparator_value:.4f}; difference {p0.difference:.4f} [{p0.difference_ci95_low:.4f}, {p0.difference_ci95_high:.4f}]; "
             f"relative gain {p0.relative_gain if pd.isna(p0.relative_gain) else round(float(p0.relative_gain), 4)}; {str(p0.outcome).upper()}.", "",
             "Notes", "",
             "- Bootstrap intervals of hit-based metrics (hits, recall, precision) are not sampling intervals of the point estimates "
             "(D's README explains why: model-based standard errors ignore the resampling, so BH hits are more numerous in the draws).",
             "- Type 2 diabetes is D's endpoint, recomputed as a check (`D2_t2d_reproduction_check.csv`); it enters only the "
             "secondary eight-endpoint mean.",
             "- Rule outcomes of single endpoints and of the hm450 and eight-endpoint means are secondary and descriptive; only the "
             "PRIMARY row of `D2_rule_outcome.csv` carries the pre-specified decision.",
             *notes,
             "",
             "| File | Contents and columns | Rows | Size (bytes) | Stage |", "|---|---|---:|---:|---|"]
    for p in files:
        d, st = desc.get(p.name, ("", ""))
        lines.append(f"| `{p.name}` | {d} | {nrows(p)} | {p.stat().st_size:,} | {st} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    log(f"README written ({len(files)} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["freeze", "verify", "fit", "fit-worker", "analyze"], required=True)
    ap.add_argument("--devices", default="cuda:0,cuda:1", help="fit: one worker per device (speed only)")
    ap.add_argument("--device", default="cuda:1", help="fit-worker: device")
    ap.add_argument("--endpoints", default="", help="fit-worker: comma-separated endpoints")
    ap.add_argument("--batch", type=int, default=12000, help="CpGs per batched Cox fit (memory and speed only; D's run setting)")
    ap.add_argument("--min-free-gb", type=float, default=16.0, help="wait until this much GPU memory is free (execution only)")
    a = ap.parse_args()
    {"freeze": stage_freeze, "verify": stage_verify, "fit": stage_fit, "fit-worker": stage_fit_worker,
     "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
