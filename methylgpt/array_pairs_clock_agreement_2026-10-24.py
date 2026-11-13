#!/usr/bin/env python3
"""C. Clock agreement between arrays in the same people (large-gain analysis plan of 23 Oct 2026, analysis C).

Written on 24 Oct 2026 before any GEO search result was read and before any clock or model output was computed.
Every result computed is reported, whichever method it favours.

DATA, SEARCH AND INCLUSION
  Source: NCBI GEO, searched on 24 Oct 2026 with E-utilities (esearch, db=gds), entry type GSE:
    Q1 450K and EPIC v1: (GPL13534[ACCN] OR GPL16304[ACCN]) AND (GPL21145[ACCN] OR GPL23976[ACCN] OR GPL37464[ACCN]) AND gse[ETYP]
    Q2 EPIC v1 and EPIC v2: (GPL21145[ACCN] OR GPL23976[ACCN] OR GPL37464[ACCN]) AND (GPL33022[ACCN] OR GPL34510[ACCN]) AND gse[ETYP]
    Q3 the platform record that holds both EPIC and 450K samples: GPL34056[ACCN] AND gse[ETYP]
    Q4 keywords, 450K and EPIC deposited under one platform: (450K OR HM450 OR HumanMethylation450) AND (EPIC OR 850K) AND
       (compar* OR concordan* OR replicat* OR "same samples" OR "both arrays" OR "both platforms") AND "Homo sapiens"[ORGN] AND gse[ETYP]
    Q5 keywords, EPIC v1 and v2: ("EPIC v2" OR EPICv2 OR "EPIC version 2" OR 935K OR "v2.0") AND ("EPIC v1" OR EPICv1 OR 850K OR
       "EPIC version 1" OR "v1.0") AND "Homo sapiens"[ORGN] AND gse[ETYP]
  Array of a sample: its platform (GPL13534, GPL16304 = 450K; GPL21145, GPL23976, GPL37464 = EPIC v1; GPL33022, GPL34510 =
  EPIC v2). For GPL34056 and for series that deposit two arrays under one platform, the array is read from the sample
  metadata (title, characteristics, description, hybridisation protocol).
  Automatic screen: a series (SuperSeries and SubSeries merged; a series whose samples are all in another hit is covered by
  it) passes if it has at least 10 human samples on each array of a comparison (450K and EPIC v1; EPIC v1 and EPIC v2).
  For these, putative pairs are counted by matching across the two arrays the sample title with array words removed, and
  the values of characteristics whose key names an individual or sample (id, individual, subject, patient, donor, sample,
  participant, code, name, pair).
  Screen by reading: every series passing the automatic screen is read (summary, overall design, sample metadata) and
  classified with a recorded reason. Inclusion criteria:
    1. the same DNA specimen of the same individual (same extraction, or the same specimen) measured on both arrays,
       identifiable from the GEO metadata or the series' deposited documentation; different time points or specimens are
       excluded;
    2. both measurements under the series' standard protocol: protocol variants (reduced DNA input, FFPE restoration,
       alternative bisulfite kits, enzymatic conversion) are used only when the same variant applies to both arrays of a
       pair; titrations, in vitro methylated or unmethylated control DNA and DNA mixtures are excluded;
    3. processed beta values deposited for both arrays (series matrix or supplementary matrix), or IDAT files for both;
    4. at least 10 paired individuals after rules 1-2 and the rules below.
  One pair per individual and comparison; when an individual has several samples on one array (technical replicates), the
  sample with the lowest GSM accession is used. A series with all three arrays contributes two comparisons (450K-EPIC v1 and
  EPIC v1-EPIC v2); 450K-EPIC v2 is not a comparison of this plan.
  Freezing: the pairing rule of each included series (how pairs are read from its metadata) and its data source are written
  into PAIRING and DATA below after reading the metadata and before any clock or model output is computed. Stage freeze
  writes datasets_frozen.csv, pairs_frozen.csv and screening_decisions.csv; the tables and the PAIRING/DATA code are
  committed before stage raw_clocks or methylgpt runs. If no series meets the criteria the analysis stops with
  "Cannot proceed".
  Corpus membership: every GSM of pairs_frozen.csv is looked up in artifacts/corpus_2026-09-30/profiles.parquet (id = GSM,
  group = dataset); the frozen table records pairs with a profile in the corpus and their splits.

DATA PROCESSING
  Deposited processed beta values are used when deposited for both arrays of a series (the authors' processing, unchanged).
  IDATs only when one array lacks processed betas; then both arrays of that series are processed from IDATs with minfi
  (preprocessNoob; detection p > 0.01 set to missing). Values outside [0, 1]: M-values are converted with 2^M / (1 + 2^M),
  percentages divided by 100 (recorded per series). Detection p-value columns are ignored. Only cg probes are used.
  EPIC v2 probe names (cgXXXXXXXX_BC11, _TC21, ...) are mapped to the cg identifier by removing the suffix; replicate probes
  of one cg identifier are collapsed by their mean (missing values ignored). Each array's profile consists of the CpGs in
  that array's deposited matrix.

QUANTITIES AND METHODS (nothing is fitted; no data of these series adapt any model)
  age: raw = Horvath multi-tissue clock (BioLearn Horvathv1); MethylGPT = predicted age.
  GrimAge2: raw = BioLearn GrimAgeV2 (DNAmGrimAge); MethylGPT = distilled GrimAge2 output.
  DunedinPACE: raw = BioLearn DunedinPACE (its quantile normalisation to the gold-standard means over its background CpGs,
               as BioLearn implements it); MethylGPT = distilled DunedinPACE.
  PhenoAge: raw = BioLearn PhenoAge; MethylGPT = distilled PhenoAge.
  Horvathv1 (secondary): MethylGPT distilled Horvathv1 output against raw Horvathv1.
  Raw clocks: BioLearn 0.7.1, the implementation the repo uses for Generation Scotland (score_gs_clocks_*.py), run with
  /home/kying/anaconda3/envs/integrative/bin/python on all CpGs of the array's matrix. Two fills for missing clock CpGs
  (absent from the array's matrix, or missing in a sample), both comparators:
    R_mean (mean filling, as in the repo): a clock CpG observed in at least 80% of the series' samples on that array has its
      missing values filled with its mean over those samples (same array); otherwise missing values are filled with the
      MethylGPT training-split mean (mu of ppca_core_meanfilled_2026-09-30.npz, the fill of the raw-CpG comparators in the
      repo) when the CpG is in the vocabulary, else with BioLearn's reference value (DunedinPACE gold-standard mean for
      DunedinPACE CpGs, the sesame 450K median for the others).
    R_biolearn (BioLearn's defaults): BioLearn's hybrid imputation (CpGs observed in at least 80% of samples filled with the
      array's series mean, the others from the reference: imputation "dunedin" for DunedinPACE, "sesame_450k" for the rest).
  The comparator for the decision is the one of R_mean and R_biolearn with the lower primary metric (best comparator).
  GrimAge2 age and sex inputs: chronological age and sex from the GEO metadata when reported for every included individual
  of the series; otherwise the first array's Horvathv1 estimate (same fill variant) and BioLearn's SexEstimation on the
  first array. The same age and sex enter both arrays of an individual (BioLearn's GrimAge is linear in age and sex, so
  they do not change between-array differences, only the between-individual spread).
  ClockBase convention (secondary S8): MethylGPT's clock outputs were distilled from ClockBase values
  (build_clock_targets_2026-10-01.py), whose GrimAge age input is the sample's own Horvathv1 estimate and Female = 0
  (checked in the ClockBase files). S8 computes raw GrimAge2 that way: age = each array's own R_mean Horvathv1, Female = 0.
  MethylGPT: released bundle artifacts/release/methylgpt2-1.1; the three training runs runs/multitask_bio_2026-10-01,
  runs/multitask_bio_seed20261002, runs/multitask_bio_seed20261003 (final.pt = bundle weights.pt, weights_run2.pt,
  weights_run3.pt). Input: the array's vocabulary CpGs as measured (missing = unobserved, values clipped to [0, 1] as in the
  package); no filling. Clock outputs: bio head x bio_scale + bio_median of each run, averaged over the three runs; age: mean
  over runs of Finv(age head x fs + fm). Check: run-1 outputs equal the package's MethylGPT.predict() on the first array of
  the first series (largest difference reported).

PRIMARY METRIC (pre-specified; never changed after results)
  For comparison d (one series and array pair), quantity c in {GrimAge2, DunedinPACE} and method m:
    D(d, c, m) = mean over paired individuals i of |y_iB - y_iA| / SD_i(y_iA),
  A = the older array (450K in 450K-EPIC v1, EPIC v1 in EPIC v1-EPIC v2), B = the newer, and SD (ddof 1) is over the
  comparison's paired individuals of method m's own estimates on array A. Standardising each method by its own
  between-individual SD means a compressed output scale cannot lower D.
    M(m) = mean over comparisons of the mean over the two quantities of D.
  Relative reduction = 1 - M(MethylGPT) / M(best), best = R_mean or R_biolearn, whichever has the lower M.
  Interval: 2,000 bootstrap draws, seed 20261024, resampling paired individuals with replacement within each comparison
  (all methods and quantities on the same draw, every statistic recomputed on the draw); 95% percentile interval of
  M(MethylGPT) - M(best), with best fixed as chosen on the observed data.
  DECISION (plan, error metric): pass if the relative reduction against the best comparator is at least 10% and the 95%
  interval of M(MethylGPT) - M(best) excludes zero (lies below zero). Applied as written.

SECONDARY (all reported)
  S1 D, M and the relative reduction for age, PhenoAge and distilled Horvathv1, each alone, and for GrimAge2 and DunedinPACE
     each alone, with intervals (same draws).
  S2 ICC(A,1) (two-way random effects, absolute agreement, single measurement; McGraw and Wong 1996) per comparison, quantity
     and method, mean over comparisons, relative improvement (ICC_m - ICC_best) / ICC_best against the comparator with the
     higher mean ICC, and the interval of the difference (same draws), for the five quantities.
  S3 D on a common scale: every method's differences divided by the SD of the R_mean estimate on array A (the example scale
     in the analysis brief), M over GrimAge2 and DunedinPACE, relative reduction and interval.
  S4 Mean absolute difference in clock units and mean signed difference (B - A) per quantity, method and comparison.
  S5 Scale check (pre-specified): Pearson r between each MethylGPT output and its raw clock (R_mean and R_biolearn) on each
     array within each comparison, mean over comparisons, and pooled over comparisons; SD ratio SD(MethylGPT) / SD(raw) on
     array A. Reported beside the primary result, so a lower difference can be read against whether the output tracks
     the clock; it does not change the decision.
  S6 The primary metric restricted to comparisons none of whose paired profiles is in the corpus training split.
  S7 Per-comparison D for each quantity and method; number of comparisons in which MethylGPT has the lower D.
  S8 GrimAge2 under the ClockBase convention (above), as a third raw variant (not a decision comparator).
  S9 Coverage per comparison and array: clock CpGs present on the array, vocabulary CpGs observed.

FREEZE NOTE (24 Oct 2026, after the search and reading, before any clock or model output was computed)
  The search returned 655 series (113 series-comparisons passed the automatic screen); 6 comparisons met the criteria and
  are frozen in datasets_frozen.csv / pairs_frozen.csv (354 pairs): GSE86833 (10), GSE92580 (12) and GSE154566 (233) for
  450K-EPIC v1; GSE229715 (16), GSE286313 (67) and GSE350403 (16) for EPIC v1-EPIC v2. Clarification of the pairing rule:
  when an individual has several specimen types on both arrays, the blood specimen is used (GSE350403: buffy coat).
  Every paired profile of the three 450K-EPIC v1 comparisons has its 450K profile (and for GSE86833 and GSE92580 also its
  EPIC profile) in the corpus training split; no profile of the EPIC v1-EPIC v2 comparisons is in the corpus, so S6 is
  the three EPIC v1-EPIC v2 comparisons.

OUTPUTS
  Aggregate tables and README under artifacts/large_gain_2026-10-24/C/. Per-sample values of these public series stay
  under artifacts/external/array_pairs_2026-10-24/ (not committed); raw downloads under artifacts/external/<GSE>/.
Stages: --stage search | freeze | download | prepare | raw_clocks | methylgpt | analyse
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
OUT = ARTIFACTS / "large_gain_2026-10-24/C"
EXT = ARTIFACTS / "external"
META = EXT / "geo_meta_2026-10-24"
WORK = EXT / "array_pairs_2026-10-24"
BIOLEARN_PY = "/home/kying/anaconda3/envs/integrative/bin/python"
SEED, DRAWS, MIN_PAIRS = 20261024, 2000, 10
ARRAY_OF_GPL = {"GPL13534": "450K", "GPL16304": "450K", "GPL21145": "EPICv1", "GPL23976": "EPICv1", "GPL37464": "EPICv1",
                "GPL33022": "EPICv2", "GPL34510": "EPICv2"}
MIXED_GPL = {"GPL34056"}
COMPARISONS = {"450K-EPICv1": ("450K", "EPICv1"), "EPICv1-EPICv2": ("EPICv1", "EPICv2")}
QUERIES = {
    "Q1": "(GPL13534[ACCN] OR GPL16304[ACCN]) AND (GPL21145[ACCN] OR GPL23976[ACCN] OR GPL37464[ACCN]) AND gse[ETYP]",
    "Q2": "(GPL21145[ACCN] OR GPL23976[ACCN] OR GPL37464[ACCN]) AND (GPL33022[ACCN] OR GPL34510[ACCN]) AND gse[ETYP]",
    "Q3": "GPL34056[ACCN] AND gse[ETYP]",
    "Q4": '(450K OR HM450 OR HumanMethylation450) AND (EPIC OR 850K) AND (compar* OR concordan* OR replicat* OR "same samples" '
          'OR "both arrays" OR "both platforms") AND "Homo sapiens"[ORGN] AND gse[ETYP]',
    "Q5": '("EPIC v2" OR EPICv2 OR "EPIC version 2" OR 935K OR "v2.0") AND ("EPIC v1" OR EPICv1 OR 850K OR "EPIC version 1" OR '
          '"v1.0") AND "Homo sapiens"[ORGN] AND gse[ETYP]',
}
RUNS = ["runs/multitask_bio_2026-10-01/final.pt", "runs/multitask_bio_seed20261002/final.pt", "runs/multitask_bio_seed20261003/final.pt"]
# quantity -> (BioLearn model giving the raw value, MethylGPT output)
QUANT = {"GrimAge2": ("GrimAgeV2", "GrimAge2"), "DunedinPACE": ("DunedinPACE", "DunedinPACE"), "age": ("Horvathv1", "age"),
         "PhenoAge": ("PhenoAge", "PhenoAge"), "Horvathv1": ("Horvathv1", "Horvathv1")}
PRIMARY = ["GrimAge2", "DunedinPACE"]
RAW_VARIANTS = ["R_mean", "R_biolearn"]
ID_KEY = re.compile(r"(^|[^a-z])(id|individual|subject|patient|donor|sample|participant|code|name|pair)([^a-z]|$)", re.I)
ARRAY_WORDS = re.compile(r"(450 ?k|hm ?450|humanmethylation ?450|methylation ?450|850 ?k|935 ?k|epic ?v?[12](\.0)?|epic|v[12](\.0)?|"
                         r"array|beadchip|chip|illumina|infinium|methylation|platform)", re.I)

# ---------------------------------------------------------------------------------------------------------------
# FREEZE (written on 24 Oct 2026 after reading the 113 screened series-comparisons, before any clock or model output
# was computed). DECISIONS: every row of screen_automatic.csv; PAIRING: how pairs are read from the metadata of each
# included series; DATA: which deposited files are read (stage prepare).
# Clarification of the pairing rule (not covered by the pre-specification): when an individual has several specimen
# types measured on both arrays, the blood specimen is used (the tissue GrimAge2, DunedinPACE and PhenoAge were
# developed in; "prefer" below); otherwise the specimen type of the individual's lowest-GSM sample on array A.
_D = "different individuals or specimens on the two arrays; no individual measured on both arrays is identifiable from the metadata"
_T = "the same individuals appear on the two arrays only at different time points or with different specimens (criterion 1)"
_C = "the two arrays were used for different cohorts or sample sets"
_F = "fewer than 10 individuals with the same specimen on both arrays (criterion 4)"
_EXCLUDE = {
    "GSE107082": _D + " (participant subsets per array, titles are array positions)",
    "GSE110184": _D, "GSE111165": _F + ": the publication (PMID 30705257) states that six subjects overlap between the 450K and EPIC "
                 "datasets (15 samples run on both); the GEO subject labels do not identify which",
    "GSE115278": _C, "GSE117130": _D, "GSE117861": _D + " (sample numbers restart on each array; ages differ)", "GSE122078": _D,
    "GSE122126": _D, "GSE122920": _D, "GSE122994": _D, "GSE123601": _D, "GSE124569": _D, "GSE124617": _D, "GSE124633": _D,
    "GSE128686": _D + " (different cell lines)", "GSE130051": _D, "GSE131482": _D, "GSE132804": _D, "GSE136790": _D, "GSE137503": _C,
    "GSE140686": _D + " (reference and validation cases are numbered separately)", "GSE141039": _D, "GSE141256": _D,
    "GSE142512": _T + " (longitudinal samples; the series states no technical replicates)", "GSE147430": _D,
    "GSE157131": _D, "GSE157397": _D, "GSE157989": _D, "GSE158064": _T + " (fetal cord tissue on 450K, childhood buffy coat on EPIC)",
    "GSE161692": _D, "GSE165764": _T + " (xenografts before treatment on 450K, after treatment on EPIC)", "GSE166373": _D,
    "GSE171363": _D, "GSE175364": _D, "GSE180060": _D, "GSE184757": _D, "GSE184900": _D, "GSE185008": _D,
    "GSE185391": _T + " (different time points)", "GSE190365": _D, "GSE196228": _D, "GSE197672": _D, "GSE197724": _D, "GSE200707": _D,
    "GSE207998": _D, "GSE210256": _D + " (sample numbers restart on each array; ages differ)", "GSE210843": _D, "GSE211687": _D,
    "GSE212449": _D, "GSE212621": _D, "GSE212937": _F + ": sample labels differ between arrays, the same DNA is not stated, and at "
                 "most 8 patients match on age and sex", "GSE213554": _D, "GSE214568": _D, "GSE215240": _D, "GSE216262": _D,
    "GSE217733": _D, "GSE222595": _C + " (discovery cohort on 450K, replication cohort on EPIC)", "GSE223467": _D, "GSE224218": _D,
    "GSE224850": _D, "GSE226961": _D, "GSE227488": _D, "GSE228103": _D,
    "GSE228820": _F + ": four cell lines at standard input on both arrays; the other samples are input titrations of the same lines "
                 "and methylation control DNA (criterion 2)",
    "GSE233616": _D, "GSE237299": _D, "GSE237561": _C,
    "GSE240482": _F + ": six prostate tumours with the same DNA on both arrays; the cell-line and xenograft EPIC v1 profiles are "
                 "re-deposited from earlier series (same specimen not stated), and MCF7/TAMR and the eight xenografts come from one "
                 "and two individuals",
    "GSE242994": _D, "GSE243240": _D, "GSE247198": _F + " (two individuals, time series)", "GSE248471": _D, "GSE249712": _D,
    "GSE261324": _D, "GSE267015": _D, "GSE267654": _C + " (institution A on EPIC v1, institution B on EPIC v2)",
    "GSE269760": _F + " (one cell line)", "GSE270084": _D, "GSE277573": _D, "GSE279871": _F + " (four cell lines on both arrays)",
    "GSE280206": _D, "GSE283494": _D, "GSE288337": _D, "GSE289137": _D, "GSE290981": _D, "GSE292327": _D, "GSE293799": _D,
    "GSE295450": _D, "GSE296589": _D, "GSE299543": _F + " (one cell line, H1)", "GSE301739": _C, "GSE304293": _D, "GSE308071": _D,
    "GSE308134": _D, "GSE310580": _D, "GSE313522": _D, "GSE314841": _D, "GSE315764": _C, "GSE317378": _D, "GSE317578": _D,
    "GSE318454": _D, "GSE319573": _D, "GSE320217": _D, "GSE330883": _D, "GSE92469": _D,
}


def _title(r):
    return r.get("title", "").strip()


def _age_years(v):
    if v is None:
        return None
    v = str(v).strip()
    m = re.fullmatch(r"([\d.]+)\s*[Mm]", v)
    if m:
        return float(m.group(1)) / 12.0
    m = re.search(r"[\d.]+", v)
    return float(m.group(0)) if m else None


def _sex(v):
    v = (v or "").strip().lower()
    return "F" if v in ("f", "female") else "M" if v in ("m", "male") else None


def _k86833(r):
    t = _title(r)
    m = re.fullmatch(r"(CAF|NAF) ([ABC])", t)
    if m:
        return (f"patient {m.group(2)}", m.group(1))
    m = re.fullmatch(r"(LNCaP|PrEC) Rep\d", t)
    if m:
        return (m.group(1), "cells")
    return (t, "blood") if t.startswith("Guthrie ") else None


def _k92580(r):
    m = re.fullmatch(r"Pediatric brain tumor (\d+)\s+-\s+(.+)", _title(r))
    return (f"tumor {int(m.group(1))}", m.group(2).strip()) if m else None


def _k154566(r):
    c = chars(r)
    if c.get("tissue") != "Blood":
        return None
    t = _title(r)
    return (c["twinid"], "blood age 18") if (t.endswith("_450K") or t.endswith("_18B")) else None


def _k286313(r):
    m = re.fullmatch(r"(.+?)(_rep)?_EPICv(1_EPICv2|1|2)", _title(r))
    return (m.group(1), "whole blood") if m else None


PAIRING = {
    "GSE86833": {"comparisons": ["450K-EPICv1"], "key": _k86833, "age": lambda r: None, "sex": lambda r: None,
                 "note": "the same DNA profiled on HM450 and EPIC (series design); individuals: LNCaP, PrEC, three patients "
                         "(patient-matched CAF and NAF: one specimen per patient), five Guthrie-card infants; Rep1/Rep2 are "
                         "technical replicates"},
    "GSE92580": {"comparisons": ["450K-EPICv1"], "key": _k92580, "age": lambda r: None, "sex": lambda r: None,
                 "note": "fresh-frozen tumours T01-T09 on both arrays and REPLI-g-restored FFPE tumours T10-T12 on both arrays "
                         "(SubSeries GSE92577/GSE92579 design); pairs by sample title (the source name of tumour 7 reads 8 on 450K "
                         "and 2 on EPIC; the title and the design are followed)"},
    "GSE154566": {"comparisons": ["450K-EPICv1"], "key": _k154566, "age": lambda r: 18.0, "sex": lambda r: _sex(chars(r).get("gender")),
                  "note": "E-Risk age-18 blood DNA on 450K and EPIC (233 overlapping samples, series summary); buccal samples not used"},
    "GSE229715": {"comparisons": ["EPICv1-EPICv2"], "key": lambda r: (re.sub(r"^VLD_", "", _title(r)), "FFPE DNA"),
                  "age": lambda r: _age_years(chars(r).get("age")), "sex": lambda r: _sex(chars(r).get("sex")),
                  "note": "technical DNA replicates of 16 FFPE brain tumours on EPIC v1 and v2 (series design)"},
    "GSE286313": {"comparisons": ["EPICv1-EPICv2"], "key": _k286313, "age": lambda r: _age_years(chars(r).get("age")),
                  "sex": lambda r: _sex(chars(r).get("sex")),
                  "note": "matched whole-blood DNA on EPIC v1 and v2 (BeCOME, CALERIE, CLHNS, VHAS); _rep samples are technical replicates"},
    "GSE350403": {"comparisons": ["EPICv1-EPICv2"], "key": lambda r: (chars(r).get("participant id"), chars(r).get("tissue")),
                  "prefer": "buffy coat", "age": lambda r: _age_years(chars(r).get("age (years)")), "sex": lambda r: _sex(chars(r).get("sex")),
                  "note": "each specimen assayed on EPIC v1 and v2 (series design); 16 participants with buffy coat, PBMC and saliva: "
                          "the buffy coat (blood) specimen is used"},
}
INCLUDE = {(g, c) for g, v in PAIRING.items() for c in v["comparisons"]}
# GSE -> {array: (file under artifacts/external/<GSE>/, kind)}; kind: "table" (probe x sample matrix; detection p-value
# columns ignored) or "series_matrix". Columns are matched to GSMs by GSM, sample title, Sentrix barcode of the
# sample's IDAT file names, or the title with non-alphanumerics removed (in that order).
_GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
DATA = {
    "GSE86833": {"450K": (f"{_GEO}/GSE86nnn/GSE86829/suppl/GSE86829_hm450_processed.txt.gz", "table"),
                 "EPICv1": (f"{_GEO}/GSE86nnn/GSE86831/suppl/GSE86831_epic_processed.txt.gz", "table")},
    "GSE92580": {"450K": (f"{_GEO}/GSE92nnn/GSE92577/matrix/GSE92577_series_matrix.txt.gz", "series_matrix"),
                 "EPICv1": (f"{_GEO}/GSE92nnn/GSE92579/matrix/GSE92579_series_matrix.txt.gz", "series_matrix")},
    "GSE154566": {"450K": (f"{_GEO}/GSE154nnn/GSE154566/suppl/GSE154566_Overlaping_probes_normalised_seperately_450K_betas.csv.gz", "table"),
                  "EPICv1": (f"{_GEO}/GSE154nnn/GSE154566/suppl/GSE154566_Overlaping_probes_normalised_seperately_EPIC_betas.csv.gz", "table")},
    "GSE229715": {"EPICv1": (f"{_GEO}/GSE229nnn/GSE229715/suppl/GSE229715_1_betasnpval_all.csv.gz", "table"),
                  "EPICv2": (f"{_GEO}/GSE229nnn/GSE229715/suppl/GSE229715_1_betasnpval_all.csv.gz", "table")},
    "GSE286313": {"EPICv1": (f"{_GEO}/GSE286nnn/GSE286313/suppl/GSE286313_MatrixProcessed_GPL21145.csv.gz", "table"),
                  "EPICv2": (f"{_GEO}/GSE286nnn/GSE286313/suppl/GSE286313_MatrixProcessed_GPL33022.csv.gz", "table")},
    "GSE350403": {"EPICv1": (f"{_GEO}/GSE350nnn/GSE350403/suppl/GSE350403_EPICv1_regenerated_Noob_beta.tsv.gz", "table"),
                  "EPICv2": (f"{_GEO}/GSE350nnn/GSE350403/suppl/GSE350403_EPICv2_regenerated_Noob_beta.tsv.gz", "table")},
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def require(p):
    p = Path(p)
    if not p.exists():
        raise SystemExit(f"Cannot proceed - required data file not found: {p}")
    return p


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


# ------------------------------------------------------------------ GEO access
def http(url, data=None, tries=6, timeout=120):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "methylgpt-revision/2026-10-24"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network errors are retried, then reported
            log(f"retry {k + 1}/{tries} {url[:120]}: {e}")
            time.sleep(3 * (k + 1))
    raise SystemExit(f"download failed after {tries} tries: {url}")


def eutils(tool, params):
    time.sleep(0.4)  # below the 3 requests per second limit without an API key
    return json.loads(http(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{tool}.fcgi",
                           urllib.parse.urlencode({**params, "retmode": "json"}).encode()).decode())


def fetch(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if url.startswith("ftp://"):
        url = "https://" + url[len("ftp://"):]
    subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "5", "-o", str(tmp), url], check=True)
    tmp.rename(dest)
    return dest


def soft_samples(gse):
    """Sample records (GEO SOFT, brief view) of a series, cached under artifacts/external/geo_meta_2026-10-24/."""
    p = META / f"{gse}_gsm_brief.soft"
    if not p.exists():
        META.mkdir(parents=True, exist_ok=True)
        p.write_bytes(http(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}&targ=gsm&form=text&view=brief"))
        time.sleep(0.4)
    recs, cur = [], None
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("^SAMPLE"):
            cur = {"gsm": line.split("=", 1)[1].strip(), "characteristics": [], "description": [], "series": [], "supplementary": []}
            recs.append(cur)
        elif cur is not None and line.startswith("!Sample_"):
            k, _, v = line[8:].partition(" = ")
            k = k.strip()
            if k.startswith("characteristics"):
                cur["characteristics"].append(v)
            elif k == "description":
                cur["description"].append(v)
            elif k == "series_id":
                cur["series"].append(v)
            elif k == "supplementary_file":
                cur["supplementary"].append(v)
            elif k in ("title", "platform_id", "source_name_ch1", "organism_ch1", "hyb_protocol", "data_row_count"):
                cur[k] = v
    return recs


def soft_series(gse):
    p = META / f"{gse}_self_brief.soft"
    if not p.exists():
        META.mkdir(parents=True, exist_ok=True)
        p.write_bytes(http(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}&targ=self&form=text&view=brief"))
        time.sleep(0.4)
    out = {}
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("!Series_"):
            k, _, v = line[8:].partition(" = ")
            out.setdefault(k.strip(), []).append(v)
    return out


def chars(rec):
    d = {}
    for c in rec["characteristics"]:
        k, _, v = c.partition(":")
        d[k.strip().lower()] = v.strip()
    return d


def array_of(rec):
    gpl = rec.get("platform_id", "")
    if gpl in ARRAY_OF_GPL:
        return ARRAY_OF_GPL[gpl]
    if gpl in MIXED_GPL or gpl:
        text = " ".join([rec.get("title", ""), rec.get("hyb_protocol", ""), " ".join(rec["characteristics"]), " ".join(rec["description"])])
        if re.search(r"epic ?v2|v2\.0|935 ?k|epicv2", text, re.I):
            return "EPICv2"
        if re.search(r"450 ?k|hm ?450|methylation ?450", text, re.I) and not re.search(r"epic|850 ?k", text, re.I):
            return "450K"
        if re.search(r"epic|850 ?k", text, re.I) and not re.search(r"450 ?k|hm ?450|methylation ?450", text, re.I):
            return "EPICv1"
    return None


def norm_title(t):
    t = ARRAY_WORDS.sub(" ", t.lower())
    return re.sub(r"[^a-z0-9]+", "", t)


# ------------------------------------------------------------------ stage: search
def stage_search():
    OUT.mkdir(parents=True, exist_ok=True)
    hits = {}
    counts = {}
    for q, term in QUERIES.items():
        r = eutils("esearch", {"db": "gds", "term": term, "retmax": 10000})["esearchresult"]
        counts[q] = int(r["count"])
        for uid in r["idlist"]:
            hits.setdefault(uid, set()).add(q)
        log(f"{q}: {r['count']} series")
    uids = sorted(hits)
    summ = {}
    for i in range(0, len(uids), 200):
        res = json.loads(http("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                              urllib.parse.urlencode({"db": "gds", "id": ",".join(uids[i:i + 200]), "retmode": "json"}).encode()))["result"]
        summ.update({u: res[u] for u in res.get("uids", [])})
        time.sleep(0.4)
    rows = []
    for u in uids:
        s = summ.get(u, {})
        if s.get("entrytype") != "GSE":
            continue
        rows.append({"gse": s["accession"], "queries": ";".join(sorted(hits[u])), "title": s.get("title", ""), "gpl": s.get("gpl", ""),
                     "taxon": s.get("taxon", ""), "n_samples": int(s.get("n_samples", 0)), "pdat": s.get("pdat", ""),
                     "superseries": "SuperSeries is composed" in s.get("summary", ""), "suppfile": s.get("suppfile", ""),
                     "gsms": ";".join(sorted(x["accession"] for x in s.get("samples", [])))})
    hits_df = pd.DataFrame(rows).sort_values("gse")
    # series whose samples are all in another (larger) hit are covered by it
    sets = {r.gse: set(r.gsms.split(";")) - {""} for r in hits_df.itertuples()}
    covered = {}
    for g, sg in sets.items():
        for h, sh in sets.items():
            if h != g and sg and sg <= sh and (len(sh) > len(sg) or h < g):
                covered[g] = h
                break
    hits_df["covered_by"] = hits_df.gse.map(covered).fillna("")
    hits_df.drop(columns="gsms").to_csv(OUT / "geo_search_hits.csv", index=False)
    write_json(OUT / "geo_search_counts.json", {"date": time.strftime("%Y-%m-%d"), "queries": QUERIES, "counts": counts,
                                                "unique_series": int(len(hits_df)), "covered_by_another_hit": int(len(covered))})
    # automatic screen
    scr = []
    for r in hits_df.itertuples():
        if r.covered_by or "Homo sapiens" not in r.taxon:
            continue
        recs = [x for x in soft_samples(r.gse) if x.get("organism_ch1", "Homo sapiens") == "Homo sapiens"]
        arr = pd.Series([array_of(x) for x in recs], dtype=object)
        n = arr.value_counts().to_dict()
        for comp, (a, b) in COMPARISONS.items():
            if n.get(a, 0) < MIN_PAIRS or n.get(b, 0) < MIN_PAIRS:
                continue
            A = [x for x, y in zip(recs, arr) if y == a]
            B = [x for x, y in zip(recs, arr) if y == b]
            ta, tb = {norm_title(x.get("title", "")) for x in A} - {""}, {norm_title(x.get("title", "")) for x in B} - {""}
            best_key, best_n = "", 0
            ca, cb = [chars(x) for x in A], [chars(x) for x in B]
            for k in {k for d in ca for k in d if ID_KEY.search(k)}:
                va, vb = {d.get(k) for d in ca} - {None, ""}, {d.get(k) for d in cb} - {None, ""}
                if len(va & vb) > best_n:
                    best_key, best_n = k, len(va & vb)
            scr.append({"gse": r.gse, "comparison": comp, "title": r.title, "n_samples": r.n_samples, f"n_A": n.get(a, 0), "n_B": n.get(b, 0),
                        "putative_pairs_title": len(ta & tb), "putative_pairs_characteristic": best_n, "characteristic_key": best_key,
                        "superseries": r.superseries, "suppfile": r.suppfile})
        log(f"{r.gse}: arrays {n}")
    pd.DataFrame(scr).to_csv(OUT / "screen_automatic.csv", index=False)
    log(f"automatic screen: {len(scr)} series-comparisons with >= {MIN_PAIRS} samples on both arrays")


# ------------------------------------------------------------------ stage: freeze (PAIRING and DATA filled after reading)
def stage_freeze():
    scr = pd.read_csv(require(OUT / "screen_automatic.csv"))
    dec = []
    for r in scr.itertuples():
        if (r.gse, r.comparison) in INCLUDE:
            dec.append({"gse": r.gse, "comparison": r.comparison, "decision": "include", "reason": PAIRING[r.gse]["note"]})
        elif r.gse in _EXCLUDE:
            dec.append({"gse": r.gse, "comparison": r.comparison, "decision": "exclude", "reason": _EXCLUDE[r.gse]})
        else:
            raise SystemExit(f"no screening decision for {r.gse} {r.comparison}")
    prof = pd.read_parquet(require(ARTIFACTS / "corpus_2026-09-30/profiles.parquet"), columns=["id", "group", "split"])
    prof = prof.drop_duplicates("id").set_index("id")
    ds_rows, pair_rows = [], []
    for gse, spec in PAIRING.items():
        recs = sorted(soft_samples(gse), key=lambda x: int(x["gsm"][3:]))
        ser = soft_series(gse)
        for comp in spec["comparisons"]:
            a, b = COMPARISONS[comp]
            by = {a: {}, b: {}}          # array -> individual -> specimen -> [records in GSM order]
            for rec in recs:
                arr = array_of(rec)
                if arr not in by:
                    continue
                k = spec["key"](rec)
                if k is None or k[0] is None:
                    continue
                by[arr].setdefault(k[0], {}).setdefault(k[1], []).append(rec)
            n_pairs = 0
            for ind in sorted(set(by[a]) & set(by[b])):
                both = [sp for sp in by[a][ind] if sp in by[b][ind]]
                if not both:
                    continue
                if spec.get("prefer") in both:
                    sp = spec["prefer"]
                else:
                    sp = min(both, key=lambda x: int(by[a][ind][x][0]["gsm"][3:]))
                ra, rb = by[a][ind][sp][0], by[b][ind][sp][0]
                age_a, age_b = spec["age"](ra), spec["age"](rb)
                sex_a, sex_b = spec["sex"](ra), spec["sex"](rb)
                pair_rows.append({"gse": gse, "comparison": comp, "pair_id": f"{gse}|{ind}", "individual": ind, "specimen": sp,
                                  "array_A": a, "array_B": b, "gsm_A": ra["gsm"], "gsm_B": rb["gsm"],
                                  "title_A": ra.get("title", ""), "title_B": rb.get("title", ""),
                                  "replicates_A": len(by[a][ind][sp]), "replicates_B": len(by[b][ind][sp]),
                                  "age": age_a if age_a == age_b else (None if age_a is None or age_b is None else float("nan")),
                                  "sex": sex_a if sex_a == sex_b else None,
                                  **{f"{x}_{role}": (prof.loc[g, x] if g in prof.index else None)
                                     for role, g in (("A", ra["gsm"]), ("B", rb["gsm"])) for x in ("group", "split")}})
                n_pairs += 1
            P = [p for p in pair_rows if p["gse"] == gse and p["comparison"] == comp]
            in_corpus = sum((p["split_A"] is not None) or (p["split_B"] is not None) for p in P)
            splits = sorted({str(p[k]) for p in P for k in ("split_A", "split_B") if p[k] is not None})
            ds_rows.append({"gse": gse, "comparison": comp, "array_A": a, "array_B": b, "n_pairs": n_pairs,
                            "title": ser.get("title", [""])[0], "pmid": ";".join(ser.get("pubmed_id", [])),
                            "specimens": ";".join(sorted({p["specimen"] for p in P})),
                            "age_and_sex_from_metadata": bool(P) and all(p["age"] is not None and p["age"] == p["age"] for p in P)
                                                          and all(p["sex"] in ("F", "M") for p in P),
                            "pairs_with_a_profile_in_corpus": int(in_corpus), "corpus_splits": ";".join(splits),
                            "corpus_groups": ";".join(sorted({str(p[k]) for p in P for k in ("group_A", "group_B") if p[k] is not None})),
                            "meets_min_pairs": n_pairs >= MIN_PAIRS, "pairing_rule": spec["note"]})
            log(f"{gse} {comp}: {n_pairs} pairs; {in_corpus} with a profile in the corpus ({splits})")
    ds = pd.DataFrame(ds_rows)
    pairs = pd.DataFrame(pair_rows)
    drop = ds[~ds.meets_min_pairs]
    for r in drop.itertuples():
        for d in dec:
            if d["gse"] == r.gse and d["comparison"] == r.comparison:
                d["decision"], d["reason"] = "exclude", f"{_F}: {r.n_pairs} pairs after the pairing rules"
    ds = ds[ds.meets_min_pairs].drop(columns="meets_min_pairs")
    pairs = pairs.merge(ds[["gse", "comparison"]], on=["gse", "comparison"])
    # GrimAge2 age and sex inputs: metadata only when reported for every included individual of the comparison
    for r in ds.itertuples():
        if not r.age_and_sex_from_metadata:
            m = (pairs.gse == r.gse) & (pairs.comparison == r.comparison)
            pairs.loc[m, ["age", "sex"]] = None
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dec).to_csv(OUT / "screening_decisions.csv", index=False)
    ds.to_csv(OUT / "datasets_frozen.csv", index=False)
    pairs.to_csv(OUT / "pairs_frozen.csv", index=False)
    log(f"frozen: {len(ds)} comparisons, {len(pairs)} pairs; decisions for {len(dec)} screened series-comparisons")


# ------------------------------------------------------------------ frozen tables and per-array matrices
def frozen():
    ds = pd.read_csv(require(OUT / "datasets_frozen.csv"))
    pairs = pd.read_csv(require(OUT / "pairs_frozen.csv"))
    if ds.empty:
        raise SystemExit("Cannot proceed - no GEO series met the inclusion criteria (datasets_frozen.csv is empty)")
    return ds, pairs


def load_matrix(gse, array):
    """Per-array matrix written by stage prepare: beta (n_samples, n_cpgs) float32, cpgs, gsms."""
    z = np.load(require(WORK / gse / f"{array}.npz"), allow_pickle=False)
    return z["beta"], z["cpgs"].astype(str), z["gsms"].astype(str)


def collapse_epic_v2(beta, probes):
    """EPIC v2: strip the probe suffix (cgXXXXXXXX_BC11 -> cgXXXXXXXX) and average replicate probes (NaN ignored).
    beta: (n_samples, n_probes)."""
    cg = pd.Index([p.split("_")[0] for p in probes])
    df = pd.DataFrame(beta.T, index=cg)
    df = df.groupby(level=0, sort=True).mean()
    return df.to_numpy(np.float32).T, df.index.to_numpy(str)


def to_beta(v):
    """M-values to beta, percentages to fractions; returns (values, conversion)."""
    lo, hi = np.nanmin(v), np.nanmax(v)
    if lo < -0.01:
        return (2.0 ** v / (1 + 2.0 ** v)).astype(np.float32), "M-value"
    if hi > 1.5:
        return (v / 100.0).astype(np.float32), "percent"
    return v.astype(np.float32), "beta"


# ------------------------------------------------------------------ stages: download, prepare
def stage_download():
    ds, _ = frozen()
    for g in sorted(set(ds.gse)):
        for array, (url, _) in DATA[g].items():
            fetch(url, EXT / g / Path(url).name)
            log(f"{g} {array}: {Path(url).name}")


def _alnum(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def column_to_gsm(gse, array, columns):
    recs = [r for r in soft_samples(gse) if array_of(r) == array]
    by_title = {r.get("title", "").strip(): r["gsm"] for r in recs}
    by_alnum = {_alnum(r.get("title", "")): r["gsm"] for r in recs}
    by_bar = {}
    for r in recs:
        for f in r["supplementary"]:
            m = re.search(r"(\d{9,12}_R\d\dC\d\d)", f)
            if m:
                by_bar[m.group(1)] = r["gsm"]
    gsms = {r["gsm"] for r in recs}
    out = {}
    for c in columns:
        c0 = str(c).strip()
        g = c0 if c0 in gsms else by_title.get(c0) or by_bar.get(c0) or by_alnum.get(_alnum(c0))
        if g is not None:
            out[c] = g
    return out


def read_matrix(path, kind):
    if kind == "series_matrix":
        with gzip.open(path, "rt") as f:
            start = next(i for i, line in enumerate(f) if line.startswith("!series_matrix_table_begin"))
        df = pd.read_csv(path, sep="\t", skiprows=start + 1, index_col=0, low_memory=False)
        return df[~df.index.astype(str).str.startswith("!")]
    with gzip.open(path, "rt") as f:
        head = f.readline()
    sep = "\t" if head.count("\t") > head.count(",") else ","
    return pd.read_csv(path, sep=sep, index_col=0, low_memory=False)


def stage_prepare():
    ds, pairs = frozen()
    rec = []
    for g in sorted(set(ds.gse)):
        cache = {}
        for array, (url, kind) in DATA[g].items():
            need = set(pairs[(pairs.gse == g) & (pairs.array_A == array)].gsm_A) | set(pairs[(pairs.gse == g) & (pairs.array_B == array)].gsm_B)
            if not need:
                continue
            path = require(EXT / g / Path(url).name)
            if path not in cache:
                cache[path] = read_matrix(path, kind)
            df = cache[path]
            cmap = {c: v for c, v in column_to_gsm(g, array, [c for c in df.columns if "pval" not in str(c).lower()
                                                              and not re.search(r"\.\d+$", str(c))]).items() if v in need}
            missing = need - set(cmap.values())
            if missing:
                raise SystemExit(f"{g} {array}: paired samples without a column in {path.name}: {sorted(missing)[:5]}")
            if len(set(cmap.values())) != len(cmap):
                raise SystemExit(f"{g} {array}: several columns map to one GSM")
            sub = df[list(cmap)].apply(pd.to_numeric, errors="coerce")
            sub.columns = [cmap[c] for c in sub.columns]
            sub = sub[sub.index.astype(str).str.startswith("cg")]
            beta, conv = to_beta(sub.to_numpy(np.float64))
            probes = sub.index.astype(str).to_numpy()
            n_probes = len(probes)
            suffixed = int(sum("_" in x for x in probes))
            if array == "EPICv2" or suffixed:
                beta_t, cpgs = collapse_epic_v2(beta.T, probes)
                beta = beta_t.T
            else:
                cpgs = probes
                if len(set(cpgs)) != len(cpgs):
                    raise SystemExit(f"{g} {array}: duplicated probe names")
            # drop probes missing in every paired sample of this array (not measured in the deposited matrix)
            keep = np.isfinite(beta).any(1)
            beta, cpgs = beta[keep], cpgs[keep]
            gsms = np.asarray(sub.columns, dtype=str)
            (WORK / g).mkdir(parents=True, exist_ok=True)
            np.savez(WORK / g / f"{array}.npz", beta=beta.T.astype(np.float32), cpgs=cpgs.astype(str), gsms=gsms)
            rec.append({"gse": g, "array": array, "file": path.name, "kind": kind, "samples": len(gsms), "cg_probes_in_file": n_probes,
                        "probes_with_suffix": suffixed, "cpgs_after_collapse_with_data": int(len(cpgs)), "value_scale": conv,
                        "missing_fraction": float(np.mean(~np.isfinite(beta)))})
            log(f"{g} {array}: {len(gsms)} samples, {len(cpgs)} CpGs ({conv}; {suffixed} suffixed probes)")
        del cache
    pd.DataFrame(rec).to_csv(WORK / "prepare_summary.csv", index=False)


# ------------------------------------------------------------------ stage: raw_clocks (BioLearn)
BIOLEARN_DATA = Path("/home/kying/anaconda3/envs/integrative/lib/python3.10/site-packages/biolearn/data")


def clock_cpgs():
    out = {}
    for name, fn, col in (("Horvathv1", "Horvath1.csv", "CpGmarker"), ("PhenoAge", "PhenoAge.csv", "CpGmarker"),
                          ("DunedinPACE", "DunedinPACE.csv", "CpGmarker"), ("GrimAgeV2", "GrimAgeV2.csv", "var")):
        v = pd.read_csv(require(BIOLEARN_DATA / fn))[col].dropna().astype(str)
        out[name] = sorted(set(v[v.str.startswith("cg")]))
    return out


def r_mean_fill(beta, cpgs, clock):
    """R_mean: clock CpG rows (n_samples, n_clock_cpgs) filled as pre-specified."""
    vocab = (ARTIFACTS / "release/methylgpt2-1.1/vocab.txt").read_text().split()
    mu = np.load(require(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz"))["mu"]
    mu = dict(zip(vocab, mu.astype(float)))
    gold = pd.read_csv(require(BIOLEARN_DATA / "DunedinPACE_Gold_Means.csv"), index_col=0)["mean"]
    ses = pd.read_csv(require(BIOLEARN_DATA / "sesame_450k_median.csv"), index_col=0)["median"]
    dun = set(clock["DunedinPACE"])
    allc = sorted(set().union(*clock.values()))
    pos = {c: i for i, c in enumerate(cpgs)}
    n = beta.shape[0]
    F = np.full((n, len(allc)), np.nan, np.float32)
    src = {"series_mean": 0, "training_mean": 0, "gold_mean": 0, "sesame_median": 0, "absent_from_array": 0}
    for j, c in enumerate(allc):
        x = beta[:, pos[c]].astype(np.float32) if c in pos else np.full(n, np.nan, np.float32)
        if c not in pos:
            src["absent_from_array"] += 1
        ok = np.isfinite(x)
        if ok.mean() >= 0.8:
            fill = float(x[ok].mean())
            src["series_mean"] += int((~ok).any())
        elif c in mu:
            fill = mu[c]; src["training_mean"] += 1
        elif c in dun:
            fill = float(gold[c]); src["gold_mean"] += 1
        else:
            fill = float(ses[c]); src["sesame_median"] += 1
        F[:, j] = np.where(ok, x, fill)
    return F, np.asarray(allc), src


BIOLEARN_CODE = r"""
import json, sys
import numpy as np, pandas as pd
from biolearn.data_library import GeoData
from biolearn.model_gallery import ModelGallery
z = np.load(sys.argv[1], allow_pickle=False)
spec = json.loads(open(sys.argv[2]).read())
gal = ModelGallery()
ROWS = set(z["clock_rows"].astype(str))          # model CpGs of the four clocks and DunedinPACE's gold-standard CpGs
_cache = {}
def frame(arr, full=False):
    # Every clock reads only its own CpGs (DunedinPACE also its 20,000 gold-standard CpGs, among which its model
    # CpGs are), and BioLearn's imputations act row by row, so restricting the rows leaves the clock values unchanged;
    # SexEstimation standardises over all autosomal probes and gets the full matrix.
    key = (arr, full)
    if key not in _cache:
        X = z[f"{arr}_beta"].T                      # cpgs x samples
        cp = z[f"{arr}_cpgs"].astype(str)
        keep = np.ones(len(cp), bool) if full else np.array([c in ROWS for c in cp])
        df = pd.DataFrame(X[keep], index=cp[keep], columns=z[f"{arr}_gsms"].astype(str))
        _cache[key] = df[~df.index.duplicated()]
    return _cache[key].copy()
def filled(arr):
    df = frame(arr)
    F = pd.DataFrame(z[f"{arr}_fill"].T, index=z["fill_cpgs"].astype(str), columns=df.columns)
    return pd.concat([df.drop(index=F.index.intersection(df.index)), F])
def meta(cols, age=None, sex=None):
    m = pd.DataFrame(index=cols)
    m["age"] = 40.0 if age is None else age
    m["sex"] = 2 if sex is None else sex       # BioLearn coding: 1 female, 2 male
    return m
def pred(name, df, imp, age=None, sex=None):
    out = gal.get(name, imputation_method=imp).predict(GeoData(meta(df.columns, age, sex), df.copy()))
    col = "DNAmGrimAge" if name.startswith("GrimAge") else ("Predicted" if "Predicted" in out.columns else out.columns[0])
    return out[col].reindex(df.columns).to_numpy(float)
rows = []
sexest = None
for variant in ("R_mean", "R_biolearn"):
    res = {}
    for arr in ("A", "B"):
        df = filled(arr) if variant == "R_mean" else frame(arr)
        imp = {"R_mean": {"Horvathv1": "averaging", "PhenoAge": "averaging", "DunedinPACE": "none"},
               "R_biolearn": {"Horvathv1": "sesame_450k", "PhenoAge": "sesame_450k", "DunedinPACE": "dunedin"}}[variant]
        for name in ("Horvathv1", "PhenoAge", "DunedinPACE"):
            res[(arr, name)] = pred(name, df, imp[name])
    age = np.asarray(spec["age"], float) if spec["age"] is not None else res[("A", "Horvathv1")]
    if spec["sex"] is not None:
        sex = np.asarray(spec["sex"], int)
    else:
        if sexest is None:
            fa = frame("A", full=True)
            s = gal.get("SexEstimation").predict(GeoData(meta(fa.columns), fa))
            sexest = s["predicted_sex"].reindex(fa.columns).map({"F": 1, "M": 2, "Female": 1, "Male": 2}).fillna(2).astype(int).to_numpy()
        sex = sexest
    for arr in ("A", "B"):
        df = filled(arr) if variant == "R_mean" else frame(arr)
        res[(arr, "GrimAgeV2")] = pred("GrimAgeV2", df, "averaging" if variant == "R_mean" else "sesame_450k", age, sex)
        for i, g in enumerate(z[f"{arr}_gsms"].astype(str)):
            rows.append({"gsm": g, "array_role": arr, "variant": variant, **{k[1]: float(v[i]) for k, v in res.items() if k[0] == arr},
                         "age_input": float(age[i]), "sex_input": int(sex[i])})
    if variant == "R_mean":
        for arr in ("A", "B"):
            g8 = pred("GrimAgeV2", filled(arr), "averaging", res[(arr, "Horvathv1")], np.full(len(age), 2))
            for i, g in enumerate(z[f"{arr}_gsms"].astype(str)):
                rows.append({"gsm": g, "array_role": arr, "variant": "S8_clockbase", "GrimAgeV2": float(g8[i]),
                             "age_input": float(res[(arr, "Horvathv1")][i]), "sex_input": 2})
pd.DataFrame(rows).to_csv(sys.argv[3], index=False)
print("sex source:", "metadata" if spec["sex"] is not None else "SexEstimation", "; age source:", "metadata" if spec["age"] is not None else "Horvathv1 array A")
"""


def stage_raw_clocks():
    ds, pairs = frozen()
    clock = clock_cpgs()
    cover = []
    for r in ds.itertuples():
        P = pairs[(pairs.gse == r.gse) & (pairs.comparison == r.comparison)].sort_values("pair_id")
        d = WORK / r.gse / r.comparison
        d.mkdir(parents=True, exist_ok=True)
        arrs = {}
        for role, array, col in (("A", r.array_A, "gsm_A"), ("B", r.array_B, "gsm_B")):
            beta, cpgs, gsms = load_matrix(r.gse, array)
            idx = pd.Index(gsms).get_indexer(P[col].astype(str))
            if (idx < 0).any():
                raise SystemExit(f"{r.gse} {array}: paired samples missing from the matrix: {list(P[col][idx < 0])[:5]}")
            B = beta[idx]
            F, fc, src = r_mean_fill(B, cpgs, clock)
            arrs.update({f"{role}_beta": B, f"{role}_cpgs": cpgs, f"{role}_gsms": np.asarray(P[col].astype(str).tolist(), dtype=str), f"{role}_fill": F})
            arrs["fill_cpgs"] = fc
            cs = set(cpgs)
            cover.append({"gse": r.gse, "comparison": r.comparison, "array_role": role, "array": array, "cpgs_on_array": len(cpgs),
                          **{f"{k}_cpgs_on_array": int(sum(c in cs for c in v)) for k, v in clock.items()},
                          **{f"{k}_cpgs_total": len(v) for k, v in clock.items()}, **{f"r_mean_fill_{k}": v for k, v in src.items()}})
        gold = pd.read_csv(require(BIOLEARN_DATA / "DunedinPACE_Gold_Means.csv"), index_col=0).index.astype(str)
        arrs["clock_rows"] = np.asarray(sorted(set().union(*clock.values()) | set(gold)), dtype=str)
        np.savez(d / "biolearn_input.npz", **arrs)
        age = P["age"].tolist() if "age" in P and P["age"].notna().all() else None
        sex = P["sex"].map({"F": 1, "M": 2}).tolist() if "sex" in P and P["sex"].isin(["F", "M"]).all() else None
        (d / "biolearn_spec.json").write_text(json.dumps({"age": age, "sex": sex}))
        p = subprocess.run([BIOLEARN_PY, "-c", BIOLEARN_CODE, str(d / "biolearn_input.npz"), str(d / "biolearn_spec.json"),
                            str(d / "raw_clocks.csv")], capture_output=True, text=True, cwd=str(d))
        (d / "biolearn.log").write_text(p.stdout + "\n" + p.stderr)
        if p.returncode != 0:
            raise SystemExit(f"BioLearn failed for {r.gse} {r.comparison}: {p.stderr[-3000:]}")
        (d / "biolearn_input.npz").unlink()
        log(f"{r.gse} {r.comparison}: raw clocks written ({p.stdout.strip()})")
    pd.DataFrame(cover).to_csv(WORK / "coverage.csv", index=False)


# ------------------------------------------------------------------ stage: methylgpt
def stage_methylgpt(device):
    import torch
    from methylgpt_lib import Finv, cuda_preinit
    from methylgpt_model import BioHeads, embed_batch, load_checkpoint
    cuda_preinit()
    dev = torch.device(device)
    ds, pairs = frozen()
    vocab = (ARTIFACTS / "release/methylgpt2-1.1/vocab.txt").read_text().split()
    col = {c: i for i, c in enumerate(vocab)}
    inputs = {}
    for r in ds.itertuples():
        for array in (r.array_A, r.array_B):
            if (r.gse, array) in inputs:
                continue
            beta, cpgs, gsms = load_matrix(r.gse, array)
            keep = [i for i, c in enumerate(cpgs) if c in col]
            X = np.full((len(gsms), len(vocab)), np.nan, np.float32)
            X[:, [col[cpgs[i]] for i in keep]] = np.clip(beta[:, keep], 0, 1)
            inputs[(r.gse, array)] = (X, gsms)
    outs = {k: [] for k in inputs}
    names = None
    for rel in RUNS:
        ck = torch.load(require(ARTIFACTS / rel), map_location=dev, weights_only=False)
        model, core, _ = load_checkpoint(HERE / ck["ckpt"], ck.get("config", {}).get("core"), dev)
        model.load_state_dict(ck["model"]); model.eval()
        heads = BioHeads(core.rank + model.latents.shape[1], len(ck["tissues"]) + 1, len(ck["bio_names"])).to(dev)
        heads.load_state_dict(ck["heads"]); heads.eval()
        names = list(ck["bio_names"]) if names is None else names
        assert list(ck["bio_names"]) == names
        med, sc = np.asarray(ck["bio_median"], float), np.asarray(ck["bio_scale"], float)
        for k, (X, _) in inputs.items():
            ages, bios = [], []
            with torch.no_grad():
                for i in range(0, len(X), 16):
                    x = torch.from_numpy(X[i:i + 16]).to(dev)
                    h = heads(embed_batch(model, core, x, torch.isfinite(x)))
                    ages.append(h[0].float().cpu().numpy()); bios.append(h[3].float().cpu().numpy())
            outs[k].append((Finv(np.concatenate(ages) * ck["fs"] + ck["fm"]), np.concatenate(bios) * sc + med))
        log(f"{rel}: done")
        del model, core, heads, ck
        torch.cuda.empty_cache()
    check = None
    for k, (X, gsms) in inputs.items():
        age = np.mean([o[0] for o in outs[k]], 0)
        bio = np.mean([o[1] for o in outs[k]], 0)
        df = pd.DataFrame(bio, columns=names)
        df.insert(0, "age", age)
        for j, o in enumerate(outs[k]):
            df[f"age_run{j + 1}"] = o[0]
            for q in ("GrimAge2", "DunedinPACE", "PhenoAge", "Horvathv1"):
                df[f"{q}_run{j + 1}"] = o[1][:, names.index(q)]
        df.insert(0, "vocab_cpgs_observed", np.isfinite(X).sum(1))
        df.insert(0, "gsm", gsms)
        (WORK / k[0]).mkdir(parents=True, exist_ok=True)
        df.to_csv(WORK / k[0] / f"methylgpt_{k[1]}.csv", index=False)
        if check is None:  # run 1 against the released package on the first array of the first series
            sys.path.insert(0, str(HERE / "package/src"))
            from methylgpt import MethylGPT
            pkg = MethylGPT.from_pretrained(ARTIFACTS / "release/methylgpt2-1.1", device=device)
            cols = pd.DataFrame(X, columns=vocab)
            ref = pkg.predict(cols)
            run1 = outs[k][0][1]
            check = {"series": k[0], "array": k[1], "profiles": int(len(X)),
                     "max_abs_diff_clock_outputs_in_head_sd_units": float(np.max(np.abs((ref[names].to_numpy() - run1) / sc))),
                     "max_abs_diff_age_ensemble_years": float(np.max(np.abs(ref["age"].to_numpy() - age)))}
            del pkg
            torch.cuda.empty_cache()
            log(f"package check: {check}")
    write_json(WORK / "methylgpt_check.json", check)


# ------------------------------------------------------------------ stage: analyse
def assemble():
    """vals[(comparison_id, quantity, method)] = (y_A, y_B), aligned by pair."""
    ds, pairs = frozen()
    vals, meta = {}, {}
    for r in ds.itertuples():
        cid = f"{r.gse}|{r.comparison}"
        P = pairs[(pairs.gse == r.gse) & (pairs.comparison == r.comparison)].sort_values("pair_id")
        raw = pd.read_csv(require(WORK / r.gse / r.comparison / "raw_clocks.csv"))
        mg = {arr: pd.read_csv(require(WORK / r.gse / f"methylgpt_{arr}.csv")).set_index("gsm") for arr in (r.array_A, r.array_B)}
        ga, gb = P.gsm_A.astype(str).to_numpy(), P.gsm_B.astype(str).to_numpy()
        for q, (bl, mo) in QUANT.items():
            vals[(cid, q, "MethylGPT")] = (mg[r.array_A].loc[ga, mo].to_numpy(float), mg[r.array_B].loc[gb, mo].to_numpy(float))
            for v in RAW_VARIANTS + (["S8_clockbase"] if q == "GrimAge2" else []):
                t = raw[raw.variant == v]
                ya = t[t.array_role == "A"].set_index("gsm").loc[ga, bl].to_numpy(float)
                yb = t[t.array_role == "B"].set_index("gsm").loc[gb, bl].to_numpy(float)
                vals[(cid, q, v)] = (ya, yb)
        in_train = bool((P.split_A.astype(str).eq("train") | P.split_B.astype(str).eq("train")).any())
        meta[cid] = {"gse": r.gse, "comparison": r.comparison, "n_pairs": int(len(P)), "any_pair_in_training_split": in_train}
    for k, (a, b) in vals.items():
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise SystemExit(f"non-finite values for {k}")
    return vals, meta


def stats_one(vals, cids, idx=None):
    """Per-comparison statistics on one (possibly resampled) set of individuals."""
    out = {}
    for (cid, q, m), (a, b) in vals.items():
        if cid not in cids:
            continue
        t = slice(None) if idx is None else idx[cid]
        a_, b_ = a[t], b[t]
        ra = vals[(cid, q, "R_mean")][0][t]
        out[(cid, q, m)] = {"D": std_diff(a_, b_), "D_common": std_diff(a_, b_, ra), "ICC": icc_a1(a_, b_),
                            "MAD_units": float(np.mean(np.abs(b_ - a_))), "mean_signed_B_minus_A": float(np.mean(b_ - a_)),
                            "sd_A": float(np.std(a_, ddof=1))}
    return out


def agg(st, cids, quants, method, metric):
    return float(np.mean([np.mean([st[(c, q, method)][metric] for q in quants]) for c in cids]))


FILES = {
    "geo_search_counts.json": ("search", "GEO queries (E-utilities, 24 Oct 2026), hit counts per query, unique series", "E-utilities esearch/esummary"),
    "geo_search_hits.csv": ("search", "every series returned by the queries (one row per series; covered_by = a hit containing all its samples)", "E-utilities esummary"),
    "screen_automatic.csv": ("search", "series-comparisons with at least 10 samples on each array, with putative pair counts", "GEO SOFT sample records (artifacts/external/geo_meta_2026-10-24/)"),
    "screening_decisions.csv": ("freeze", "decision (include/exclude) and reason for every row of screen_automatic.csv", "screen_automatic.csv, GEO series and sample records"),
    "datasets_frozen.csv": ("freeze", "the frozen comparisons (series x array pair), pairs, specimens, corpus membership, pairing rule", "GEO SOFT records, artifacts/corpus_2026-09-30/profiles.parquet"),
    "pairs_frozen.csv": ("freeze", "the frozen pairs: GSM on each array, specimen, metadata age and sex, corpus group and split of each GSM", "GEO SOFT records, profiles.parquet"),
    "coverage.csv": ("raw_clocks", "per comparison and array: CpGs in the deposited matrix, clock CpGs present, and how R_mean filled the clock CpGs", "deposited matrices (prepare stage), BioLearn 0.7.1 coefficient files"),
    "per_comparison_metrics.csv": ("analyse", "per comparison, quantity and method: D (own-SD standardised mean absolute between-array difference), D_common (R_mean SD), ICC(A,1), mean absolute difference in clock units, mean signed difference B - A, SD on array A", "per-sample raw clocks and MethylGPT outputs (artifacts/external/array_pairs_2026-10-24/, not committed)"),
    "summary_metrics.csv": ("analyse", "M (mean over comparisons) per subset, quantity set and metric for each method, best comparator, difference and relative gain with 95% bootstrap intervals (2,000 draws, seed 20261024, individuals resampled within comparisons), comparisons in which MethylGPT is better", "per_comparison inputs as above"),
    "scale_check.csv": ("analyse", "S5: Pearson r between each MethylGPT output and its raw clock and SD ratio, per comparison, array role and raw variant, plus the mean over comparisons and the pooled values", "per-sample values as above"),
    "rule_outcome.csv": ("analyse", "primary metric for MethylGPT, best comparator and value, relative gain, 95% interval of the difference, pass or fail, scale check", "summary_metrics.csv, scale_check.csv"),
    "results.json": ("analyse", "rule outcome, per-comparison metadata, MethylGPT package check, bootstrap settings", "as above"),
}


def write_readme():
    lines = ["# Analysis C: clock agreement between arrays in the same people", "",
             "Source script: `array_pairs_clock_agreement_2026-10-24.py` (pre-specification in its docstring). "
             "Plan: large-gain analysis plan of 23 Oct 2026, analysis C.", "",
             "Per-sample clock values of the public GEO series stay under `artifacts/external/array_pairs_2026-10-24/` and are not committed; "
             "the tables here are aggregates, the frozen dataset list and the GEO identifiers of the pairs.", ""]
    ro = OUT / "rule_outcome.csv"
    if ro.exists():
        r = pd.read_csv(ro).iloc[0]
        lines += ["## Primary result", "",
                  f"- {r.comparisons} comparisons; MethylGPT {r.methylgpt:.4f}, best comparator {r.best_comparator} {r.best_comparator_value:.4f} "
                  f"({r.other_comparator} {r.other_comparator_value:.4f}); relative reduction {r.relative_gain:.1%}; difference "
                  f"{r.difference:.4f} (95% interval {r.difference_ci95_low:.4f} to {r.difference_ci95_high:.4f}); pass = {bool(r['pass'])}.",
                  f"- Scale check (array A, mean over comparisons, against {r.best_comparator}): r = {r.scale_check_mean_r_GrimAge2_array_A:.3f} "
                  f"(GrimAge2), {r.scale_check_mean_r_DunedinPACE_array_A:.3f} (DunedinPACE); SD ratio MethylGPT/raw "
                  f"{r.scale_check_sd_ratio_GrimAge2_array_A:.3f} and {r.scale_check_sd_ratio_DunedinPACE_array_A:.3f}.", "",
                  "## Data notes", "",
                  "- Deposited processed beta values were used for every series (no IDAT processing).",
                  "- GSE154566: the deposited separately normalised matrices hold only probes shared by both arrays (317,976 CpGs), "
                  "so both arrays have the same clock CpGs.",
                  "- GSE229715: one deposited matrix holds both arrays; EPIC v2 values are already under cg names (no suffixes); "
                  "EPIC v1-only probes are missing in the EPIC v2 columns. GSE286313 and GSE350403: EPIC v2 suffixed probes were "
                  "collapsed to cg identifiers by their mean.",
                  "- The 450K profiles of all three 450K-EPIC v1 comparisons (and the EPIC profiles of GSE86833 and GSE92580) are in "
                  "the corpus training split; no profile of the EPIC v1-EPIC v2 comparisons is in the corpus (subset S6).", ""]
    for f, (stage, desc, src) in FILES.items():
        path = OUT / f
        if not path.exists():
            lines += [f"## {f}", "", "Not written.", ""]
            continue
        if f.endswith(".csv"):
            d = pd.read_csv(path)
            cols, rows = ", ".join(d.columns), len(d)
        else:
            cols, rows = "JSON object", "-"
        lines += [f"## {f}", "", f"- Stage: `--stage {stage}`", f"- Content: {desc}", f"- Rows: {rows}", f"- Columns: {cols}", f"- Inputs: {src}", ""]
    (OUT / "README.md").write_text("\n".join(lines))


def stage_analyse():
    vals, meta = assemble()
    cids = sorted(meta)
    methods = ["MethylGPT"] + RAW_VARIANTS
    obs = stats_one(vals, cids)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(DRAWS):
        idx = {c: rng.integers(0, meta[c]["n_pairs"], meta[c]["n_pairs"]) for c in cids}
        boots.append(stats_one(vals, cids, idx))
    qsets = {"primary (GrimAge2, DunedinPACE)": PRIMARY, **{q: [q] for q in QUANT}}
    subsets = {"all comparisons": cids, "S6 no profile in the corpus training split": [c for c in cids if not meta[c]["any_pair_in_training_split"]]}
    summary = []
    for sname, cs in subsets.items():
        if not cs:
            summary.append({"subset": sname, "note": "no comparison in this subset"})
            continue
        for qname, qs in qsets.items():
            for metric in ("D", "D_common", "ICC"):
                ms = methods + (["S8_clockbase"] if qs == ["GrimAge2"] else [])
                v = {m: agg(obs, cs, qs, m, metric) for m in ms}
                hib = metric == "ICC"
                best = (max if hib else min)(RAW_VARIANTS, key=lambda m: v[m])
                bd = np.array([agg(b, cs, qs, "MethylGPT", metric) - agg(b, cs, qs, best, metric) for b in boots])
                rel = (v["MethylGPT"] - v[best]) / v[best] if hib else 1 - v["MethylGPT"] / v[best]
                br = np.array([((agg(b, cs, qs, "MethylGPT", metric) - agg(b, cs, qs, best, metric)) / agg(b, cs, qs, best, metric)) if hib
                               else 1 - agg(b, cs, qs, "MethylGPT", metric) / agg(b, cs, qs, best, metric) for b in boots])
                row = {"subset": sname, "comparisons": len(cs), "quantities": qname, "metric": metric,
                       **{f"value_{m}": v[m] for m in ms}, "best_comparator": best, "best_comparator_value": v[best],
                       "difference_methylgpt_minus_best": v["MethylGPT"] - v[best],
                       "difference_ci95_low": float(np.percentile(bd, 2.5)), "difference_ci95_high": float(np.percentile(bd, 97.5)),
                       "relative_gain": rel, "relative_gain_ci95_low": float(np.percentile(br, 2.5)), "relative_gain_ci95_high": float(np.percentile(br, 97.5)),
                       "comparisons_methylgpt_better": int(sum((np.mean([obs[(c, q, 'MethylGPT')][metric] for q in qs]) > np.mean([obs[(c, q, best)][metric] for q in qs]))
                                                               if hib else (np.mean([obs[(c, q, 'MethylGPT')][metric] for q in qs]) < np.mean([obs[(c, q, best)][metric] for q in qs])) for c in cs))}
                summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary_metrics.csv", index=False)
    per = pd.DataFrame([{"comparison_id": c, "gse": meta[c]["gse"], "comparison": meta[c]["comparison"], "n_pairs": meta[c]["n_pairs"],
                         "any_pair_in_training_split": meta[c]["any_pair_in_training_split"], "quantity": q, "method": m, **obs[(c, q, m)]}
                        for (c, q, m) in obs])
    per.to_csv(OUT / "per_comparison_metrics.csv", index=False)
    # S5 scale check
    sc = []
    for c in cids:
        for q in QUANT:
            for v in RAW_VARIANTS:
                for role, k in (("A", 0), ("B", 1)):
                    mgv, rv = vals[(c, q, "MethylGPT")][k], vals[(c, q, v)][k]
                    sc.append({"comparison_id": c, "quantity": q, "raw_variant": v, "array_role": role, "n": len(mgv),
                               "pearson_r": float(np.corrcoef(mgv, rv)[0, 1]), "sd_ratio_methylgpt_over_raw": float(np.std(mgv, ddof=1) / np.std(rv, ddof=1))})
    sc = pd.DataFrame(sc)
    pooled = []
    for q in QUANT:
        for v in RAW_VARIANTS:
            for role, k in (("A", 0), ("B", 1)):
                mgv = np.concatenate([vals[(c, q, "MethylGPT")][k] for c in cids]); rv = np.concatenate([vals[(c, q, v)][k] for c in cids])
                t = sc[(sc.quantity == q) & (sc.raw_variant == v) & (sc.array_role == role)]
                pooled.append({"comparison_id": "mean over comparisons", "quantity": q, "raw_variant": v, "array_role": role, "n": int(t.n.sum()),
                               "pearson_r": float(t.pearson_r.mean()), "sd_ratio_methylgpt_over_raw": float(t.sd_ratio_methylgpt_over_raw.mean())})
                pooled.append({"comparison_id": "pooled", "quantity": q, "raw_variant": v, "array_role": role, "n": len(mgv),
                               "pearson_r": float(np.corrcoef(mgv, rv)[0, 1]), "sd_ratio_methylgpt_over_raw": float(np.std(mgv, ddof=1) / np.std(rv, ddof=1))})
    sc = pd.concat([sc, pd.DataFrame(pooled)], ignore_index=True)
    sc.to_csv(OUT / "scale_check.csv", index=False)
    # rule outcome (primary)
    p = summary[(summary.subset == "all comparisons") & (summary.quantities == "primary (GrimAge2, DunedinPACE)") & (summary.metric == "D")].iloc[0]
    passed = bool(p.relative_gain >= 0.10 and p.difference_ci95_high < 0)
    chk = sc[(sc.comparison_id == "mean over comparisons") & (sc.array_role == "A") & (sc.raw_variant == p.best_comparator)].set_index("quantity")
    rule = pd.DataFrame([{"analysis": "C clock agreement between arrays in the same people",
                          "primary_metric": "mean over comparisons and over GrimAge2 and DunedinPACE of mean |y_B - y_A| / SD(y_A), each method on its own SD (lower is better)",
                          "comparisons": int(p.comparisons), "methylgpt": p.value_MethylGPT, "best_comparator": p.best_comparator,
                          "best_comparator_value": p.best_comparator_value, "other_comparator": [m for m in RAW_VARIANTS if m != p.best_comparator][0],
                          "other_comparator_value": p[f"value_{[m for m in RAW_VARIANTS if m != p.best_comparator][0]}"],
                          "relative_gain": p.relative_gain, "difference": p.difference_methylgpt_minus_best,
                          "difference_ci95_low": p.difference_ci95_low, "difference_ci95_high": p.difference_ci95_high,
                          "rule": "pass if relative gain >= 0.10 and the 95% interval of the difference lies below zero", "pass": passed,
                          "scale_check_mean_r_GrimAge2_array_A": float(chk.loc["GrimAge2", "pearson_r"]),
                          "scale_check_mean_r_DunedinPACE_array_A": float(chk.loc["DunedinPACE", "pearson_r"]),
                          "scale_check_sd_ratio_GrimAge2_array_A": float(chk.loc["GrimAge2", "sd_ratio_methylgpt_over_raw"]),
                          "scale_check_sd_ratio_DunedinPACE_array_A": float(chk.loc["DunedinPACE", "sd_ratio_methylgpt_over_raw"])}])
    rule.to_csv(OUT / "rule_outcome.csv", index=False)
    cov = pd.read_csv(require(WORK / "coverage.csv"))
    cov.to_csv(OUT / "coverage.csv", index=False)
    chk_mg = json.loads(require(WORK / "methylgpt_check.json").read_text())
    write_json(OUT / "results.json", {"rule_outcome": rule.iloc[0].to_dict(), "comparisons": meta, "methylgpt_package_check": chk_mg,
                                      "bootstrap": f"{DRAWS} draws, seed {SEED}, individuals resampled within comparisons"})
    write_readme()
    log(f"primary: MethylGPT {p.value_MethylGPT:.4f} vs {p.best_comparator} {p.best_comparator_value:.4f}; gain {p.relative_gain:.3f}; "
        f"difference {p.difference_methylgpt_minus_best:.4f} [{p.difference_ci95_low:.4f}, {p.difference_ci95_high:.4f}]; pass {passed}")


# ------------------------------------------------------------------ ICC and metrics
def icc_a1(a, b):
    """ICC(A,1): two-way random effects, absolute agreement, single measurement (McGraw and Wong 1996)."""
    Y = np.stack([a, b], 1).astype(float)
    n, k = Y.shape
    gm, rm, cm = Y.mean(), Y.mean(1), Y.mean(0)
    msr = k * ((rm - gm) ** 2).sum() / (n - 1)
    msc = n * ((cm - gm) ** 2).sum() / (k - 1)
    mse = ((Y - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def std_diff(a, b, scale_a=None):
    """Mean |b - a| / SD(a); scale_a replaces SD(a) by the SD of another estimate on array A (S3)."""
    sd = np.std(a if scale_a is None else scale_a, ddof=1)
    return float(np.mean(np.abs(b - a)) / sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["search", "freeze", "download", "prepare", "raw_clocks", "methylgpt", "analyse"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.stage == "search":
        stage_search()
    elif a.stage == "freeze":
        stage_freeze()
    elif a.stage == "download":
        stage_download()
    elif a.stage == "prepare":
        stage_prepare()
    elif a.stage == "raw_clocks":
        stage_raw_clocks()
    elif a.stage == "methylgpt":
        stage_methylgpt(a.device)
    elif a.stage == "analyse":
        stage_analyse()


if __name__ == "__main__":
    main()
