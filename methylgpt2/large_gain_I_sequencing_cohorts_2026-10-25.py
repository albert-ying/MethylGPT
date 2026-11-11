#!/usr/bin/env python3
"""Large-gain analysis I: few-label disease classification on public bisulfite- and enzymatic-conversion sequencing
case-control cohorts, where each sample covers its own subset of CpGs. This tests on sequencing data the result that
analysis G found under simulated missingness (G: MethylGPT-imputed CpGs 0.2165 against nearest-neighbour imputation
0.1697, mean AUC - 0.5 at n = 50, 24 held-out array datasets, each profile keeping a random 5% of its CpGs). Here every
input is an empirical per-CpG call from sequencing; nothing is masked or simulated. Every result is reported, whichever
representation it favours.

Written on 25 Oct 2026 for the large-gain analysis plan of 23 Oct 2026 (Decision rule; analysis A's protocol) and the
analysis brief for I, BEFORE any GEO search result is used, any dataset is downloaded or any model is fitted. The
released MethylGPT is used exactly as G locked it (release bundle artifacts/release/methylgpt2-1.1, MethylGPT.impute,
factor method; G/locked_setting_2026-10-25.json: representation "imputed"); nothing is trained. Analysis A's protocol
(draws, classifier, metric, bootstrap) is imported from large_gain_A_disease_few_labels_2026-10-24.py through G
(large_gain_G_sparse_inputs_2026-10-25.py) and F. Nothing under /Entropy/kying/09_shallowBS_is_all_you_need is used.

1. DATA SEARCH (stage search; run once, date and query translations recorded)
  NCBI E-utilities esearch, db = gds, every query AND "Homo sapiens"[Organism] AND "gse"[Entry Type], all UIDs retrieved
  (retmax 5,000 per page, paged). The six queries:
    Q1 RRBS        RRBS[All Fields] OR ERRBS[All Fields] OR "reduced representation bisulfite"[All Fields] OR
                   "reduced representation bisulphite"[All Fields]
    Q2 targeted    "targeted bisulfite"[All Fields] OR "targeted bisulphite"[All Fields] OR "bisulfite amplicon"[All Fields]
                   OR "amplicon bisulfite"[All Fields] OR "bisulfite padlock"[All Fields] OR "padlock probe"[All Fields] OR
                   "padlock probes"[All Fields] OR "targeted methylation sequencing"[All Fields] OR "targeted DNA methylation
                   sequencing"[All Fields] OR "bisulfite sequencing panel"[All Fields]
    Q3 capture     "methylation capture"[All Fields] OR "MC-seq"[All Fields] OR "capture bisulfite"[All Fields] OR
                   "bisulfite capture"[All Fields] OR "SureSelect Methyl"[All Fields] OR "SureSelectXT Methyl"[All Fields]
                   OR "SeqCap Epi"[All Fields] OR "TruSeq Methyl"[All Fields] OR "Methyl-Seq"[All Fields] OR "Twist Human
                   Methylome"[All Fields]
    Q4 WGBS        "whole genome bisulfite"[All Fields] OR "whole-genome bisulfite"[All Fields] OR "whole genome
                   bisulphite"[All Fields] OR WGBS[All Fields] OR "low-pass bisulfite"[All Fields] OR "low coverage
                   bisulfite"[All Fields] OR "low-coverage bisulfite"[All Fields] OR "shallow bisulfite"[All Fields] OR
                   PBAT[All Fields] OR "post-bisulfite adaptor"[All Fields]
    Q5 EM-seq      "EM-seq"[All Fields] OR EMseq[All Fields] OR "enzymatic methyl-seq"[All Fields] OR "enzymatic methyl
                   seq"[All Fields] OR "enzymatic methylation sequencing"[All Fields] OR "enzymatic conversion"[All Fields]
    Q6 broad net   "methylation profiling by high throughput sequencing"[DataSet Type]
  Hits: the union of UIDs (GSE accession = "GSE" + (UID - 200000000)), with the queries each hit matched.
  Written: I/search_queries_2026-10-25.json (queries, translations, counts, time), I/search_hits_2026-10-25.csv.

2. SCREENING (stage screen fetches GEO records; stage freeze applies the rules). Every hit gets one row in
  I/screening_2026-10-25.csv with the first rule it fails (reason) or "included". Rules, applied in this order:
    S0 (automated, esummary)  a Series whose taxa include Homo sapiens.
    S1 (automated, esummary)  at least 60 samples in the series (fewer cannot hold 30 cases and 30 controls).
                              A SuperSeries passing S1 is not screened itself ("SuperSeries; SubSeries screened");
                              its SubSeries (Series_relation of the series record) are added as hits ("added: SubSeries
                              of GSE...") and screened by all rules.
    S2 (automated, GSM records, acc.cgi targ=gsm view=brief)  at least 60 human samples assayed by bisulfite or
                              enzymatic-conversion sequencing: Sample_library_strategy "Bisulfite-Seq", or "OTHER" with
                              bisulfite/bisulphite/EM-seq/enzymatic methyl/enzymatic conversion in the sample's protocol
                              or processing text; samples whose title, source or characteristics mark oxBS, TAB-seq or
                              oxidative bisulfite are not counted; array samples (no library strategy) and MeDIP, MBD,
                              MRE, cfMeDIP or other enrichment-only assays are not counted.
    S3 (read: series title, summary and design; GSM titles, source names, characteristics and descriptions)  a disease
                              case-control contrast is recorded in the GEO metadata. Roles: case = in-vivo human sample
                              with one disease or disease outcome recorded (A's sense: not a control label, not
                              non-specific, not an exposure such as smoking, diet, drugs or assisted conception; age or sex
                              groups are not diseases); control = in-vivo sample of the same study recorded as healthy,
                              normal, control, or without that disease (for example "POAF: no"; the README states which).
                              Not cases or controls: cell lines, in-vitro culture or treatment, xenografts, organoids,
                              post-treatment or post-intervention time points when a pre-treatment sample of the same
                              individual exists, replicate libraries (the first listed GSM of an individual is kept).
                              Contrast: A's rule (choose_contrast): within one tissue, one disease against the controls of
                              that tissue; among strata with >= 30 cases and >= 30 controls the one with most cases (ties:
                              most controls, then tissue and disease alphabetically). The label rule (GEO field, case and
                              control values, tissue) is written per dataset in I/label_rules_2026-10-25.json and applied
                              by code to the GSM records.
    S4 (read: series and sample supplementary file names, Sample_data_processing)  processed per-CpG methylation calls with
                              per-sample read counts deposited at GEO for the contrast samples: Bismark coverage
                              (.cov/.bismark.cov) or CpG reports, methylKit or BS-Seeker2 call files, CGmap, MethylDackel or
                              bedGraph/BED files with counts, or matrices of methylated and total (or unmethylated) counts,
                              or of beta values together with coverage. A beta-only file or matrix is accepted only when the
                              processing text states a per-CpG minimum of >= 10 reads (non-missing values are then
                              observed). Excluded: region-, amplicon-, block- or DMR-level summaries; calls deposited only
                              as raw reads (SRA) or under controlled access (no realignment is done here).
    S5 (automated)            overlap with the MethylGPT corpus (artifacts/corpus_2026-09-30/profiles.parquet): contrast
                              GSMs among profiles.id, and the accession among profiles.group; counts recorded. A dataset
                              with any contrast profile in the corpus training split is reported separately, not in the
                              primary set (expected none: the corpus is array data).
    S6 (automated)            series sharing contrast samples (for example a SubSeries and a re-analysis series): the one
                              with more contrast samples is kept (ties: lower accession), as in A.
  Freezing: stage freeze writes I/datasets_frozen_I_2026-10-25.csv (accession, assay, tissue, disease, case and control
  counts from the metadata, label rule, corpus overlap, supplementary files) and the per-dataset sample tables
  I/samples_<GSE>_2026-10-25.csv (GSM, role, individual); these and the screening table are committed BEFORE any
  download, mapping or fit. If no dataset passes, the README states "Cannot proceed - required data file not found:"
  followed by the search and its counts, and the analysis stops.

3. DOWNLOAD AND MAPPING (stages download, map). Files go to artifacts/external/<GSE>/ (raw data not committed).
  Mapping follows the existing RRBS pipeline (artifacts/external/GSE194154/scripts/03_build_matrix.py,
  rrbs_genomewide_matrix_2026-10-12.py):
    Vocabulary coordinates: the Zhou-lab EPIC v2 manifest of the dataset's genome build (EPICv2.hg38 or EPICv2.hg19
      .manifest.tsv.gz, downloaded on 25 Oct 2026 from github.com/zhou-lab/InfiniumAnnotationData into
      artifacts/external/zhou_manifests_2026-10-25/; the pipeline used the same public hg38 file): suffix stripped from
      Probe_ID, first row per ID, CpG targets of 2 bp on chr1-22, X, Y, M; CpG_beg is the 0-based start of the CG, the
      + strand C is at 1-based CpG_beg + 1. Checked (stage map writes the result): hg38 coordinates identical to
      artifacts/external/GSE194154/cpg_mapping.csv, and hg19 coordinates on a CG of the hg19 FASTA.
    Genome build: the build stated in the GSM records (genome build or data processing text). Position convention
      (1-based C position or 0-based start): the one under which the larger fraction of >= 20,000 call rows sampled
      from up to 5 samples lies on a reference CpG cytosine (hg38: /Entropy/Projects/DNA_Mech/Data/
      GRCh38.primary_assembly.genome.fa; hg19: /Entropy/Projects/DNA_Mech/Data/hg19.fa); rows whose context column
      marks a non-CpG context are removed first. If the stated build gives < 90% under both conventions the other build is
      tried; if neither reaches 90% the dataset is excluded ("coordinates do not match hg19 or hg38"). The fraction and
      the number of vocabulary CpGs matched under each choice are written (I/mapping_check_2026-10-25.csv).
    Strands: a row on the + strand C (reference C at P, G at P + 1) is keyed to P; a row on the - strand C (reference G at
      P, C at P - 1) is keyed to P - 1; methylated and total reads of the two strands (and of duplicate rows) are summed,
      beta = methylated / total.
    Counts: methylated and total reads as deposited; methylKit-format files with truncated integer percentages use the
      pipeline's exact count recovery (03_build_matrix.py); percent-plus-coverage formats use percent / 100 x coverage.
    Minimum coverage: a CpG is observed in a sample when it has >= 10 reads, both strands summed (the pipeline's rule).
    Vocabulary matrix X (samples x 49,156; NaN = unobserved). Genome-wide matrix G (all CpGs with >= 10 reads in >= 18
      contrast samples: an exact superset of every CpG covered in >= 90% of a training draw at n >= 20).
    Sample QC: samples with fewer than 100 vocabulary CpGs at >= 10 reads are dropped (technical failures; listed).
  Technical exclusions after the freeze (mechanical, recorded in I/datasets_analysis_2026-10-25.csv, committed before
  stage fit): T1 files cannot be downloaded or parsed; T2 coordinates match neither build; T3 fewer than 30 cases or 30
  controls with calls after sample QC; T4 the median sample has fewer than 1,000 vocabulary CpGs at >= 10 reads (about
  the sparsest setting G tested: 27K arrays at f = 0.05 kept about 1,280 CpGs; below it the dataset tests the assay's
  overlap with the array rather than the representation). Per-sample vocabulary coverage (observed CpGs) is written for
  every sample (I/coverage_per_sample_2026-10-25.csv) and summarized per dataset.

4. REPRESENTATIONS (all from the observed calls only; MethylGPT as G locked it)
  MethylGPT
    imputed    PRIMARY. Release bundle MethylGPT.impute (artifacts/release/methylgpt2-1.1, checksums verified, method
               "factor", FP32 as G): observed values unchanged, every other vocabulary CpG = mu + V s clipped to [0, 1],
               then the classifier on the 49,156 CpGs.
    embedding  secondary. A's embedding (age-stage encoder, 2,048 dimensions; F.load_encoder / F.embed_profiles, TF32 as
               A and G) from the observed CpGs.
    latent     secondary. Its first 1,536 dimensions (latent scores / prior SD).
  Comparators (the decision set)
    raw_mean   the 49,156 CpGs, unobserved values = corpus training means (mu of ppca_core_meanfilled_2026-09-30), as G.
    raw_knn    the 49,156 CpGs, unobserved values by k = 10 nearest neighbours with G's implementation
               (knn_predict_lowmem of si_reconstruction_diagnostics_2026-10-18.py; squared Euclidean distance over the
               query's observed CpGs) against G's held-out reference: corpus training-split profiles minus F's flagged
               duplicates (145,316 rows, SHA-256 39242110... in G/knn_reference_heldout_2026-10-25.json, checked), minus
               any corpus row whose id is a profile of these datasets (count recorded; expected 0).
    clocks     GrimAge2, DunedinPACE, PhenoAge and Horvath (Horvathv1) from the mean-filled profile (observed values,
               other vocabulary CpGs = mu) with G's stage clocks (biolearn 0.7.1, A's GrimAge2 convention), one feature
               set of four clocks.
    raw_cov90  the vocabulary CpGs observed (>= 10 reads) in >= 90% of the training draw's samples (ceil(0.9 n)); chosen
               per draw from the training samples only; missing values of those CpGs, in training and scoring samples,
               take the training-draw mean of the CpG (A's rule for missing feature values); no other CpGs. A draw with no
               qualifying CpG scores AUC 0.5 (counted).
  Secondary comparator (reported, not in the decision set, which follows the brief)
    raw_cov90_genome  as raw_cov90 over the genome-wide matrix G (every CpG the assay covers, not only vocabulary CpGs).

5. CLASSIFIER, DRAWS, METRIC (A's, unchanged)
  A.draws_for: balanced draws of n/2 cases and n/2 controls, n in {20, 50, 100}, 50 draws per size, rng
  default_rng([20261024, accession number, n, draw]); a size is evaluated when >= n/2 + 10 cases and controls remain.
  A.fit_select_score: logistic elastic net, alpha in {0.1, 0.5, 0.9} x 10 lambdas, stratified 5-fold CV within the
  draw (fold seed = draw index), FP32 as A's fit stage, called per dataset and size as G (raw_cov90 and
  raw_cov90_genome: per draw, their column set being draw-specific). AUC on all remaining contrast samples. Dataset value =
  mean AUC over the 50 draws. M(rep) = mean over datasets of (dataset value - 0.5).

6. PRIMARY RESULT AND DECISION (plan's Decision rule)
  PRIMARY: M(imputed) at n = 50 over the datasets evaluable at n = 50 (every representation is available for every
  dataset). Best comparator C* = the comparator in {raw_mean, raw_knn, clocks, raw_cov90} with the largest M on the same
  datasets (fixed on the full set). Relative gain = (M(imputed) - M(C*)) / M(C*). PASS iff the relative gain is >= 10%
  and the lower end of the 95% interval of M(imputed) - M(C*) is > 0; otherwise FAIL.
  If no dataset is evaluable at n = 50, the primary result is computed at n = 20 (the brief's size cannot be evaluated;
  stated in the README and the report).
  95% interval: with K >= 5 primary datasets, A.compare's paired bootstrap over datasets (2,000 resamples, seed 20261024,
  one resample of datasets shared by all representations). With K < 5 datasets, a paired bootstrap over samples within
  datasets: per dataset, replicate b resamples cases and controls separately with replacement (rng
  default_rng([20261024, accession number]), 2,000 replicates, the same resample for every representation); the fitted
  models and their scores are fixed (refitting is not resampled); each draw's AUC is recomputed on its scoring samples
  with the bootstrap multiplicities as weights (a draw whose scoring samples get zero weight in a class is skipped in that
  replicate); M_b = mean over datasets of (mean over draws - 0.5); interval = 2.5th and 97.5th percentiles of
  M_b(imputed) - M_b(C*). The method used is stated with the result.
  SECONDARY (all reported): embedding and latent against C*; MethylGPT minus each comparator separately (interval of the
  same kind); n = 20 and 100; raw_cov90_genome and MethylGPT minus it; per-dataset values with within-dataset sample
  bootstrap intervals (same resampling as above) for every representation and size; the other kind of interval when
  K >= 5; per-sample and per-dataset vocabulary coverage; selection diagnostics of the classifier (selected alpha, lambda
  index, nonzero coefficients, draws without a nonzero coefficient, raw_cov90 feature counts).

7. SEEDS AND RESAMPLES: draws A's (base 20261024, 50 per size); inner folds seeded by the draw index; bootstrap 2,000
  replicates, seed 20261024; kNN and imputation deterministic. GPU choice (cuda:0 or cuda:1, whichever has more free
  memory) and batch sizes are execution only.

NOTES AT THE FREEZE (25 Oct 2026, after reading the hits and before any download, mapping or fit; no design change):
  (a) Entrez dropped quoted phrases absent from its phrase index (for example "bisulfite amplicon", "capture bisulfite");
      the translated queries that ran are recorded in I/search_queries_2026-10-25.json.
  (b) One sample per individual and role: the first listed GSM, applied also to repeated sampling without treatment
      (GSE282512: several gestational time points per patient id).
  (c) S6 counts a GSM and the GSM it re-analyses (Sample_relation "Reanalysis of") as the same sample (GSE215937
      re-analyses GSE194154's pre-operative samples).
  (d) Series with several diseases: the case disease is read per sample from the label field and A's contrast rule
      chooses the stratum (label_rules: disease_from_value).
  (e) Reading decisions are in I/screening_read_2026-10-25.csv (S3, S4 with reasons); I/screening_2026-10-25.csv merges
      them with the automated rules; 13 datasets are frozen (commit f61b8cc).

AMENDMENT 26 Oct 2026 (code only; no design change; before any mapping output, representation or fit):
  (1) Downloads: files of a dataset are fetched by parallel threads (I_DL_THREADS, execution only); HTTP 503 failures
      were fetched by two further passes with fewer connections; all 1,504 files then passed gzip -t
      (work_I/logs/gzip_check.txt empty). The downloads ran with this threaded code before this amendment was committed;
      file contents are unaffected.
  (2) Call reading: chromosome names are read as categories and carried as integer codes (CHR2I) instead of strings
      (memory for genome-wide WGBS reports of about 56 million rows); same rows kept, same keys.
  (3) GSE93203: its deposited file is one series matrix (rows = sample titles; per CpG marker X<chr>.<pos> the columns
      beta, T, C); a reader takes methylated = C and total = C + T per sample (the S4 rule's matrix of counts).
  (4) Coordinate-convention scan: ties between the 1-based and 0-based readings go to 1-based, the pipeline's convention
      (only possible when every checked row is a + strand C; the CpG keys are then identical under both readings).
  (5) Per-dataset file formats are recorded in I/format_specs_2026-10-25.json. GSE145153: the count columns are read as
      methylated then unmethylated, as in the data (CpG rows mean 0.65, CHH rows 0.03 under that order; the reverse would
      give CHH 0.97); the GEO description lists them in the reverse order. GSE194155 has the methylKit layout of GSE194154
      and uses the same count recovery.
  A mapping run started before this commit was stopped when the session ended; its only output (work_I/GSE135748
  matrices, no table) was deleted unread and the stage is rerun from the committed code.

AMENDMENT 26 Oct 2026, second (execution only; no design change; before any mapping output is used):
  Stages map, represent, clocks and fit are resumable per dataset (and per representation in fit): outputs are written
  atomically (temporary name, then rename), a dataset is skipped only when its outputs load with the expected shape
  (mapping: work_I/map/<GSE>.json record plus matrices), and each stage writes a done-marker
  artifacts/logs/large_gain_I_<stage>_2026-10-27.done when complete. Stages run from run_large_gain_I_2026-10-27.sh
  in a transient systemd user unit (logs artifacts/logs/large_gain_I_<stage>_2026-10-27.log). T1 now also applies when fewer files were downloaded than
  expected (all 13 datasets are complete, so no dataset changes status). The second mapping run also died with the
  session (during GSE140730); its GSE135748 matrices are recomputed because no record was written.

AMENDMENT 28 Oct 2026 (execution only; no design change; before any mapping output is used):
  The runner unit large-gain-I-2026-10-27 was killed by systemd-oomd during stage map (peak 124.9 GB with 16 parser
  processes; no new mapping record was written). Stage map takes --workers (parser processes, default 16) and the
  runner uses 4, with unbuffered logging. Records and matrices are unchanged by the number of workers.

Stages: search, screen, freeze, download, map, represent, clocks (anaconda python with biolearn; no torch), fit, analyse.
Outputs: artifacts/large_gain_2026-10-25/I/ (README.md and aggregate tables); working files (metadata records,
matrices, representations, scores) in artifacts/large_gain_2026-10-25/work_I/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OUT = ART / "large_gain_2026-10-25"
OUTI = OUT / "I"
WORKI = OUT / "work_I"
META_DIR = WORKI / "geo_records"
EXT = ART / "external"
MANIFEST_DIR = EXT / "zhou_manifests_2026-10-25"
CORPUS_DIR = ART / "corpus_2026-09-30"
G_SCRIPT = HERE / "large_gain_G_sparse_inputs_2026-10-25.py"
DATE = "2026-10-25"
SEED = 20261024
MIN_CASE = MIN_CTRL = 30
MIN_SERIES_SAMPLES = 60
MIN_COV = 10
MIN_SAMPLE_VOCAB = 100
MIN_MEDIAN_VOCAB = 1000
COV90 = 0.90
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ACC_CGI = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
SUFFIX = ' AND "Homo sapiens"[Organism] AND "gse"[Entry Type]'
QUERIES = {
    "Q1_RRBS": 'RRBS[All Fields] OR ERRBS[All Fields] OR "reduced representation bisulfite"[All Fields] OR '
               '"reduced representation bisulphite"[All Fields]',
    "Q2_targeted": '"targeted bisulfite"[All Fields] OR "targeted bisulphite"[All Fields] OR "bisulfite amplicon"[All Fields] '
                   'OR "amplicon bisulfite"[All Fields] OR "bisulfite padlock"[All Fields] OR "padlock probe"[All Fields] OR '
                   '"padlock probes"[All Fields] OR "targeted methylation sequencing"[All Fields] OR "targeted DNA methylation '
                   'sequencing"[All Fields] OR "bisulfite sequencing panel"[All Fields]',
    "Q3_capture": '"methylation capture"[All Fields] OR "MC-seq"[All Fields] OR "capture bisulfite"[All Fields] OR '
                  '"bisulfite capture"[All Fields] OR "SureSelect Methyl"[All Fields] OR "SureSelectXT Methyl"[All Fields] '
                  'OR "SeqCap Epi"[All Fields] OR "TruSeq Methyl"[All Fields] OR "Methyl-Seq"[All Fields] OR "Twist Human '
                  'Methylome"[All Fields]',
    "Q4_WGBS": '"whole genome bisulfite"[All Fields] OR "whole-genome bisulfite"[All Fields] OR "whole genome '
               'bisulphite"[All Fields] OR WGBS[All Fields] OR "low-pass bisulfite"[All Fields] OR "low coverage '
               'bisulfite"[All Fields] OR "low-coverage bisulfite"[All Fields] OR "shallow bisulfite"[All Fields] OR '
               'PBAT[All Fields] OR "post-bisulfite adaptor"[All Fields]',
    "Q5_EMseq": '"EM-seq"[All Fields] OR EMseq[All Fields] OR "enzymatic methyl-seq"[All Fields] OR "enzymatic methyl '
                'seq"[All Fields] OR "enzymatic methylation sequencing"[All Fields] OR "enzymatic conversion"[All Fields]',
    "Q6_broad": '"methylation profiling by high throughput sequencing"[DataSet Type]',
}
CONVERSION_TEXT = re.compile(r"bisulfite|bisulphite|em-?seq|enzymatic methyl|enzymatic conversion", re.I)
OXIDATIVE = re.compile(r"\boxbs\b|oxbs-seq|oxidative bisulfite|tab-seq|\btab\b", re.I)
MGPT_REPS = ["imputed", "embedding", "latent"]
COMP_REPS = ["raw_mean", "raw_knn", "clocks", "raw_cov90"]
SECONDARY_COMP = ["raw_cov90_genome"]
PRIMARY_REP = "imputed"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def need(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    return path


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def git_committed(path):
    r = subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(path)], capture_output=True)
    if r.returncode:
        return False
    return subprocess.run(["git", "-C", str(HERE), "diff", "--quiet", "HEAD", "--", str(path)]).returncode == 0


def load_g():
    spec = importlib.util.spec_from_file_location("large_gain_G", G_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["large_gain_G"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ NCBI access
def fetch(url, params=None, tries=5, timeout=120, post=False):
    data = None
    if params and post:
        data = urllib.parse.urlencode(params).encode()
    elif params:
        url = url + "?" + urllib.parse.urlencode(params)
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers={"User-Agent": "methylgpt-revision/1.0"}),
                                        timeout=timeout) as r:
                data = r.read()
            time.sleep(0.4)                      # stay under 3 requests per second
            return data
        except Exception as e:                   # retried, then reported
            last = e
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"fetch failed after {tries} tries: {url[:200]} ({last})")


def esearch_all(term):
    first = json.loads(fetch(f"{EUTILS}/esearch.fcgi", {"db": "gds", "term": term, "retmax": 0, "retmode": "json",
                                                       "usehistory": "y"}))["esearchresult"]
    count = int(first["count"])
    ids = []
    for start in range(0, count, 5000):
        r = json.loads(fetch(f"{EUTILS}/esearch.fcgi", {"db": "gds", "term": term, "retmax": 5000, "retstart": start,
                                                       "retmode": "json"}))["esearchresult"]
        ids += r["idlist"]
    return count, sorted(set(ids)), first.get("querytranslation", "")


def esummary(uids, size=20):
    """esummary records; a batch whose JSON exceeds NCBI's 10 MB limit is split."""
    out = {}
    for i in range(0, len(uids), size):
        part = uids[i:i + size]
        j = json.loads(fetch(f"{EUTILS}/esummary.fcgi", {"db": "gds", "id": ",".join(part), "retmode": "json"}, post=True))
        if "result" not in j:
            if len(part) > 1:
                out.update(esummary(part, max(1, len(part) // 4)))
                continue
            log(f"esummary failed for {part}: {str(j)[:200]}")
            continue
        r = j["result"]
        for u in r.get("uids", []):
            out[u] = r[u]
    return out


# ------------------------------------------------------------------ stage search
def stage_search():
    OUTI.mkdir(parents=True, exist_ok=True)
    hits, info = {}, {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "db": "gds", "suffix": SUFFIX, "queries": {}}
    for name, q in QUERIES.items():
        term = f"({q}){SUFFIX}"
        count, ids, tr = esearch_all(term)
        info["queries"][name] = {"term": term, "count": count, "uids_retrieved": len(ids), "translation": tr}
        for u in ids:
            hits.setdefault(u, []).append(name)
        log(f"{name}: {count} hits")
    gse = {u: q for u, q in hits.items() if int(u) >= 200000000 and int(u) < 300000000}
    rows = [{"uid": u, "accession": f"GSE{int(u) - 200000000}", "queries": ";".join(q)} for u, q in sorted(gse.items())]
    df = pd.DataFrame(rows)
    info["union_uids"] = len(hits)
    info["union_series"] = len(df)
    df.to_csv(OUTI / f"search_hits_{DATE}.csv", index=False)
    write_json(OUTI / f"search_queries_{DATE}.json", info)
    log(f"union: {len(hits)} UIDs, {len(df)} series")


# ------------------------------------------------------------------ stage screen (fetch records, automated rules)
def parse_soft(text):
    """Records of a GEO SOFT 'brief' text: list of (accession, {field: [values]})."""
    recs, cur = [], None
    for line in text.splitlines():
        if line.startswith("^"):
            cur = {}
            recs.append((line.split("=", 1)[1].strip() if "=" in line else line[1:].strip(), cur))
        elif line.startswith("!") and cur is not None and "=" in line:
            k, v = line[1:].split("=", 1)
            cur.setdefault(k.strip(), []).append(v.strip())
    return recs


def series_record(acc):
    f = META_DIR / f"{acc}_series.txt"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(fetch(ACC_CGI, {"acc": acc, "targ": "self", "form": "text", "view": "brief"}))
    return parse_soft(f.read_text(errors="replace"))


def sample_records(acc):
    f = META_DIR / f"{acc}_gsm.txt"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(fetch(ACC_CGI, {"acc": acc, "targ": "gsm", "form": "text", "view": "brief"}, timeout=600))
    return [(a, r) for a, r in parse_soft(f.read_text(errors="replace")) if a.startswith("GSM")]


def is_conversion_seq(r):
    org = " ".join(r.get("Sample_organism_ch1", []))
    if "Homo sapiens" not in org:
        return False
    ident = " ".join(r.get("Sample_title", []) + r.get("Sample_source_name_ch1", []) + r.get("Sample_characteristics_ch1", []))
    if OXIDATIVE.search(ident):
        return False
    strat = " ".join(r.get("Sample_library_strategy", [])).strip()
    if strat == "Bisulfite-Seq":
        return True
    if strat == "OTHER":
        txt = " ".join(r.get("Sample_extract_protocol_ch1", []) + r.get("Sample_data_processing", []) +
                       r.get("Sample_description", []) + r.get("Sample_library_selection", []))
        return bool(CONVERSION_TEXT.search(txt))
    return False


def digest(acc, srec, gsm):
    """Text digest of a series for reading (S3, S4)."""
    s = srec[0][1] if srec else {}
    conv = [(a, r) for a, r in gsm if is_conversion_seq(r)]
    lines = [f"===== {acc}: {' '.join(s.get('Series_title', []))}",
             f"type: {'; '.join(s.get('Series_type', []))} | samples {len(gsm)} | conversion-seq human {len(conv)} | "
             f"platforms {';'.join(sorted(set(p for _, r in gsm for p in r.get('Sample_platform_id', []))))}",
             "summary: " + " ".join(s.get("Series_summary", []))[:1500],
             "design: " + " ".join(s.get("Series_overall_design", []))[:800],
             "series files: " + " | ".join(x.rsplit("/", 1)[-1] for x in s.get("Series_supplementary_file", [])),
             "relations: " + " | ".join(s.get("Series_relation", []))[:400]]
    keys = {}
    for _, r in conv:
        for c in r.get("Sample_characteristics_ch1", []):
            k, _, v = c.partition(":")
            keys.setdefault(k.strip().lower(), []).append(v.strip())
        keys.setdefault("_source", []).extend(r.get("Sample_source_name_ch1", []))
        keys.setdefault("_title_prefix", []).append(re.sub(r"[\d_\-\s]+$", "", " ".join(r.get("Sample_title", [])))[:40])
    for k, vs in keys.items():
        vc = pd.Series(vs).value_counts()
        lines.append(f"  [{k}] ({vc.size} values) " + "; ".join(f"{i}={n}" for i, n in vc.head(12).items()))
    ext = pd.Series([re.sub(r"^GSM\d+_?", "", x.rsplit("/", 1)[-1]) for _, r in conv
                     for k, v in r.items() if k.startswith("Sample_supplementary_file") for x in v if x and x != "NONE"])
    if len(ext):
        suf = ext.str.extract(r"((?:\.[A-Za-z0-9]+){1,3})$")[0].value_counts()
        lines.append("sample files: " + "; ".join(f"{i}={n}" for i, n in suf.head(8).items()) + " e.g. " + " | ".join(ext.head(3)))
    else:
        lines.append("sample files: none")
    dp = pd.Series([" ".join(r.get("Sample_data_processing", [])) for _, r in conv]).value_counts()
    if len(dp):
        lines.append("processing: " + dp.index[0][:1200])
    gb = pd.Series([" ".join(v) for _, r in conv for k, v in r.items() if "genome" in k.lower() and "build" in k.lower()])
    if len(gb):
        lines.append("genome build field: " + "; ".join(gb.value_counts().head(3).index))
    return "\n".join(lines)


def stage_screen():
    hits = pd.read_csv(need(OUTI / f"search_hits_{DATE}.csv"))
    summ = esummary([str(u) for u in hits.uid])
    rows, queue, seen = [], list(hits.itertuples()), set()
    corpus = pd.read_parquet(need(CORPUS_DIR / "profiles.parquet"), columns=["id", "group", "split"])
    cid = dict(zip(corpus.id.astype(str), corpus.split.astype(str)))
    cgroups = set(corpus.group.astype(str))
    digests = []
    while queue:
        h = queue.pop(0)
        acc = h.accession
        if acc in seen:
            continue
        seen.add(acc)
        s = summ.get(str(h.uid), {}) if hasattr(h, "uid") and not pd.isna(h.uid) else {}
        row = {"accession": acc, "queries": h.queries, "title": s.get("title", ""), "n_samples": s.get("n_samples"),
               "taxon": s.get("taxon", ""), "gdstype": s.get("gdstype", ""), "suppfile": s.get("suppfile", ""),
               "pdat": s.get("pdat", "")}
        if not s:
            srec = series_record(acc)
            r0 = srec[0][1] if srec else {}
            row.update(title=" ".join(r0.get("Series_title", [])), taxon=";".join(r0.get("Series_sample_taxid", [])),
                       n_samples=len(r0.get("Series_sample_id", [])), gdstype=";".join(r0.get("Series_type", [])))
        row["S0_human"] = "Homo sapiens" in str(row["taxon"]) or "9606" in str(row["taxon"])
        row["S1_ge60_samples"] = bool(row["n_samples"]) and int(row["n_samples"]) >= MIN_SERIES_SAMPLES
        if not row["S0_human"]:
            row["reason"] = "S0: not a human series"
        elif not row["S1_ge60_samples"]:
            row["reason"] = f"S1: fewer than {MIN_SERIES_SAMPLES} samples in the series"
        else:
            srec = series_record(acc)
            r0 = srec[0][1] if srec else {}
            summary = " ".join(r0.get("Series_summary", []))
            subs = [m.group(1) for rel in r0.get("Series_relation", []) if rel.lower().startswith("superseries of")
                    for m in [re.search(r"(GSE\d+)", rel)] if m]
            if subs or "This SuperSeries is composed of" in summary:
                row["reason"] = "S1: SuperSeries; SubSeries screened"
                row["subseries"] = ";".join(subs)
                for sub in subs:
                    if sub not in seen:
                        queue.append(SimpleNamespace(accession=sub, queries=f"added: SubSeries of {acc}", uid=np.nan))
            else:
                gsm = sample_records(acc)
                conv = [a for a, r in gsm if is_conversion_seq(r)]
                row["n_gsm_records"] = len(gsm)
                row["n_human_conversion_seq"] = len(conv)
                row["corpus_gsm_overlap"] = int(sum(a in cid for a in conv))
                row["corpus_gsm_overlap_train"] = int(sum(cid.get(a) == "train" for a in conv))
                row["corpus_group_overlap"] = acc in cgroups
                if len(conv) < MIN_SERIES_SAMPLES:
                    row["reason"] = f"S2: fewer than {MIN_SERIES_SAMPLES} human bisulfite/enzymatic-conversion sequencing samples"
                else:
                    row["reason"] = "to read (S3, S4)"
                    digests.append(digest(acc, srec, gsm))
        rows.append(row)
        if len(rows) % 100 == 0:
            log(f"screened {len(rows)} series")
    df = pd.DataFrame(rows)
    df.to_csv(OUTI / f"screening_auto_{DATE}.csv", index=False)
    (WORKI / f"digests_{DATE}.txt").write_text("\n\n".join(digests))
    log(f"automated screening: {df.reason.value_counts().to_dict()}")


# ------------------------------------------------------------------ stage freeze
def gsm_field(r, field):
    """Value of a label field of a GSM record: a characteristics key (lower case, before ':'), or _title, _source,
    _description."""
    if field == "_title":
        return " ".join(r.get("Sample_title", [])).strip()
    if field == "_source":
        return " ".join(r.get("Sample_source_name_ch1", [])).strip()
    if field == "_description":
        return " ".join(r.get("Sample_description", [])).strip()
    for c in r.get("Sample_characteristics_ch1", []):
        k, _, v = c.partition(":")
        if k.strip().lower() == field:
            return v.strip()
    return ""


def value_match(v, pats):
    v = str(v).strip().lower()
    for p in pats:
        p = str(p)
        if p.startswith("re:"):
            if re.search(p[3:], v, re.I):
                return True
        elif v == p.strip().lower():
            return True
    return False


def apply_label_rule(acc, rule):
    """Sample table (gsm, role, disease, tissue, individual, order) of a dataset under its label rule."""
    gsm = sample_records(acc)
    rows = []
    for k, (a, r) in enumerate(gsm):
        if not is_conversion_seq(r):
            continue
        if any(value_match(gsm_field(r, ex["field"]), ex["values"]) for ex in rule.get("exclude", [])):
            role = "excluded"
        elif value_match(gsm_field(r, rule["case"]["field"]), rule["case"]["values"]):
            role = "case"
        elif value_match(gsm_field(r, rule["control"]["field"]), rule["control"]["values"]):
            role = "control"
        else:
            role = "none"
        tissue = rule["tissue"]["value"] if "value" in rule["tissue"] else gsm_field(r, rule["tissue"]["field"]).lower() or "unlabelled"
        ind = gsm_field(r, rule["individual"]) if rule.get("individual") else a
        disease = rule["disease"]
        if role == "case" and rule["case"].get("disease_from_value"):
            disease = rule["case"].get("names", {}).get(gsm_field(r, rule["case"]["field"]), gsm_field(r, rule["case"]["field"]).lower())
        rows.append({"gsm": a, "role": role, "disease": disease, "tissue": tissue, "individual": ind or a,
                     "order": k, "title": " ".join(r.get("Sample_title", []))[:80]})
    st = pd.DataFrame(rows)
    if not len(st):
        return st
    # one sample per individual and role (the first listed GSM), as in the label rules of the docstring
    st["duplicate_of_individual"] = st.duplicated(["individual", "role"], keep="first") & st.role.isin(["case", "control"])
    return st


def stage_freeze():
    G = load_g()
    A = G.load_f().A
    auto = pd.read_csv(need(OUTI / f"screening_auto_{DATE}.csv"))
    read = pd.read_csv(need(OUTI / f"screening_read_{DATE}.csv"))
    rules = json.loads(need(OUTI / f"label_rules_{DATE}.json").read_text())
    corpus = pd.read_parquet(need(CORPUS_DIR / "profiles.parquet"), columns=["id", "group", "split"])
    cid = dict(zip(corpus.id.astype(str), corpus.split.astype(str)))
    cgroups = set(corpus.group.astype(str))
    tab = auto.merge(read, on="accession", how="left")
    to_read = tab.reason == "to read (S3, S4)"
    missing = sorted(set(tab.accession[to_read]) - set(read.accession))
    if missing:
        sys.exit(f"series to read without a recorded decision: {missing[:10]} ({len(missing)})")
    contrast, samples = {}, {}
    for i, row in tab[to_read].iterrows():
        acc = row.accession
        if row.S3 != "pass":
            tab.loc[i, "reason"] = f"S3: {row.S3_note}"
            continue
        if row.S4 != "pass":
            tab.loc[i, "reason"] = f"S4: {row.S4_note}"
            continue
        if acc not in rules:
            sys.exit(f"{acc}: S3 and S4 pass but no label rule")
        st = apply_label_rule(acc, rules[acc])
        use = st[st.role.isin(["case", "control"]) & ~st.duplicate_of_individual]
        cc, other = A.choose_contrast(use)
        if cc is None:
            tab.loc[i, "reason"] = "S3: fewer than 30 cases or 30 controls in one tissue under the label rule"
            continue
        sub = use[A.contrast_mask(use, cc)]
        st["in_contrast"] = st.gsm.isin(set(sub.gsm))
        samples[acc] = st
        contrast[acc] = (cc, other)
        tab.loc[i, "corpus_gsm_overlap"] = int(sum(g in cid for g in sub.gsm))
        tab.loc[i, "corpus_gsm_overlap_train"] = int(sum(cid.get(g) == "train" for g in sub.gsm))
        tab.loc[i, "corpus_group_overlap"] = acc in cgroups
        tab.loc[i, "reason"] = "passes S0-S4"
    # S5 and S6
    passing = [a for a in samples]
    for a in passing:
        k = tab.index[tab.accession == a][0]
        if tab.loc[k, "corpus_gsm_overlap_train"] > 0:
            tab.loc[k, "reason"] = "S5: contrast profiles in the corpus training split (reported separately)"
    def identities(a):
        """Contrast GSMs plus the GSMs they re-analyse (Sample_relation 'Reanalysis of'), so that a re-analysis series
        and its source share samples (S6)."""
        recs = dict(sample_records(a))
        out = set()
        for g in samples[a].gsm[samples[a].in_contrast]:
            out.add(g)
            for rel in recs[g].get("Sample_relation", []):
                m = re.match(r"\s*Reanalysis of:\s*(GSM\d+)", rel)
                if m:
                    out.add(m.group(1))
        return out
    sets = {a: identities(a) for a in passing}
    for a in passing:
        for b in passing:
            if a != b and sets[a] & sets[b]:
                keep = sorted([a, b], key=lambda x: (-int(samples[x].in_contrast.sum()), x))[0]
                if keep != a:
                    tab.loc[tab.accession == a, "reason"] = f"S6: shares contrast samples with {b} (kept)"
    tab["included"] = tab.reason == "passes S0-S4"
    tab.loc[tab.included, "reason"] = "included"
    tab.to_csv(OUTI / f"screening_{DATE}.csv", index=False)
    fr = []
    for a in tab.accession[tab.included]:
        cc, other = contrast[a]
        st = samples[a]
        st.to_csv(OUTI / f"samples_{a}_{DATE}.csv", index=False)
        r = rules[a]
        row = tab[tab.accession == a].iloc[0]
        fr.append({"accession": a, "assay": r.get("assay", ""), "disease": cc["disease"], "tissue": cc["tissue"],
                   "n_case": cc["n_case"], "n_ctrl": cc["n_ctrl"], "control_type": r.get("control_type", ""),
                   "other_strata": other, "label_rule": json.dumps(r, sort_keys=True), "files": r.get("files", ""),
                   "genome_build_stated": r.get("genome_build", ""), "corpus_gsm_overlap": int(row.corpus_gsm_overlap),
                   "corpus_gsm_overlap_train": int(row.corpus_gsm_overlap_train),
                   "corpus_group_overlap": bool(row.corpus_group_overlap), "queries": row.queries, "title": row.title,
                   "accession_number": int(A.accession_number(a)),
                   "samples_sha256": hashlib.sha256("\n".join(sorted(st.gsm[st.in_contrast])).encode()).hexdigest()})
    fr = pd.DataFrame(fr)
    fr.to_csv(OUTI / f"datasets_frozen_I_{DATE}.csv", index=False)
    log(f"screening: {tab.reason.str.split(':').str[0].value_counts().to_dict()}; frozen datasets: {len(fr)}")
    if not len(fr):
        q = json.loads(need(OUTI / f"search_queries_{DATE}.json").read_text())
        msg = ("Cannot proceed - required data file not found: no GEO series passed the screening (search "
               + "; ".join(f"{k} {v['count']} hits" for k, v in q["queries"].items())
               + f"; {q['union_series']} series; screening counts {tab.reason.value_counts().to_dict()})")
        (OUTI / "README.md").write_text("# Analysis I\n\n" + msg + "\n")
        log(msg)


# ------------------------------------------------------------------ genome and vocabulary coordinates
CHROMS = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]]
CHR2I = {c: i + 1 for i, c in enumerate(CHROMS)}
SCALE = np.int64(10**9)
FASTA = {"hg38": Path("/Entropy/Projects/DNA_Mech/Data/GRCh38.primary_assembly.genome.fa"),
         "hg19": Path("/Entropy/Projects/DNA_Mech/Data/hg19.fa")}


def norm_chrom(c):
    c = np.asarray(c).astype(str)
    c = np.where(np.char.startswith(c, "chr"), c, np.char.add("chr", c))
    return np.where(c == "chrMT", "chrM", c)


def chrom_code(names):
    """Chromosome codes (CHR2I; 0 = other contigs) of chromosome names ('1', 'chr1', 'MT', ...)."""
    return np.array([CHR2I.get(c, 0) for c in norm_chrom(names)], np.int16) if len(names) else np.zeros(0, np.int16)


class Genome:
    """Reference sequence per chromosome (uint8, upper case), cached as .npy under work_I/genome_<build>/."""

    def __init__(self, build):
        self.build, self.dir = build, WORKI / f"genome_{build}"
        self.fai = pd.read_csv(str(need(FASTA[build])) + ".fai", sep="\t", header=None,
                               names=["chr", "len", "off", "lb", "lbytes"]).set_index("chr")
        self.seq = {}

    def chrom(self, c):
        if c not in self.seq:
            f = self.dir / f"{c}.npy"
            if not f.exists():
                self.dir.mkdir(parents=True, exist_ok=True)
                r = self.fai.loc[c]
                nlines = (int(r.len) + int(r.lb) - 1) // int(r.lb)
                with open(FASTA[self.build], "rb") as fh:
                    fh.seek(int(r.off))
                    raw = fh.read(nlines * int(r.lbytes))
                s = np.frombuffer(raw.replace(b"\n", b"").replace(b"\r", b""), np.uint8)[:int(r.len)]
                np.save(f, np.where((s >= 97) & (s <= 122), s - 32, s).astype(np.uint8))
            self.seq[c] = np.load(f, mmap_mode="r")
        return self.seq[c]

    def cpg_strand(self, chrom, pos1):
        """+1 where pos1 (1-based) is the C of a CpG on the + strand, -1 where it is the C on the - strand (the G of a
        CpG), 0 otherwise."""
        out = np.zeros(len(pos1), np.int8)
        chrom = np.asarray(chrom)                                   # chromosome codes (CHR2I)
        for c in np.unique(chrom):
            if c <= 0 or CHROMS[c - 1] not in self.fai.index:
                continue
            i = np.flatnonzero(chrom == c)
            s = self.chrom(CHROMS[c - 1])
            p = pos1[i].astype(np.int64) - 1                      # 0-based index of the reported base
            ok = (p >= 1) & (p < len(s) - 1)
            pp = np.where(ok, p, 1)
            b, nx, pv = s[pp], s[pp + 1], s[pp - 1]
            plus = ok & (b == 67) & (nx == 71)
            minus = ok & (b == 71) & (pv == 67)
            out[i] = np.where(plus, 1, np.where(minus, -1, 0))
        return out


def vocab_coords(build):
    """Vocabulary order (49,156 IDs): chromosome, CpG_beg (0-based start of the CG), key of the + strand C
    (chromosome code * 1e9 + CpG_beg + 1) for CpG targets; as build_vocab_map of 03_build_matrix.py."""
    vocab = np.load(need(ART / "platform_vocab_masks_2026-10-01.npz"))["vocab"].astype(str)
    man = pd.read_csv(need(MANIFEST_DIR / f"EPICv2.{build}.manifest.tsv.gz"), sep="\t",
                      usecols=["CpG_chrm", "CpG_beg", "CpG_end", "Probe_ID"], dtype={"CpG_chrm": str})
    man["cpg"] = man.Probe_ID.str.split("_").str[0]
    man = man[man.cpg.isin(set(vocab))].drop_duplicates("cpg", keep="first")
    vm = pd.DataFrame({"cpg": vocab}).merge(man[["cpg", "CpG_chrm", "CpG_beg", "CpG_end"]], on="cpg", how="left")
    vm["cpg_target"] = vm.CpG_chrm.notna() & ((vm.CpG_end - vm.CpG_beg) == 2) & vm.CpG_chrm.isin(list(CHR2I))
    code = vm.CpG_chrm.map(CHR2I)
    vm["key"] = np.where(vm.cpg_target, code.fillna(0).astype(np.int64) * SCALE + vm.CpG_beg.fillna(0).astype(np.int64) + 1, -1)
    if pd.Index(vm.key[vm.cpg_target]).has_duplicates:
        sys.exit(f"{build}: two vocabulary IDs share one CpG")
    return vm


def check_vocab_coords():
    """hg38 against the existing pipeline's cpg_mapping.csv; hg19 against the hg19 reference."""
    out = {}
    v38 = vocab_coords("hg38")
    old = pd.read_csv(need(EXT / "GSE194154/cpg_mapping.csv"))
    m = v38.merge(old[["cpg", "cpg_target", "pos_C_plus_1based", "CpG_chrm"]], on="cpg", suffixes=("", "_old"))
    t = m.cpg_target
    out["hg38_targets"] = int(t.sum())
    out["hg38_targets_pipeline"] = int(m.cpg_target_old.sum())
    out["hg38_same_target_set_as_pipeline"] = bool((m.cpg_target == m.cpg_target_old).all())
    d = m[t]
    diff = d[((d.CpG_beg + 1) != d.pos_C_plus_1based) | (d.CpG_chrm != d.CpG_chrm_old)]
    out["hg38_targets_with_other_coordinates_than_pipeline"] = int(len(diff))
    out["hg38_targets_with_other_coordinates_ids"] = ";".join(diff.cpg)
    for b, v in (("hg38", v38), ("hg19", vocab_coords("hg19"))):
        g = Genome(b)
        vt = v[v.cpg_target]
        st = g.cpg_strand(vt.CpG_chrm.map(CHR2I).to_numpy().astype(np.int16), (vt.CpG_beg + 1).to_numpy().astype(np.int64))
        out[f"{b}_targets"] = int(len(vt))
        out[f"{b}_fraction_plus_C_on_reference_CpG"] = float((st == 1).mean())
    write_json(OUTI / f"vocab_coordinates_check_{DATE}.json", out)
    log(json.dumps(out))
    return out


# ------------------------------------------------------------------ call aggregation
M_TABLE = None


def methylkit_counts(cov, freq):
    """03_build_matrix.py's recovery of methylated reads from (coverage, truncated integer freqC)."""
    global M_TABLE
    if M_TABLE is None:
        tab = np.full((101, 101), np.nan)
        for c in range(1, 101):
            m = np.arange(c + 1)
            f = (m / c * 100).astype(np.int64)
            s_ = np.bincount(f, weights=m, minlength=101)
            k = np.bincount(f, minlength=101)
            tab[c, k > 0] = s_[k > 0] / k[k > 0]
        M_TABLE = tab
    cov, f = cov.astype(np.int64), np.rint(freq).astype(np.int64)
    small = cov <= 100
    m_tab = np.full(len(cov), np.nan)
    m_tab[small] = M_TABLE[cov[small], f[small]]
    m_lo = (f * cov + 99) // 100
    m_hi = np.where(f >= 100, cov, ((f + 1) * cov + 99) // 100 - 1)
    m_mid = np.where(m_lo <= m_hi, (m_lo + m_hi) / 2.0, np.nan)
    m = np.where(small, m_tab, m_mid)
    return np.where(np.isnan(m), f * cov / 100.0, m)


def aggregate_calls(genome, chrom, pos1, meth, total):
    """Key each call to the 1-based + strand C of its CpG (reference check), sum strands and duplicate rows.
    Returns (keys, methylated, total) of CpGs, sorted by key; non-CpG rows dropped."""
    st = genome.cpg_strand(chrom, pos1)                         # chrom: chromosome codes
    ok = (st != 0) & (chrom > 0) & (total > 0)
    key = chrom[ok].astype(np.int64) * SCALE + pos1[ok].astype(np.int64) - (st[ok] == -1)
    m, t = meth[ok].astype(np.float64), total[ok].astype(np.float64)
    o = np.argsort(key, kind="stable")
    key, m, t = key[o], m[o], t[o]
    if not len(key):
        return key, m, t
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return key[start], np.add.reduceat(m, start), np.add.reduceat(t, start)


def to_vocab(vk_sorted, vk_col, keys, meth, total, n_vocab):
    """Vocabulary vectors (beta with NaN below MIN_COV, total reads) of one sample."""
    beta = np.full(n_vocab, np.nan, np.float32)
    cov = np.zeros(n_vocab, np.int32)
    p = np.searchsorted(keys, vk_sorted)
    pc = np.minimum(p, max(len(keys) - 1, 0))
    found = (p < len(keys)) & (keys[pc] == vk_sorted) if len(keys) else np.zeros(len(vk_sorted), bool)
    col = vk_col[found]
    cov[col] = np.minimum(total[pc[found]], np.iinfo(np.int32).max).astype(np.int32)
    obs = total[pc[found]] >= MIN_COV
    beta[col[obs]] = (meth[pc[found]][obs] / total[pc[found]][obs]).astype(np.float32)
    return beta, cov


# ------------------------------------------------------------------ stage download
def https_url(u):
    return re.sub(r"^ftp://ftp\.ncbi\.nlm\.nih\.gov", "https://ftp.ncbi.nlm.nih.gov", u)


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    r = subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "5", "-m", "7200", "-o", str(tmp), url],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"download failed: {url}: {r.stderr[-300:]}")
    tmp.rename(dest)
    return dest


def frozen_table():
    fr = OUTI / f"datasets_frozen_I_{DATE}.csv"
    if not git_committed(fr):
        sys.exit("Cannot proceed - the frozen dataset list must be committed before any download or mapping")
    return pd.read_csv(fr)


def contrast_samples(acc):
    st = pd.read_csv(need(OUTI / f"samples_{acc}_{DATE}.csv"))
    return st[st.in_contrast].sort_values("order").reset_index(drop=True)


def stage_download(only=None):
    """Sample-level files (rule 'files_pattern' on the GSM supplementary file names) of the contrast samples, or
    series-level files ('series_files_pattern'); into artifacts/external/<GSE>/raw_I/ (one new directory per
    dataset). gzip files are tested."""
    fr = frozen_table()
    status = []
    for d in fr.itertuples():
        if only and d.accession not in only:
            continue
        rule = json.loads(d.label_rule)
        raw = EXT / d.accession / "raw_I"
        st = contrast_samples(d.accession)
        recs = dict(sample_records(d.accession))
        urls = []
        if rule.get("series_files_pattern"):
            srec = series_record(d.accession)[0][1]
            urls = [u for u in srec.get("Series_supplementary_file", []) if re.search(rule["series_files_pattern"], u)]
        else:
            for g in st.gsm:
                fs = [u for k, v in recs[g].items() if k.startswith("Sample_supplementary_file") for u in v
                      if u and u != "NONE" and re.search(rule["files_pattern"], u.rsplit("/", 1)[-1])]
                urls += fs
        def get(u):
            dest = raw / u.rsplit("/", 1)[-1]
            try:
                fresh = not dest.exists()
                download(https_url(u), dest)
                if fresh and dest.suffix == ".gz" and subprocess.run(["gzip", "-t", str(dest)], capture_output=True).returncode:
                    dest.unlink()
                    raise RuntimeError(f"gzip test failed: {dest.name}")
                return None
            except Exception as e:
                return str(e)[:200]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(int(__import__("os").environ.get("I_DL_THREADS", "6"))) as ex:   # execution only
            outs = list(ex.map(get, urls))
        err = [o for o in outs if o]
        ok = len(urls) - len(err)
        status.append({"accession": d.accession, "files_expected": len(urls), "files_downloaded": ok,
                       "errors": " | ".join(err[:3]), "bytes": int(sum(p.stat().st_size for p in raw.glob("*") if p.is_file()))
                       if raw.exists() else 0, "date": time.strftime("%Y-%m-%d %H:%M")})
        log(f"{d.accession}: {ok}/{len(urls)} files")
    prev = WORKI / "download_status.csv"
    df = pd.DataFrame(status)
    if prev.exists():
        old = pd.read_csv(prev)
        df = pd.concat([old[~old.accession.isin(df.accession)], df], ignore_index=True)
    df.to_csv(prev, index=False)


# ------------------------------------------------------------------ call-file reading (format spec per dataset)
def read_calls(path, spec):
    """(chrom, position as deposited, methylated reads, total reads) of one call file under a format spec:
    sep, header (True/False), comment, skiprows; columns chr, pos and two of meth / unmeth / total / percent
    (percent_scale 100 or 1; methylkit = True for truncated integer percentages); optional context column and the
    values that mark CpG context."""
    c = spec["cols"]
    use = [v for v in c.values() if v is not None]
    dt = {c["chr"]: "category"}
    if c.get("context") is not None:
        dt[c["context"]] = "category"
    df = pd.read_csv(path, sep=spec.get("sep", "\t"), header=0 if spec.get("header") else None, usecols=use,
                     comment=spec.get("comment"), skiprows=spec.get("skiprows", 0), dtype=dt, low_memory=False)
    if c.get("context") is not None:
        df = df[df[c["context"]].astype(str).isin(spec["context_cpg"])]
    cat = df[c["chr"]].astype("category").cat
    chrom = chrom_code(np.asarray(cat.categories).astype(str))[cat.codes.to_numpy()] if len(df) else np.zeros(0, np.int16)
    pos = pd.to_numeric(df[c["pos"]], errors="coerce").to_numpy()
    if c.get("meth") is not None and c.get("unmeth") is not None:
        m = pd.to_numeric(df[c["meth"]], errors="coerce").to_numpy(float)
        t = m + pd.to_numeric(df[c["unmeth"]], errors="coerce").to_numpy(float)
    elif c.get("meth") is not None and c.get("total") is not None:
        m = pd.to_numeric(df[c["meth"]], errors="coerce").to_numpy(float)
        t = pd.to_numeric(df[c["total"]], errors="coerce").to_numpy(float)
    else:
        t = pd.to_numeric(df[c["total"]], errors="coerce").to_numpy(float)
        pc = pd.to_numeric(df[c["percent"]], errors="coerce").to_numpy(float)
        if spec.get("methylkit"):
            m = methylkit_counts(t, pc)
        else:
            m = pc / float(spec.get("percent_scale", 100)) * t
    ok = np.isfinite(pos) & np.isfinite(m) & np.isfinite(t) & (t > 0) & (chrom > 0)
    return chrom[ok], pos[ok].astype(np.int64), m[ok], t[ok]


_KZ = {}


def kz_matrix_row(path, name):
    """GSE93203-style series matrix (rows = sample names; per CpG marker 'X<chr>.<pos>' three columns beta, T, C):
    (chrom, position, methylated = C, total = C + T) of one sample."""
    if path not in _KZ:
        df = pd.read_csv(path, sep="\t", header=None, skiprows=2, index_col=0, low_memory=False)
        markers = pd.read_csv(path, sep="\t", header=None, nrows=1).iloc[0, 1:].astype(str).to_numpy()
        kind = pd.read_csv(path, sep="\t", header=None, skiprows=1, nrows=1).iloc[0, 1:].astype(str).to_numpy()
        _KZ[path] = (df, markers, kind)
    df, markers, kind = _KZ[path]
    if name not in df.index:
        return None
    row = pd.to_numeric(df.loc[name], errors="coerce").to_numpy(float)
    t_i, c_i = np.flatnonzero(kind == "T"), np.flatnonzero(kind == "C")
    mk = markers[c_i]
    if not (markers[t_i] == mk).all():
        raise ValueError("matrix T and C columns are not aligned")
    chrom = chrom_code(np.array(["chr" + m[1:].split(".")[0] for m in mk]))
    pos = np.array([int(m.split(".")[1]) for m in mk], np.int64)
    C, T = row[c_i], row[t_i]
    ok = np.isfinite(C) & np.isfinite(T) & ((C + T) > 0)
    return chrom[ok], pos[ok], C[ok], (C + T)[ok]


def sample_files(acc, gsm, spec):
    raw = EXT / acc / "raw_I"
    fs = sorted(p for p in raw.glob(f"{gsm}[_.]*") if not p.name.endswith(".part"))
    if spec.get("file_regex"):
        fs = [p for p in fs if re.search(spec["file_regex"], p.name)]
    return fs


def parse_sample(args):
    """Worker: one sample -> vocabulary beta and coverage, genome-wide keys and betas (>= MIN_COV), counts."""
    acc, gsm, spec, build, base, vk_sorted, vk_col, n_vocab = args[:8]
    g = Genome(build)
    ch, pos, m, t = [], [], [], []
    if spec.get("kz_matrix"):
        a = kz_matrix_row(str(sorted((EXT / acc / "raw_I").glob("*processed_data.txt.gz"))[0]), args[8])
        if a is not None:
            ch.append(a[0]); pos.append(a[1]); m.append(a[2]); t.append(a[3])
    for f in ([] if spec.get("kz_matrix") else sample_files(acc, gsm, spec)):
        a = read_calls(f, spec)
        ch.append(a[0]); pos.append(a[1]); m.append(a[2]); t.append(a[3])
    if not ch:
        return gsm, None
    ch, pos, m, t = np.concatenate(ch), np.concatenate(pos), np.concatenate(m), np.concatenate(t)
    pos1 = pos + (1 if base == 0 else 0)
    keys, mm, tt = aggregate_calls(g, ch, pos1, m, t)
    beta, cov = to_vocab(vk_sorted, vk_col, keys, mm, tt, n_vocab)
    ok = tt >= MIN_COV
    info = {"rows": int(len(pos)), "cpg_sites": int(len(keys)), "cpg_sites_cov10": int(ok.sum()),
            "median_depth_cov10": float(np.median(tt[ok])) if ok.any() else np.nan,
            "vocab_cpgs_any_read": int((cov > 0).sum()), "vocab_cpgs_cov10": int(np.isfinite(beta).sum())}
    return gsm, (beta, cov, keys[ok], (mm[ok] / tt[ok]).astype(np.float32), info)


def convention_scan(acc, st, spec, stated):
    """Fraction of call rows on a reference CpG cytosine under each build and position base (up to 5 samples,
    20,000 rows); vocabulary CpGs matched (>= 1 read) under each choice."""
    rng = np.random.default_rng(SEED)
    rows = []
    sample = []
    for gsm, title in zip(st.gsm[:5], st.title[:5]):
        files = [None] if spec.get("kz_matrix") else sample_files(acc, gsm, spec)
        for f in files:
            if spec.get("kz_matrix"):
                a = kz_matrix_row(str(sorted((EXT / acc / "raw_I").glob("*processed_data.txt.gz"))[0]), title)
            else:
                a = read_calls(f, spec)
            k = rng.choice(len(a[1]), size=min(4000, len(a[1])), replace=False)
            sample.append((a[0][k], a[1][k], a[2][k], a[3][k]))
            break
    ch = np.concatenate([s[0] for s in sample])                 # chromosome codes
    pos = np.concatenate([s[1] for s in sample])
    for build in ([stated] + [b for b in ("hg38", "hg19") if b != stated]):
        g = Genome(build)
        vm = vocab_coords(build)
        vk = set(vm.key[vm.cpg_target].astype(np.int64))
        for base in (1, 0):
            p1 = pos + (1 if base == 0 else 0)
            s_ = g.cpg_strand(ch, p1)
            key = ch.astype(np.int64) * SCALE + p1 - (s_ == -1)
            rows.append({"accession": acc, "build": build, "position_base": base, "rows_checked": int(len(pos)),
                         "fraction_on_reference_cpg": float((s_ != 0).mean()),
                         "fraction_plus_strand": float((s_ == 1).mean()), "fraction_minus_strand": float((s_ == -1).mean()),
                         "vocab_cpgs_matched_in_checked_rows": int(len(set(key[s_ != 0].tolist()) & vk))})
    df = pd.DataFrame(rows)
    order = {stated: 0}
    best = None
    for b in [stated] + [x for x in ("hg38", "hg19") if x != stated]:
        sub = df[df.build == b].sort_values("fraction_on_reference_cpg", ascending=False, kind="stable").iloc[0]   # ties: 1-based
        if sub.fraction_on_reference_cpg >= 0.90:
            best = (b, int(sub.position_base))
            break
    df["chosen"] = [best is not None and (r.build, r.position_base) == best for r in df.itertuples()]
    return df, best


LOGDIR = ART / "logs"
RUN_DATE = "2026-10-27"


def atomic_save(path, arr=None, **arrays):
    """np.save / np.savez to a temporary name, then rename (a killed process leaves no partial file)."""
    import os
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    if arrays:
        np.savez(tmp, **arrays)
    else:
        np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_text(path, text):
    import os
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def done_marker(stage, info):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    atomic_text(LOGDIR / f"large_gain_I_{stage}_{RUN_DATE}.done",
                json.dumps({"stage": stage, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **info}, indent=1, default=str))


def npy_ok(path, rows, cols=None):
    """True when the .npy file loads (memory-mapped) with the expected shape."""
    try:
        a = np.load(path, mmap_mode="r")
        return a.shape[0] == rows and (cols is None or a.shape[1] == cols)
    except Exception:
        return False


def map_record_ok(acc):
    """A dataset's mapping is done when its record exists and, if included, its matrices load with the recorded shape."""
    f = WORKI / "map" / f"{acc}.json"
    if not f.exists():
        return False
    try:
        rec = json.loads(f.read_text())
    except Exception:
        return False
    row = rec["row"]
    if not row.get("included"):
        return True
    d = WORKI / acc
    n = int(row["n_case"]) + int(row["n_ctrl"])
    try:
        st = pd.read_csv(d / "analysis_samples.csv")
        g = np.load(d / "genome.npz")
        gk = g["keys"]
        ok_g = g["beta"].shape == (n, len(gk)) and len(gk) == int(row["genome_cpgs_cov10_in_ge18_samples"])
    except Exception:
        return False
    return len(st) == n and npy_ok(d / "X.npy", n, 49156) and npy_ok(d / "coverage.npy", n, 49156) and ok_g


def map_one(d, specs, dl, workers):
    """Mapping of one frozen dataset; returns its record (row, convention scan, per-sample coverage)."""
    from multiprocessing import get_context
    acc = d.accession
    rule = json.loads(d.label_rule)
    st = contrast_samples(acc)
    base_row = {"accession": acc, "assay": d.assay, "disease": d.disease, "tissue": d.tissue,
                "n_case_frozen": int(d.n_case), "n_ctrl_frozen": int(d.n_ctrl)}
    rec = {"row": None, "scan": [], "coverage": []}
    s_dl = dl[dl.accession == acc]
    if acc not in specs or not len(s_dl) or int(s_dl.files_downloaded.iloc[0]) < int(s_dl.files_expected.iloc[0]):
        rec["row"] = {**base_row, "included": False, "reason": "T1: files could not be downloaded or no format spec"}
        return rec
    spec = specs[acc]
    try:
        scan, best = convention_scan(acc, st, spec, rule.get("genome_build", "hg38"))
    except Exception as e:
        rec["row"] = {**base_row, "included": False, "reason": f"T1: files could not be parsed ({str(e)[:150]})"}
        return rec
    rec["scan"] = scan.to_dict("records")
    if best is None:
        rec["row"] = {**base_row, "included": False, "reason": "T2: coordinates do not match hg19 or hg38"}
        return rec
    build, base = best
    vm = vocab_coords(build)
    tgt = np.flatnonzero(vm.cpg_target.to_numpy())
    keys = vm.key.to_numpy(np.int64)[tgt]
    o = np.argsort(keys)
    vk_sorted, vk_col = keys[o], tgt[o]
    n_vocab = len(vm)
    for c in CHROMS:                                        # cache the reference before forking
        if c in Genome(build).fai.index:
            Genome(build).chrom(c)
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        res = dict(pool.map(parse_sample, [(acc, g, spec, build, base, vk_sorted, vk_col, n_vocab, ttl)
                                           for g, ttl in zip(st.gsm, st.title)], chunksize=1))
    log(f"{acc}: parsed {len(res)} samples ({time.time() - t0:.0f} s)")
    ok_s = [g for g in st.gsm if res.get(g) is not None]
    info = {g: res[g][4] for g in ok_s}
    qc = {g: info[g]["vocab_cpgs_cov10"] >= MIN_SAMPLE_VOCAB for g in ok_s}
    for g in st.gsm:
        r0 = st[st.gsm == g].iloc[0]
        rec["coverage"].append({"accession": acc, "gsm": g, "role": r0.role, "parsed": g in info,
                                **(info.get(g) or {}), "sample_qc_pass": bool(qc.get(g, False))})
    keep = st[st.gsm.isin([g for g in ok_s if qc[g]])].reset_index(drop=True)
    nca, nco = int((keep.role == "case").sum()), int((keep.role == "control").sum())
    v = [info[g]["vocab_cpgs_cov10"] for g in keep.gsm]
    row = {**base_row, "build": build, "position_base": base, "n_case": nca, "n_ctrl": nco,
           "samples_parsed": len(ok_s), "samples_failing_qc": int(sum(not x for x in qc.values())),
           "median_vocab_cpgs_cov10": float(np.median(v)) if v else 0.0,
           "q25_vocab_cpgs_cov10": float(np.percentile(v, 25)) if v else 0.0,
           "q75_vocab_cpgs_cov10": float(np.percentile(v, 75)) if v else 0.0,
           "min_vocab_cpgs_cov10": int(min(v)) if v else 0, "max_vocab_cpgs_cov10": int(max(v)) if v else 0}
    if nca < MIN_CASE or nco < MIN_CTRL:
        rec["row"] = {**row, "included": False, "reason": "T3: fewer than 30 cases or 30 controls with calls after sample QC"}
        return rec
    if row["median_vocab_cpgs_cov10"] < MIN_MEDIAN_VOCAB:
        rec["row"] = {**row, "included": False,
                      "reason": f"T4: median sample has fewer than {MIN_MEDIAN_VOCAB} vocabulary CpGs at >= {MIN_COV} reads"}
        return rec
    dd = WORKI / acc
    dd.mkdir(parents=True, exist_ok=True)
    X = np.stack([res[g][0] for g in keep.gsm])
    atomic_save(dd / "X.npy", X)
    atomic_save(dd / "coverage.npy", np.stack([res[g][1] for g in keep.gsm]))
    allk, cnt = np.unique(np.concatenate([res[g][2] for g in keep.gsm]), return_counts=True)
    gk = allk[cnt >= int(np.ceil(COV90 * 20))]
    del allk, cnt
    Gm = np.full((len(keep), len(gk)), np.nan, np.float32)
    for i, g in enumerate(keep.gsm):
        k, b = res[g][2], res[g][3]
        j = np.searchsorted(gk, k)
        hit = (j < len(gk)) & (gk[np.minimum(j, len(gk) - 1)] == k)
        Gm[i, j[hit]] = b[hit]
    atomic_save(dd / "genome.npz", beta=Gm, keys=gk)
    atomic_text(dd / "analysis_samples.csv", keep.to_csv(index=False))
    sizes = [n for n in (20, 50, 100) if nca >= n // 2 + 10 and nco >= n // 2 + 10]
    arr = pd.read_csv(need(EXT / "corpus_whole_blood_mean_beta.csv")).set_index("cpg").blood_array_mean_beta
    vocab = vm.cpg.to_numpy()
    half = np.isfinite(X).mean(0) >= 0.5
    mb = np.nanmean(X[:, half], 0) if half.any() else np.zeros(0)
    am = arr.reindex(vocab[half]).to_numpy()
    okm = np.isfinite(am)
    rec["row"] = {**row, "included": True, "reason": "", "sizes_evaluated": ";".join(map(str, sizes)),
                  "sizes_skipped": ";".join(str(n) for n in (20, 50, 100) if n not in sizes),
                  "vocab_cpgs_observed_in_ge50pct": int(half.sum()),
                  "vocab_cpgs_observed_in_any": int(np.isfinite(X).any(0).sum()),
                  "genome_cpgs_cov10_in_ge18_samples": int(len(gk)),
                  "pearson_r_mean_beta_vs_corpus_blood_array_mean": float(np.corrcoef(mb[okm], am[okm])[0, 1]) if okm.sum() > 10 else None,
                  "samples_sha256": hashlib.sha256("\n".join(sorted(keep.gsm)).encode()).hexdigest()}
    log(f"{acc}: {nca} cases, {nco} controls; median vocabulary CpGs {row['median_vocab_cpgs_cov10']:.0f}; genome-wide CpGs {len(gk)}")
    return rec


def assemble_map_tables(fr):
    recs = {a: json.loads((WORKI / "map" / f"{a}.json").read_text()) for a in fr.accession if map_record_ok(a)}
    pd.DataFrame([r["row"] for r in recs.values()]).sort_values("accession").to_csv(OUTI / f"datasets_analysis_{DATE}.csv", index=False)
    pd.DataFrame([x for r in recs.values() for x in r["scan"]]).to_csv(OUTI / f"mapping_check_{DATE}.csv", index=False)
    pd.DataFrame([x for r in recs.values() for x in r["coverage"]]).to_csv(OUTI / f"coverage_per_sample_{DATE}.csv", index=False)
    return recs


def stage_map(only=None, workers=16):
    """Resumable per dataset: a dataset whose record passes map_record_ok is skipped. The analysis, mapping-check and
    coverage tables are assembled from the records; the done-marker is written when every frozen dataset has one."""
    fr = frozen_table()
    specs = json.loads(need(OUTI / f"format_specs_{DATE}.json").read_text())
    dl = pd.read_csv(need(WORKI / "download_status.csv"))
    (WORKI / "map").mkdir(parents=True, exist_ok=True)
    for d in fr.itertuples():
        if (only and d.accession not in only) or map_record_ok(d.accession):
            continue
        rec = map_one(d, specs, dl, workers)
        atomic_text(WORKI / "map" / f"{d.accession}.json", json.dumps(rec, default=float))
        log(f"{d.accession}: {rec['row']['reason'] or 'included'}")
    recs = assemble_map_tables(fr)
    if len(recs) == len(fr):
        done_marker("map", {"datasets": len(recs), "included": int(sum(r["row"]["included"] for r in recs.values()))})


# ------------------------------------------------------------------ shared loading for later stages
def analysis_table():
    return pd.read_csv(need(OUTI / f"datasets_analysis_{DATE}.csv"))


def included_datasets():
    t = analysis_table()
    return t[t.included].sort_values("accession")


def dataset_arrays(acc):
    """(samples table, y, X) of a dataset's analysis samples (vocabulary matrix, NaN = unobserved)."""
    d = WORKI / acc
    st = pd.read_csv(need(d / "analysis_samples.csv"))
    X = np.load(need(d / "X.npy")).astype(np.float32)
    if len(st) != len(X):
        sys.exit(f"{acc}: sample table and matrix differ in length")
    return st, (st.role == "case").to_numpy(np.float32), X


def pick_device(device=None):
    import torch
    if device:
        return torch.device(device)
    free = [torch.cuda.mem_get_info(i)[0] for i in range(min(2, torch.cuda.device_count()))]
    i = int(np.argmax(free))
    log(f"GPU free memory {[round(f / 2**30, 1) for f in free]} GiB; using cuda:{i}")
    return torch.device(f"cuda:{i}")


# ------------------------------------------------------------------ stage represent
def stage_represent(device):
    import torch
    G = load_g()
    F = G.load_f()
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    if not git_committed(OUTI / f"datasets_analysis_{DATE}.csv"):
        sys.exit("Cannot proceed - the analysis table must be committed before representations are computed")
    dev = pick_device(device)
    ds = included_datasets()
    mu = np.load(need(F.CORE))["mu"].astype(np.float32)
    clock_rows = G.clock_cpg_rows()
    sets = {a: dataset_arrays(a) for a in ds.accession}
    (WORKI / "clocks").mkdir(parents=True, exist_ok=True)
    # embedding (TF32 as A's embed stage) and clock inputs (mean-filled profile)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, core, _, _ = F.load_encoder("age_stage", dev)
    for a, (st, y, X) in sets.items():
        fe = WORKI / a / "embedding.npy"
        if not npy_ok(fe, len(X), 2048):
            atomic_save(fe, F.embed_profiles(model, core, X, dev)[0])
        fc = WORKI / "clocks" / f"{a}_clockin.parquet"
        try:
            ok_c = list(pd.read_parquet(fc).columns) == list(st.gsm.astype(str))
        except Exception:
            ok_c = False
        if not ok_c:
            Z = np.where(np.isfinite(X), X, mu[None, :])[:, clock_rows[0]]
            tmp = fc.with_name(fc.stem + ".tmp.parquet")
            pd.DataFrame(Z.T.astype(np.float64), index=clock_rows[1], columns=st.gsm.astype(str)).to_parquet(tmp)
            __import__("os").replace(tmp, fc)
        log(f"{a}: embedding and clock input written")
    del model
    torch.cuda.empty_cache()
    # MethylGPT imputation (release bundle API, factor method, FP32 as G)
    torch.backends.cuda.matmul.allow_tf32 = False
    want = dict(reversed(l.split()) for l in need(G.CHECKSUMS).read_text().splitlines() if l.strip())
    bad = sorted(fn for fn in want if sha256_file(G.BUNDLE / fn) != want[fn])
    if bad:
        sys.exit(f"release bundle checksum mismatch: {bad}")
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    mg = MethylGPT.from_pretrained(str(G.BUNDLE), device=str(dev))
    vocab = list(np.load(need(F.MASKS))["vocab"].astype(str))
    if mg.vocab != vocab:
        sys.exit("bundle vocabulary differs from the platform-mask vocabulary")
    for a, (st, y, X) in sets.items():
        fi = WORKI / a / "imputed.npy"
        if npy_ok(fi, len(X), 49156):
            continue
        imp = mg.impute(pd.DataFrame(X, columns=vocab), batch_size=8).to_numpy(np.float32)
        obs = np.isfinite(X)
        if not np.array_equal(imp[obs], np.clip(X[obs], 0, 1)):
            sys.exit(f"{a}: impute() changed observed values")
        atomic_save(fi, imp)
        log(f"{a}: MethylGPT imputation written")
    del mg
    torch.cuda.empty_cache()
    # nearest neighbours: G's held-out reference and implementation
    spec = importlib.util.spec_from_file_location("recon_diag", HERE / "si_reconstruction_diagnostics_2026-10-18.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    todo = [a for a in sets if not npy_ok(WORKI / a / "knn.npy", len(sets[a][2]), 49156)]
    if todo:
        free, total = torch.cuda.mem_get_info(dev)
        log(f"GPU free {free / 2**30:.1f} of {total / 2**30:.1f} GiB before loading the kNN reference")
        R, rrows = G.knn_reference(F, "heldout", dev)
        sha = hashlib.sha256(rrows.astype(np.int64).tobytes()).hexdigest()
        gref = json.loads(need(G.OUTG / f"knn_reference_heldout_{G.DATE}.json").read_text())
        prof = pd.read_parquet(need(CORPUS_DIR / "profiles.parquet"), columns=["id"])
        ours = set(g for a in sets for g in sets[a][0].gsm.astype(str))
        drop = np.isin(prof.id.astype(str).to_numpy()[rrows], list(ours))
        if drop.any():
            keep_t = torch.from_numpy(np.flatnonzero(~drop)).to(dev)
            R = R[keep_t]
        write_json(OUTI / f"knn_reference_{DATE}.json", {"rows": int(len(rrows)), "rows_sha256": sha,
                   "identical_to_G_heldout_reference": sha == gref["rows_sha256"] and len(rrows) == gref["rows"],
                   "rows_removed_as_profiles_of_these_datasets": int(drop.sum()), "rows_used": int(len(R))})
        mu_h = torch.as_tensor(mu, device=dev).half()
        for a in todo:
            st, y, X = sets[a]
            obs = np.isfinite(X)
            Xt = torch.from_numpy(np.where(obs, X, np.nan).astype(np.float32)).to(dev)
            kt = torch.from_numpy(obs).to(dev)
            pr = rd.knn_predict_lowmem(R, mu_h, Xt, kt, ~kt, k=G.KNN_K, batch=256, chunk=8192)
            out = X.copy()
            for i, p in enumerate(pr):
                out[i, ~obs[i]] = p
            if not np.isfinite(out).all():
                sys.exit(f"{a}: kNN output has missing values")
            atomic_save(WORKI / a / "knn.npy", out)
            del Xt, kt
            log(f"{a}: kNN done")
        del R
        torch.cuda.empty_cache()
    done_marker("represent", {"datasets": len(sets)})


def stage_clocks():
    """G's stage clocks (biolearn; A's GrimAge2 convention) on work_I/clocks/*_clockin.parquet."""
    G = load_g()
    G.WORKG = WORKI
    for fo in sorted((WORKI / "clocks").glob("*_clocks.csv")):      # resumable: drop partial outputs
        try:
            ok = list(pd.read_csv(fo, index_col=0).index.astype(str)) == \
                list(pd.read_parquet(fo.with_name(fo.name.replace("_clocks.csv", "_clockin.parquet"))).columns)
        except Exception:
            ok = False
        if not ok:
            fo.unlink()
    G.stage_clocks("clocks")
    done_marker("clocks", {"files": len(list((WORKI / "clocks").glob("*_clocks.csv")))})


# ------------------------------------------------------------------ stage fit
def features(a, rep, X, mu):
    d = WORKI / a
    if rep == "embedding":
        return np.load(need(d / "embedding.npy"))
    if rep == "latent":
        return np.load(need(d / "embedding.npy"))[:, :1536]
    if rep == "imputed":
        return np.load(need(d / "imputed.npy"))
    if rep == "raw_knn":
        return np.load(need(d / "knn.npy"))
    if rep == "raw_mean":
        return np.where(np.isfinite(X), X, mu[None, :]).astype(np.float32)
    if rep == "clocks":
        st = pd.read_csv(need(d / "analysis_samples.csv"))
        c = pd.read_csv(need(WORKI / "clocks" / f"{a}_clocks.csv"), index_col=0)
        c.index = c.index.astype(str)
        return c.reindex(st.gsm.astype(str))[["GrimAge2", "DunedinPACE", "PhenoAge", "Horvathv1"]].to_numpy(np.float32)
    if rep == "raw_cov90":
        return X
    if rep == "raw_cov90_genome":
        return np.load(need(d / "genome.npz"))["beta"].astype(np.float32)
    raise ValueError(rep)


def fit_one(A, Z, y, trs, dev, rep):
    """A.fit_select_score for one dataset and size; draw-specific columns for the coverage-restricted comparators."""
    if rep not in ("raw_cov90", "raw_cov90_genome"):
        res = A.fit_select_score(np.asarray(Z, np.float32), y, trs, dev, fold_seeds=list(range(A.N_DRAWS)))
        return res, [int(Z.shape[1])] * len(trs)
    obs = np.isfinite(Z)
    res, nf = [], []
    for r, tr in enumerate(trs):
        need_n = int(np.ceil(COV90 * len(tr)))
        cols = np.flatnonzero(obs[tr].sum(0) >= need_n)
        nf.append(int(len(cols)))
        if not len(cols):
            mask = np.ones(len(y), bool)
            mask[tr] = False
            res.append({"score": np.zeros(int(mask.sum())), "alpha": np.nan, "lambda_index": -1, "nonzero": 0,
                        "cv_deviance": np.nan, "no_feature": True})
            continue
        res.append(A.fit_select_score(np.ascontiguousarray(Z[:, cols]), y, [tr], dev, fold_seeds=[r])[0])
    return res, nf


REPS_FIT = MGPT_REPS + COMP_REPS + SECONDARY_COMP


def stage_fit(device, reps=None):
    import torch
    G = load_g()
    F = G.load_f()
    A = F.A
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False          # A's fit stage precision
    if not git_committed(OUTI / f"datasets_analysis_{DATE}.csv"):
        sys.exit("Cannot proceed - the analysis table must be committed before the fit")
    dev = pick_device(device)
    mu = np.load(need(F.CORE))["mu"].astype(np.float32)
    sd = WORKI / "scores"
    sd.mkdir(parents=True, exist_ok=True)
    for d in included_datasets().itertuples():
        a = d.accession
        st, y, X = dataset_arrays(a)
        sizes = [int(x) for x in str(d.sizes_evaluated).split(";") if x and x != "nan"]
        for rep in (reps or REPS_FIT):
            fo = sd / f"{a}_{rep}.npz"
            fc = sd / f"{a}_{rep}.csv"
            try:
                done = len(pd.read_csv(fc)) == len(sizes) * A.N_DRAWS and all(f"score_{n}" in np.load(fo).files for n in sizes)
            except Exception:
                done = False
            if done:
                continue
            t0 = time.time()
            Z = features(a, rep, X, mu)
            rows, store = [], {}
            for n in sizes:
                trs = A.draws_for(int(A.accession_number(a)), n, y)
                res, nf = fit_one(A, Z, y, trs, dev, rep)
                for r, (tr, rr) in enumerate(zip(trs, res)):
                    mask = np.ones(len(y), bool)
                    mask[tr] = False
                    ysc = y[mask]
                    au = 0.5 if rr.get("no_feature") else A.auc(rr["score"], ysc)
                    rows.append({"accession": a, "representation": rep, "n_train": n, "draw": r, "auc": au,
                                 "alpha": rr["alpha"], "lambda_index": rr["lambda_index"], "nonzero": rr["nonzero"],
                                 "cv_deviance": rr["cv_deviance"], "n_score_case": int(ysc.sum()),
                                 "n_score_ctrl": int((1 - ysc).sum()), "features": nf[r],
                                 "no_feature": bool(rr.get("no_feature", False))})
                store[f"train_{n}"] = np.stack(trs)
                store[f"score_{n}"] = np.stack([np.asarray(rr["score"], np.float64) for rr in res])
            atomic_save(fo, y=y, **store)
            atomic_text(fc, pd.DataFrame(rows).to_csv(index=False))
            log(f"{a} {rep}: fitted ({time.time() - t0:.0f} s); mean AUC by n "
                f"{pd.DataFrame(rows).groupby('n_train').auc.mean().round(4).to_dict()}")
            del Z
            torch.cuda.empty_cache()
    if not reps:
        done_marker("fit", {"datasets": int(len(included_datasets()))})


# ------------------------------------------------------------------ stage analyse
def weighted_auc_boot(score, ysc, W):
    """AUC of one draw under bootstrap multiplicities W (B x n_score); NaN where a class has zero weight."""
    s_c, s_n = score[ysc == 1], score[ysc == 0]
    H = (s_c[:, None] > s_n[None, :]).astype(np.float64) + 0.5 * (s_c[:, None] == s_n[None, :])
    Wc, Wn = W[:, ysc == 1], W[:, ysc == 0]
    num = ((Wc @ H) * Wn).sum(1)
    den = Wc.sum(1) * Wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def sample_boot(acc_num, y, B=2000):
    """Bootstrap multiplicities (B x N): cases and controls resampled separately with replacement."""
    rng = np.random.default_rng([SEED, acc_num])
    N = len(y)
    W = np.zeros((B, N), np.float64)
    for cls in (1, 0):
        idx = np.flatnonzero(y == cls)
        draws = rng.integers(0, len(idx), size=(B, len(idx)))
        for b in range(B):
            np.add.at(W[b], idx[draws[b]], 1)
    return W


def boot_dataset_values(acc, rep, n, W):
    """B-vector of a dataset's (mean AUC over draws) under the sample bootstrap."""
    z = np.load(need(WORKI / "scores" / f"{acc}_{rep}.npz"))
    y, trs, sc = z["y"], z[f"train_{n}"], z[f"score_{n}"]
    vals = []
    for tr, s in zip(trs, sc):
        mask = np.ones(len(y), bool)
        mask[tr] = False
        vals.append(weighted_auc_boot(s, y[mask], W[:, mask]))
    return np.nanmean(np.stack(vals, 1), 1)


def stage_analyse():
    G = load_g()
    A = G.load_f().A
    ds = included_datasets().set_index("accession")
    parts = [pd.read_csv(f) for f in sorted((WORKI / "scores").glob("*.csv"))]
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(OUTI / f"per_draw_auc_{DATE}.csv.gz", index=False)
    reps_all = MGPT_REPS + COMP_REPS + SECONDARY_COMP
    g = d.groupby(["accession", "representation", "n_train"]).auc
    per = g.mean().rename("auc_mean").to_frame().join(g.std().rename("auc_sd")).reset_index()
    nz = d.groupby(["accession", "representation", "n_train"]).agg(
        draws_no_nonzero=("nonzero", lambda v: int((v == 0).sum())), features_median=("features", "median"),
        draws_no_feature=("no_feature", "sum")).reset_index()
    per = per.merge(nz, on=["accession", "representation", "n_train"])
    sizes = sorted(per.n_train.unique())
    primary_n = 50 if (per.n_train == 50).any() else 20
    # within-dataset sample bootstrap for every dataset, representation and size
    Bv = {}
    for a in ds.index:
        y = np.load(need(WORKI / "scores" / f"{a}_{PRIMARY_REP}.npz"))["y"]
        W = sample_boot(int(A.accession_number(a)), y)
        for rep in reps_all:
            for n in sizes:
                if ((per.accession == a) & (per.representation == rep) & (per.n_train == n)).any():
                    Bv[(a, rep, n)] = boot_dataset_values(a, rep, n, W)
    ci = []
    for (a, rep, n), v in Bv.items():
        ci.append({"accession": a, "representation": rep, "n_train": n, "auc_ci95_low": float(np.nanpercentile(v, 2.5)),
                   "auc_ci95_high": float(np.nanpercentile(v, 97.5))})
    per = per.merge(pd.DataFrame(ci), on=["accession", "representation", "n_train"], how="left")
    per = per.join(ds[["disease", "tissue", "assay"]], on="accession")
    per.to_csv(OUTI / f"per_dataset_auc_{DATE}.csv", index=False)

    def sample_compare(accs, n, mg, comps, label):
        wide = per[(per.n_train == n) & per.accession.isin(accs)].pivot(index="accession", columns="representation",
                                                                        values="auc_mean") - 0.5
        reps = [r for r in reps_all if r in wide and wide[r].notna().all()]
        M = {r: float(wide[r].mean()) for r in reps}
        Bm = {r: np.mean(np.stack([Bv[(a, r, n)] for a in wide.index]), 0) - 0.5 for r in reps}
        best = max(comps, key=lambda c: M[c])
        diff = Bm[mg] - Bm[best]
        out = {"set": label, "interval": "paired bootstrap over samples within datasets (2,000 replicates)",
               "datasets": int(len(wide)), "methylgpt": mg, "M": M,
               "M_ci95": {r: [float(np.nanpercentile(Bm[r], 2.5)), float(np.nanpercentile(Bm[r], 97.5))] for r in reps},
               "best_comparator": best, "best_comparator_M": M[best], "methylgpt_M": M[mg], "difference": M[mg] - M[best],
               "difference_ci95": [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))],
               "relative_gain": (M[mg] - M[best]) / M[best] if M[best] > 0 else float("nan"), "vs_each": {}}
        for c in [r for r in reps if r != mg]:
            dd = Bm[mg] - Bm[c]
            out["vs_each"][c] = {"difference": M[mg] - M[c], "ci95": [float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))],
                                 "relative_gain": (M[mg] - M[c]) / M[c] if M[c] > 0 else float("nan"),
                                 "datasets_methylgpt_higher": int((wide[mg] > wide[c]).sum())}
        out["pass"] = bool(np.isfinite(out["relative_gain"]) and out["relative_gain"] >= 0.10 and out["difference_ci95"][0] > 0)
        return out

    rows, res = [], {"primary_n": primary_n, "primary": None, "secondary": {}}
    for n in sizes:
        accs = sorted(per[(per.n_train == n) & (per.representation == PRIMARY_REP)].accession.unique())
        K = len(accs)
        for mg in MGPT_REPS:
            for comps, cname in ((COMP_REPS, "decision comparators"), (COMP_REPS + SECONDARY_COMP, "with raw_cov90_genome")):
                kinds = ["sample"] + (["dataset"] if K >= 5 else [])
                for kind in kinds:
                    label = f"n={n}|{mg}|{cname}|{kind} bootstrap"
                    if kind == "dataset":
                        wide = per[(per.n_train == n) & per.accession.isin(accs)].pivot(
                            index="accession", columns="representation", values="auc_mean") - 0.5
                        r = A.compare(wide, reps_all, mg, comps, label)
                        r["interval"] = "paired bootstrap over datasets (A.compare, 2,000 resamples)"
                    else:
                        r = sample_compare(accs, n, mg, comps, label)
                    decision = (n == primary_n and mg == PRIMARY_REP and cname == "decision comparators" and
                                kind == ("dataset" if K >= 5 else "sample"))
                    if decision:
                        res["primary"] = r
                    else:
                        res["secondary"][label] = r
                    rows.append({"n_train": n, "methylgpt_representation": mg, "comparator_set": cname,
                                 "interval": r["interval"], "datasets": r["datasets"], "M_methylgpt": r["methylgpt_M"],
                                 "M_ci95_low": r["M_ci95"][mg][0], "M_ci95_high": r["M_ci95"][mg][1],
                                 **{f"M_{x}": r["M"].get(x) for x in reps_all}, "best_comparator": r["best_comparator"],
                                 "best_comparator_M": r["best_comparator_M"], "difference": r["difference"],
                                 "difference_ci95_low": r["difference_ci95"][0], "difference_ci95_high": r["difference_ci95"][1],
                                 "relative_gain": r["relative_gain"], "meets_rule": r["pass"], "decision_set": decision,
                                 **{f"diff_vs_{c}": v["difference"] for c, v in r["vs_each"].items()},
                                 **{f"diff_vs_{c}_ci95_low": v["ci95"][0] for c, v in r["vs_each"].items()},
                                 **{f"diff_vs_{c}_ci95_high": v["ci95"][1] for c, v in r["vs_each"].items()}})
    pd.DataFrame(rows).to_csv(OUTI / f"summary_{DATE}.csv", index=False)
    p = res["primary"]
    pd.DataFrame([{"analysis": f"I: sequencing case-control cohorts (empirical per-CpG calls, >= {MIN_COV} reads), "
                               f"{primary_n} labelled samples",
                   "primary_metric": f"mean over datasets of (AUC - 0.5), n = {primary_n}", "datasets": p["datasets"],
                   "methylgpt_representation": PRIMARY_REP, "methylgpt_value": p["methylgpt_M"],
                   "best_comparator": p["best_comparator"], "best_comparator_value": p["best_comparator_M"],
                   "relative_gain": p["relative_gain"], "difference": p["difference"],
                   "difference_ci95_low": p["difference_ci95"][0], "difference_ci95_high": p["difference_ci95"][1],
                   "interval": p["interval"],
                   "rule": "relative gain >= 10% and interval of the difference excludes zero (lower end > 0)",
                   "outcome": "PASS" if p["pass"] else "FAIL"}]).to_csv(OUTI / f"rule_outcome_{DATE}.csv", index=False)
    write_json(OUTI / f"results_{DATE}.json", res)
    done_marker("analyse", {"primary": p})
    log(f"PRIMARY (n = {primary_n}, {p['datasets']} datasets, {p['interval']}): {PRIMARY_REP} {p['methylgpt_M']:.4f} vs "
        f"{p['best_comparator']} {p['best_comparator_M']:.4f}, gain {p['relative_gain']:.3f}, CI {p['difference_ci95']}, "
        f"{'PASS' if p['pass'] else 'FAIL'}")


def stage_crosscheck():
    """GSE194154: this analysis's vocabulary matrix against the existing pipeline's betas_cov10.parquet (same samples,
    hg38, >= 10 reads, strands summed); agreement of the observed pattern and of beta values."""
    st = pd.read_csv(need(WORKI / "GSE194154" / "analysis_samples.csv"))
    X = np.load(need(WORKI / "GSE194154" / "X.npy"))
    vocab = np.load(need(ART / "platform_vocab_masks_2026-10-01.npz"))["vocab"].astype(str)
    old = pd.read_parquet(need(EXT / "GSE194154" / "betas_cov10.parquet"))
    ours = pd.DataFrame(X, index=st.gsm.astype(str), columns=vocab)
    g = [x for x in ours.index if x in old.index]
    a = ours.loc[g].reindex(columns=vocab).to_numpy()
    b = old.loc[g].reindex(columns=vocab).to_numpy()
    oa, ob = np.isfinite(a), np.isfinite(b)
    both = oa & ob
    diff_cols = np.flatnonzero((oa != ob).any(0))
    out = {"samples_compared": len(g), "entries_observed_here": int(oa.sum()), "entries_observed_pipeline": int(ob.sum()),
           "entries_observed_in_both": int(both.sum()), "entries_observed_in_one_only": int((oa != ob).sum()),
           "cpgs_with_any_disagreement_in_observed_pattern": [str(vocab[j]) for j in diff_cols],
           "max_abs_beta_difference_where_both_observed": float(np.nanmax(np.abs(a - b)[both])) if both.any() else None,
           "mean_abs_beta_difference_where_both_observed": float(np.nanmean(np.abs(a - b)[both])) if both.any() else None}
    write_json(OUTI / f"gse194154_pipeline_agreement_{DATE}.json", out)
    done_marker("crosscheck", {"samples_compared": len(g)})
    log(json.dumps({k: v for k, v in out.items() if k != "cpgs_with_any_disagreement_in_observed_pattern"}))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["search", "screen", "freeze", "download", "map", "crosscheck", "represent", "clocks", "fit", "analyse"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--reps", default=None, help="execution only: subset of representations for stage fit")
    ap.add_argument("--only", default=None, help="execution only: comma-separated accessions for download and map")
    ap.add_argument("--workers", type=int, default=16, help="execution only: parser processes for stage map")
    a = ap.parse_args()
    WORKI.mkdir(parents=True, exist_ok=True)
    if a.stage == "search":
        stage_search()
    elif a.stage == "screen":
        stage_screen()
    elif a.stage == "freeze":
        stage_freeze()
    elif a.stage == "download":
        stage_download(a.only.split(",") if a.only else None)
    elif a.stage == "map":
        stage_map(a.only.split(",") if a.only else None, workers=a.workers)
    elif a.stage == "crosscheck":
        stage_crosscheck()
    elif a.stage == "represent":
        stage_represent(a.device)
    elif a.stage == "clocks":
        stage_clocks()
    elif a.stage == "fit":
        stage_fit(a.device, a.reps.split(",") if a.reps else None)
    elif a.stage == "analyse":
        stage_analyse()
    else:
        sys.exit(f"stage {a.stage} not written yet")


if __name__ == "__main__":
    main()
