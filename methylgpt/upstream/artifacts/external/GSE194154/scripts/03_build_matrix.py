#!/usr/bin/env python3
"""Build vocabulary-level beta and coverage matrices for GSE194154 (RRBS, whole blood, hg38).

Input call files (raw/GSM*.txt.gz, one per sample) have the columns
  chrBase  chr  base  strand  coverage  freqC  freqT
where `base` is a 1-based hg38 position, chrBase == "C" marks the + strand C of a CpG and
chrBase == "G" marks the G of a CpG (the C of the - strand), `strand` is always "F",
freqC = int(100 * m / coverage) truncated (computed as m / coverage * 100 in float64) and
freqT = 100 - freqC. These properties are checked for every row below.

Coordinate convention (verified in 02_check_convention.py against the GRCh38 FASTA):
  Zhou EPICv2 hg38 manifest CpG_beg = 0-based start of the CG dinucleotide, so the
  + strand C (chrBase "C" rows) is at 1-based CpG_beg + 1 and the G / - strand C
  (chrBase "G" rows) is at 1-based CpG_beg + 2.
Each call is keyed to the 1-based + strand C position (base for "C" rows, base - 1 for "G"
rows); methylated and total reads of the two strands are summed.

Methylated reads per row are recovered from (coverage, freqC):
  coverage <= 100: the integer m in [0, coverage] with int(m / coverage * 100) == freqC
                   (unique except coverage 100 with freqC 28 or 56, where the mean of the two
                   candidates is used);
  coverage  > 100: midpoint of the integers m with floor(100 * m / coverage) == freqC
                   (error < 0.5 percentage points of beta).

Outputs (in the GSE194154 directory):
  betas_cov10.parquet  GSM x vocabulary CpG (>= 10 reads in >= 1 sample); float32 beta, NaN if < 10 reads
  coverage.parquet     same shape; int32 total reads (0 where absent)
  cpg_mapping.csv      every vocabulary ID with its hg38 coordinate and observation summary
  sample_qc.csv        per-sample call-file and coverage summary
  mapping_check.json   array-vs-RRBS mean-beta correlation and summary numbers
"""
import json
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
VOCAB = Path("/Entropy/kying/agent-worktrees/methylgpt-retrain-2026-09-30/retrain/artifacts/release/methylgpt2-1.0/vocab.txt")
MANIFEST = Path("/Entropy/kying/gs_release/data/EPICv2.hg38.manifest.tsv.gz")
ARRAY_MEAN = Path("/Entropy/kying/agent-worktrees/methylgpt-retrain-2026-09-30/retrain/artifacts/external/corpus_whole_blood_mean_beta.csv")
MIN_COV = 10
N_WORKERS = 24
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
CHR2I = {c: i + 1 for i, c in enumerate(CHROMS)}
SCALE = np.int64(10**9)

# worker globals (set by init_worker)
VK_SORTED = None   # sorted int64 keys of mapped vocabulary CpGs
VK_ORDER = None    # column index (in mapped order) of each sorted key
M_TABLE = None     # (101, 101): mean methylated count for (coverage <= 100, freqC); NaN if impossible


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build_count_table():
    tab = np.full((101, 101), np.nan)
    for c in range(1, 101):
        m = np.arange(c + 1)
        f = (m / c * 100).astype(np.int64)
        s = np.bincount(f, weights=m, minlength=101)
        k = np.bincount(f, minlength=101)
        tab[c, k > 0] = s[k > 0] / k[k > 0]
    return tab


def init_worker(vk_sorted, vk_order):
    global VK_SORTED, VK_ORDER, M_TABLE
    import pyarrow as pa
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    VK_SORTED, VK_ORDER = vk_sorted, vk_order
    M_TABLE = build_count_table()


def build_vocab_map():
    vocab = pd.read_csv(VOCAB, header=None, names=["cpg"]).cpg
    if not vocab.is_unique:
        sys.exit("vocabulary has duplicate IDs")
    man = pd.read_csv(MANIFEST, sep="\t", usecols=["CpG_chrm", "CpG_beg", "CpG_end", "Probe_ID"],
                      dtype={"CpG_chrm": str})
    man["cpg"] = man.Probe_ID.str.split("_").str[0]
    man = man[man.cpg.isin(set(vocab))].drop_duplicates("cpg", keep="first")
    vm = pd.DataFrame({"cpg": vocab}).merge(
        man[["cpg", "Probe_ID", "CpG_chrm", "CpG_beg", "CpG_end"]], on="cpg", how="left")
    vm["in_manifest"] = vm.CpG_chrm.notna()
    width = vm.CpG_end - vm.CpG_beg
    # CpG targets span 2 bp; the 21 'ch.' (CpH) probes span 1 bp and cannot appear in CpG calls
    vm["cpg_target"] = vm.in_manifest & (width == 2) & vm.CpG_chrm.isin(list(CHR2I))
    vm["CpG_beg"] = vm.CpG_beg.astype("Int64")
    vm["CpG_end"] = vm.CpG_end.astype("Int64")
    vm["pos_C_plus_1based"] = (vm.CpG_beg + 1).where(vm.cpg_target)
    vm["pos_G_1based"] = (vm.CpG_beg + 2).where(vm.cpg_target)
    chr_code = vm.CpG_chrm.map(CHR2I).astype("Int64")
    vm["key"] = (chr_code * int(SCALE) + vm.pos_C_plus_1based).where(vm.cpg_target)
    return vm


def read_calls(path):
    import pyarrow as pa
    import pyarrow.csv as pacsv
    dict_t = pa.dictionary(pa.int32(), pa.string())
    tbl = pacsv.read_csv(
        path, parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            include_columns=["chrBase", "chr", "base", "coverage", "freqC", "freqT"],
            column_types={"chrBase": dict_t, "chr": dict_t, "base": pa.int64(),
                          "coverage": pa.int64(), "freqC": pa.int64(), "freqT": pa.int64()}))
    out = {}
    for col in ("chrBase", "chr"):
        arr = tbl.column(col).combine_chunks()
        if arr.null_count:
            raise ValueError(f"{path}: missing values in {col}")
        out[col + "_codes"] = arr.indices.to_numpy(zero_copy_only=False).astype(np.int64)
        out[col + "_dict"] = arr.dictionary.to_pylist()
    for col in ("base", "coverage", "freqC", "freqT"):
        out[col] = tbl.column(col).to_numpy()
    return out


def process_sample(task):
    gsm, path = task
    t0 = time.time()
    d = read_calls(path)
    n_rows = len(d["base"])
    if not set(d["chrBase_dict"]) <= {"C", "G"}:
        raise ValueError(f"{gsm}: unexpected chrBase values {d['chrBase_dict']}")
    unknown_chr = sorted(set(d["chr_dict"]) - set(CHR2I))
    code_map = np.array([CHR2I.get(c, 0) for c in d["chr_dict"]], dtype=np.int64)
    chrc = code_map[d["chr_codes"]]
    is_g = (d["chrBase_codes"] == d["chrBase_dict"].index("G")) if "G" in d["chrBase_dict"] \
        else np.zeros(n_rows, dtype=bool)
    cov = d["coverage"]
    f = d["freqC"]
    if (cov <= 0).any() or (f < 0).any() or (f > 100).any():
        raise ValueError(f"{gsm}: coverage/freqC out of range")
    n_freq_sum_not_100 = int(((f + d["freqT"]) != 100).sum())

    # methylated reads
    small = cov <= 100
    m_tab = np.full(n_rows, np.nan)
    m_tab[small] = M_TABLE[cov[small], f[small]]
    m_lo = (f * cov + 99) // 100
    m_hi = np.where(f >= 100, cov, ((f + 1) * cov + 99) // 100 - 1)
    m_mid = np.where(m_lo <= m_hi, (m_lo + m_hi) / 2.0, np.nan)
    m_est = np.where(small, m_tab, m_mid)
    no_count = np.isnan(m_est)
    m_est = np.where(no_count, f * cov / 100.0, m_est)
    # rounding hypothesis (round-half-up of 100 m / coverage), for documentation only
    r_lo = np.maximum(((2 * f - 1) * cov + 199) // 200, 0)
    r_hi = np.minimum(((2 * f + 1) * cov + 199) // 200 - 1, cov)

    # strand combination: key = chromosome code * 1e9 + 1-based + strand C position
    key = chrc * SCALE + d["base"] - is_g.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key, cov_s, m_s, g_s = key[order], cov[order], m_est[order], is_g[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ukey = key[starts]
    gcov = np.add.reduceat(cov_s, starts)
    gm = np.add.reduceat(m_s, starts)
    n_per = np.diff(np.r_[starts, n_rows])
    n_g = np.add.reduceat(g_s.astype(np.int64), starts)
    n_dup_sites = int(((n_g > 1) | ((n_per - n_g) > 1)).sum())

    ok = gcov >= MIN_COV
    chr_of = ukey // SCALE
    is_y = chr_of == CHR2I["chrY"]

    # vocabulary lookup
    p = np.searchsorted(ukey, VK_SORTED)
    pc = np.minimum(p, len(ukey) - 1)
    found = (p < len(ukey)) & (ukey[pc] == VK_SORTED)
    col = VK_ORDER[found]
    sel_pos = pc[found]
    v_cov = gcov[sel_pos]
    qc = {
        "gsm": gsm,
        "n_rows": n_rows,
        "n_rows_C_plus": int((~is_g).sum()),
        "n_rows_G_minus": int(is_g.sum()),
        "unknown_chromosomes": ";".join(unknown_chr),
        "n_sites_with_duplicate_strand_rows": n_dup_sites,
        "n_rows_freq_sum_not_100": n_freq_sum_not_100,
        "n_rows_cov_le100": int(small.sum()),
        "n_rows_no_consistent_count": int(no_count.sum()),
        "n_rows_cov_le100_inconsistent_with_exact_floor": int((small & (m_lo > m_hi)).sum()),
        "n_rows_cov_le100_inconsistent_with_rounding": int((small & (r_lo > r_hi)).sum()),
        "n_cpg_sites": int(len(ukey)),
        "n_cpg_sites_cov10": int(ok.sum()),
        "median_cov_sites_cov10": float(np.median(gcov[ok])) if ok.any() else np.nan,
        "mean_beta_sites_cov10": float((gm[ok] / gcov[ok]).mean()) if ok.any() else np.nan,
        "n_chrY_sites_cov10": int((ok & is_y).sum()),
        "frac_chrY_sites_cov10": float((ok & is_y).sum() / max(ok.sum(), 1)),
        "n_vocab_cpgs_any_cov": int(found.sum()),
        "n_vocab_cpgs_cov10": int((v_cov >= MIN_COV).sum()),
        "n_vocab_cpgs_cov10_both_strands": int(((v_cov >= MIN_COV) & (n_per[sel_pos] >= 2)).sum()),
        "seconds": round(time.time() - t0, 1),
    }
    return gsm, col.astype(np.int64), gm[sel_pos], v_cov.astype(np.int64), qc


def main():
    for p in (VOCAB, MANIFEST, ARRAY_MEAN, OUT / "samples.csv"):
        if not p.exists():
            sys.exit(f"Cannot proceed - required data file not found: {p}")
    samples = pd.read_csv(OUT / "samples.csv", dtype={"patient_id": str})
    tasks = []
    for gsm, lf in zip(samples.gsm, samples.local_file):
        path = OUT / lf
        if not path.exists():
            sys.exit(f"Cannot proceed - required data file not found: {path}")
        tasks.append((gsm, str(path)))

    vmap = build_vocab_map()
    mapped = vmap[vmap.cpg_target].reset_index(drop=True)
    vkeys = mapped.key.astype(np.int64).to_numpy()
    if pd.Index(vkeys).has_duplicates:
        sys.exit("two vocabulary IDs share one hg38 CpG")
    vk_order = np.argsort(vkeys, kind="stable")
    vk_sorted = vkeys[vk_order]
    log(f"vocabulary {len(vmap)}; in EPICv2 manifest {int(vmap.in_manifest.sum())}; "
        f"CpG targets with hg38 coordinates {len(mapped)}")

    n, V = len(tasks), len(mapped)
    M = np.zeros((n, V), dtype=np.float64)
    C = np.zeros((n, V), dtype=np.int64)
    row = {g: i for i, (g, _) in enumerate(tasks)}
    qcs = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(vk_sorted, vk_order)) as ex:
        for k, (gsm, col, m, cov, qc) in enumerate(ex.map(process_sample, tasks), 1):
            i = row[gsm]
            M[i, col] = m
            C[i, col] = cov
            qcs.append(qc)
            if k % 10 == 0 or k == n:
                log(f"processed {k}/{n} samples")
    s_idx = samples.set_index("gsm")
    qc = pd.DataFrame(qcs).set_index("gsm").loc[[g for g, _ in tasks]]
    qc.insert(0, "sample_label", s_idx.sample_label.loc[qc.index])
    qc.insert(1, "reported_sex", s_idx.sex.loc[qc.index])
    qc.to_csv(OUT / "sample_qc.csv")

    obs = C >= MIN_COV
    keep = obs.any(axis=0)
    cols = mapped.cpg.to_numpy()[keep]
    gsms = pd.Index([g for g, _ in tasks], name="gsm")
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(obs, M / np.where(C > 0, C, 1), np.nan).astype(np.float32)
    betas_df = pd.DataFrame(beta[:, keep], index=gsms, columns=pd.Index(cols, name="cpg"))
    cov_df = pd.DataFrame(C[:, keep].astype(np.int32), index=gsms, columns=pd.Index(cols, name="cpg"))
    betas_df.to_parquet(OUT / "betas_cov10.parquet")
    cov_df.to_parquet(OUT / "coverage.parquet")
    log(f"wrote betas_cov10.parquet and coverage.parquet: {betas_df.shape}")

    # per-CpG summary for every vocabulary ID
    n_obs = obs.sum(axis=0)
    rrbs_mean = np.where(n_obs > 0, np.where(obs, beta, 0.0).sum(axis=0) / np.maximum(n_obs, 1), np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (never >= 10 reads)
        med_cov = np.nanmedian(np.where(obs, C, np.nan), axis=0)
    summ = pd.DataFrame({"cpg": mapped.cpg, "n_samples_any_read": (C > 0).sum(axis=0),
                         "n_samples_cov10": n_obs, "median_cov_when_cov10": med_cov,
                         "rrbs_mean_beta_cov10": rrbs_mean})
    arr = pd.read_csv(ARRAY_MEAN).set_index("cpg")
    out_map = vmap.drop(columns=["key"]).merge(summ, on="cpg", how="left")
    for c_ in ("n_samples_any_read", "n_samples_cov10"):
        out_map[c_] = out_map[c_].fillna(0).astype(int)
    out_map["in_matrix"] = out_map.n_samples_cov10 > 0
    out_map = out_map.merge(arr[["blood_array_mean_beta"]], left_on="cpg", right_index=True, how="left")
    out_map.rename(columns={"CpG_beg": "CpG_beg_0based"}).to_csv(OUT / "cpg_mapping.csv", index=False)

    # mapping check: CpGs observed (>= 10 reads) in >= 50% of samples
    half = out_map[(out_map.n_samples_cov10 >= 0.5 * n) & out_map.blood_array_mean_beta.notna()]
    r = float(np.corrcoef(half.rrbs_mean_beta_cov10, half.blood_array_mean_beta)[0, 1])
    rho = float(half.rrbs_mean_beta_cov10.rank().corr(half.blood_array_mean_beta.rank()))
    mad = float((half.rrbs_mean_beta_cov10 - half.blood_array_mean_beta).abs().mean())
    allobs = out_map[out_map.in_matrix & out_map.blood_array_mean_beta.notna()]
    r_all = float(np.corrcoef(allobs.rrbs_mean_beta_cov10, allobs.blood_array_mean_beta)[0, 1])
    per_sample = qc.n_vocab_cpgs_cov10
    check = {
        "n_samples": int(n),
        "vocab_size": int(len(vmap)),
        "vocab_in_EPICv2_manifest": int(vmap.in_manifest.sum()),
        "vocab_cpg_targets_with_hg38_coordinates": int(V),
        "vocab_cpgs_with_any_read": int((C > 0).any(axis=0).sum()),
        "matrix_columns_cov10_in_any_sample": int(keep.sum()),
        "vocab_cpgs_cov10_in_ge50pct_samples": int((n_obs >= 0.5 * n).sum()),
        "vocab_cpgs_cov10_in_all_samples": int((n_obs == n).sum()),
        "per_sample_vocab_cpgs_cov10_median": float(per_sample.median()),
        "per_sample_vocab_cpgs_cov10_q25": float(per_sample.quantile(0.25)),
        "per_sample_vocab_cpgs_cov10_q75": float(per_sample.quantile(0.75)),
        "per_sample_vocab_cpgs_cov10_min": int(per_sample.min()),
        "per_sample_vocab_cpgs_cov10_max": int(per_sample.max()),
        "mapping_check_n_cpgs": int(len(half)),
        "mapping_check_pearson_r": round(r, 4),
        "mapping_check_spearman_rho": round(rho, 4),
        "mapping_check_mean_abs_diff": round(mad, 4),
        "all_observed_n_cpgs": int(len(allobs)),
        "all_observed_pearson_r": round(r_all, 4),
        "rows_total": int(qc.n_rows.sum()),
        "rows_cov_le100": int(qc.n_rows_cov_le100.sum()),
        "rows_no_consistent_count": int(qc.n_rows_no_consistent_count.sum()),
        "rows_cov_le100_inconsistent_with_exact_floor": int(qc.n_rows_cov_le100_inconsistent_with_exact_floor.sum()),
        "rows_cov_le100_inconsistent_with_rounding": int(qc.n_rows_cov_le100_inconsistent_with_rounding.sum()),
        "rows_freq_sum_not_100": int(qc.n_rows_freq_sum_not_100.sum()),
        "sites_with_duplicate_strand_rows": int(qc.n_sites_with_duplicate_strand_rows.sum()),
    }
    with open(OUT / "mapping_check.json", "w") as fh:
        json.dump(check, fh, indent=2)
    log(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
