#!/usr/bin/env python3
"""EWAS Atlas traits of the vocabulary CpGs.

Written on 11 Nov 2026 before running. Two statements are checked: that 15,129 of the 28,947 vocabulary CpGs with five or more
EWAS Atlas records have five or more distinct traits, and that the most frequent traits are aging, smoking and sex,
followed by cancers and other diseases. This script computes both statements from the source table. Every result is
reported. No Generation Scotland file is read.

INPUTS
  - EWAS Atlas association table used to build the vocabulary: dataset/EWAS_Atlas_associations.tsv of the original
    MethylGPT project, downloaded 14 Aug 2024. The path is recorded in artifacts/eval/corpus_vocabulary_audit_2026-10-09.json.
    Columns: Association_ID, probe_ID, trait, case_description, case_beta, control_description, control_beta,
    correlation, p_value, rank_in_study, effect_size, study_ID, PMID.
  - Vocabulary selection: release_tables/methylgpt_vocabulary_selection_2026-10-09.csv.gz. The CpGs selected by
    "five or more EWAS Atlas association records" are 28,947 expected.
DEFINITIONS
  - A record is a row of the table.
  - A trait is the trait string with surrounding white space removed. A case-folded count (traits differing only in
    letter case merged) is given as a sensitivity.
  - For each of the 28,947 CpGs: the number of distinct traits among its records.
  - Distribution of that number, and how many CpGs have at least 1, 2, 5 and 10 distinct traits.
  - The 20 traits that cover the most of these CpGs: CpGs covered, records and studies.
  - Trait categories: the table has no category field, so none are computed.
CHECKS (abort on failure)
  1. The table totals equal the audit: 643,805 records, 301,524 CpGs, 729 traits, 1,599 studies.
  2. For every selected CpG, the record count equals ewas_atlas_association_records of the selection table, and every
     count is >= 5.
OUTPUTS  artifacts/eval/ewas_atlas_vocabulary_traits_2026-11-10.json and .csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from methylgpt_lib import ARTIFACTS, log, require, write_json  # noqa: E402

TSV = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/dataset/EWAS_Atlas_associations.tsv")
SEL = HERE / "release_tables/methylgpt_vocabulary_selection_2026-10-09.csv.gz"
OUT = ARTIFACTS / "eval/ewas_atlas_vocabulary_traits_2026-11-10"


def main():
    audit = json.loads(require(ARTIFACTS / "eval/corpus_vocabulary_audit_2026-10-09.json").read_text())["ewas_atlas"]
    t = pd.read_csv(require(TSV), sep="\t", usecols=["probe_ID", "trait", "study_ID"], dtype=str)
    t["trait"] = t.trait.fillna("").str.strip()
    checks = []
    tot = {"association_records": len(t), "cpgs": t.probe_ID.nunique(), "traits": t.trait.nunique(), "studies": t.study_ID.nunique()}
    ok1 = all(tot[k] == audit[k] for k in tot)
    checks.append({"check": "1 table totals equal the audit", "passed": ok1, "detail": f"{tot} vs {({k: audit[k] for k in tot})}"})
    sel = pd.read_csv(require(SEL))
    v = sel[sel.selected_by == "five or more EWAS Atlas association records"]
    t = t[t.probe_ID.isin(v.cpg)]
    rec = t.groupby("probe_ID").size().reindex(v.cpg).fillna(0).astype(int)
    ok2 = len(v) == 28947 and (rec.to_numpy() == v.ewas_atlas_association_records.to_numpy()).all() and (rec >= 5).all()
    checks.append({"check": "2 record counts equal the selection table", "passed": bool(ok2), "detail": f"{len(v)} CpGs; min records {rec.min()}"})
    for c in checks:
        log(f"check {c['check']}: {'pass' if c['passed'] else 'FAIL'} {c['detail']}")
    if not all(c["passed"] for c in checks):
        sys.exit("checks failed")
    nt = t.groupby("probe_ID").trait.nunique().reindex(v.cpg).fillna(0).astype(int)
    nt_cf = t.assign(tf=t.trait.str.casefold()).groupby("probe_ID").tf.nunique().reindex(v.cpg).fillna(0).astype(int)
    dist = nt.value_counts().sort_index()
    thr = {f"at_least_{k}": int((nt >= k).sum()) for k in (1, 2, 5, 10)}
    thr_cf = {f"at_least_{k}": int((nt_cf >= k).sum()) for k in (1, 2, 5, 10)}
    top = (t.groupby("trait").agg(cpgs=("probe_ID", "nunique"), records=("probe_ID", "size"), studies=("study_ID", "nunique"))
           .sort_values(["cpgs", "records"], ascending=False).head(20).reset_index())
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    res = {"cpgs": int(len(v)), "records_of_these_cpgs": int(len(t)), "traits_among_these_records": int(t.trait.nunique()),
           "distinct_traits_per_cpg": {"median": float(nt.median()), "mean": float(nt.mean()), "max": int(nt.max()), **thr},
           "distinct_traits_per_cpg_case_folded": thr_cf,
           "distribution": {int(k): int(c) for k, c in dist.items()},
           "top20_traits_by_cpgs": top.to_dict("records"),
           "trait_categories": "not computed: the EWAS Atlas association table has no category field",
           "statement_checked": {"statement": "15,129 of the 28,947 CpGs have five or more distinct traits", "found": thr["at_least_5"]},
           "inputs": {"table": str(TSV), "selection": str(SEL.relative_to(HERE))}, "checks": checks, "definitions": __doc__}
    write_json(Path(f"{OUT}.json"), res)
    rows = [{"section": "CpGs with at least k distinct traits", "key": k, "value": n} for k, n in thr.items()]
    rows += [{"section": "CpGs with at least k distinct traits (case-folded)", "key": k, "value": n} for k, n in thr_cf.items()]
    rows += [{"section": "CpGs by number of distinct traits", "key": int(k), "value": int(c)} for k, c in dist.items()]
    rows += [{"section": "top 20 traits by CpGs covered", "key": r.trait, "value": int(r.cpgs), "records": int(r.records), "studies": int(r.studies),
              "rank": int(r["rank"])} for _, r in top.iterrows()]
    pd.DataFrame(rows).to_csv(f"{OUT}.csv", index=False)
    log(f"{thr}; top traits {top.trait.head(8).tolist()}")


if __name__ == "__main__":
    main()
