#!/usr/bin/env python3
"""Genomic annotation of the 49,156 vocabulary CpGs (Illumina manifests, hg19).

EPIC v1.0 B5 manifest: chromosome, position, RefGene names, relation to CpG island, enhancer
flags (450K and FANTOM5), DNase hypersensitivity. CpGs absent from EPIC v1 take chromosome
and position from the Zhou HM450 hg19 manifest. Output:
artifacts/cpg_annotation_2026-10-01.parquet (one row per vocabulary CpG, vocabulary order).
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, log, require

PROJECT = "/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT"
HM450 = "/Entropy/kying/09_shallowBS_is_all_you_need/data/HM450.hg19.manifest.tsv.gz"


def main():
    vocab = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))["vocab"].astype(str)
    with zipfile.ZipFile(PROJECT + "/infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip") as z:
        raw = z.read("infinium-methylationepic-v-1-0-b5-manifest-file.csv").decode()
    cols = ["Name", "CHR", "MAPINFO", "UCSC_RefGene_Name", "Relation_to_UCSC_CpG_Island", "Phantom5_Enhancers",
            "450k_Enhancer", "DNase_Hypersensitivity_NAME"]
    epic = pd.read_csv(io.StringIO(raw[raw.index("IlmnID,Name,"):].split("[Controls]")[0]), usecols=cols, low_memory=False)
    epic = epic.drop_duplicates("Name").set_index("Name")
    hm = pd.read_csv(HM450, sep="\t", usecols=["CpG_chrm", "CpG_beg", "Probe_ID"]).drop_duplicates("Probe_ID").set_index("Probe_ID")
    a = pd.DataFrame({"cpg": vocab})
    e = epic.reindex(vocab)
    h = hm.reindex(vocab)
    chr_e = e.CHR.astype("string").str.replace("chr", "", regex=False)
    chr_h = h.CpG_chrm.astype("string").str.replace("chr", "", regex=False)
    a["chr"] = chr_e.fillna(chr_h).to_numpy()
    a["pos"] = e.MAPINFO.fillna(h.CpG_beg + 1).to_numpy()
    genes = e.UCSC_RefGene_Name.fillna("").astype(str).str.split(";").map(lambda g: ";".join(sorted(set(x for x in g if x))))
    a["genes"] = genes.to_numpy()
    rel = e.Relation_to_UCSC_CpG_Island.fillna("OpenSea").astype(str)
    a["island_relation"] = rel.str.replace("N_", "", regex=False).str.replace("S_", "", regex=False).to_numpy()
    a.loc[e.index.isna() | e.CHR.isna().to_numpy(), "island_relation"] = None
    a["enhancer"] = (e["450k_Enhancer"].notna() | e.Phantom5_Enhancers.notna()).to_numpy()
    a["dnase_hs"] = e.DNase_Hypersensitivity_NAME.notna().to_numpy()
    a["in_epic_v1"] = e.CHR.notna().to_numpy()
    a.to_parquet(ARTIFACTS / "cpg_annotation_2026-10-01.parquet", index=False)
    log(f"annotated {len(a)} CpGs; chromosome known {a.chr.notna().sum()}; in EPIC v1 {a.in_epic_v1.sum()}; "
        f"with gene {(a.genes != '').sum()}; island relation {a.island_relation.value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
