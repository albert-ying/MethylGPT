"""Write the public CpG annotation table of the MethylGPT 2 vocabulary.

release_tables/methylgpt_vocabulary_2026-10-03.csv (export_release_tables_2026-10-03.py) labels the 27K- and
450K-scenario sets (CpGs shared with EPIC v1) as on_27k and on_450k and the EWAS Atlas record count as
n_ewas_catalog_traits. Array membership, the scenario sets and the record counts are given correctly in
release_tables/methylgpt_vocabulary_selection_2026-10-09.csv.gz, so the public release carries that table and this
annotation table, which keeps only the genomic annotation columns of the 2026-10-03 table.

Run: python3 build_vocabulary_annotation_2026-11-11.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
src = pd.read_csv(HERE / "release_tables/methylgpt_vocabulary_2026-10-03.csv")
sel = pd.read_csv(HERE / "release_tables/methylgpt_vocabulary_selection_2026-10-09.csv.gz")
assert (src.cpg.values == sel.cpg.values).all() and len(src) == 49156
assert (src.on_27k.values == sel.scenario_27k_shared_with_epic_v1.values).all()
assert (src.on_450k.values == sel.scenario_450k_shared_with_epic_v1.values).all()
out = src[["cpg", "chr", "pos_hg19", "island_relation", "enhancer"]].copy()
out["pos_hg19"] = out.pos_hg19.astype("Int64")
out.to_csv(HERE / "release_tables/methylgpt_vocabulary_annotation_2026-11-11.csv", index=False)
print(len(out), "CpGs written; missing positions:", int(out.pos_hg19.isna().sum()))
