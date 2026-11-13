#!/usr/bin/env python3
"""Tables distributed with the code, as stated in the manuscript's Data and Code availability.

release_tables/methylgpt_profile_split_2026-10-03.csv.gz
    one row per profile of the assembled corpus that passed the coverage filter: profile identifier
    (GEO GSM, TCGA or ArrayExpress sample ID as provided by the EWAS Data Hub or ClockBase), dataset
    group used for the split, and split (train, dev, test, excluded_downstream = AltumAge validation
    and test profiles withheld from training, excluded_qc = removed for out-of-range values).
release_tables/methylgpt_vocabulary_2026-10-03.csv
    the 49,156 vocabulary CpGs in model order with chromosome, position, CpG-island relation and
    enhancer annotation (EPIC v1 / HM450 manifests), membership of the 27K, 450K, EPIC v1 and EPIC v2
    arrays, and the number of EWAS Catalog traits with which each CpG is associated.
Only public identifiers and annotations are written; no methylation values or participant data.
Run: python3 export_release_tables_2026-10-03.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, CORPUS_DIR, require

HERE = Path(__file__).resolve().parent
OUT = HERE / "release_tables"
EWAS = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/Jinyeop/revision_experiments/"
            "05_cpg_selection_analysis/data/cpg_selection_summary.csv")


def main():
    OUT.mkdir(exist_ok=True)
    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["id", "group", "split"])
    prof.rename(columns={"id": "profile_id", "group": "dataset_group"}).to_csv(OUT / "methylgpt_profile_split_2026-10-03.csv.gz", index=False)
    z = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    vocab = z["vocab"].astype(str)
    ann = pd.read_parquet(ARTIFACTS / "cpg_annotation_2026-10-01.parquet").set_index("cpg").reindex(vocab)
    ew = pd.read_csv(require(EWAS), usecols=["illumina_probe_id", "n_ewas_traits"]).set_index("illumina_probe_id").reindex(vocab)
    v = pd.DataFrame({"cpg": vocab, "chr": ann.chr.to_numpy(), "pos_hg19": ann.pos.to_numpy(),
                      "island_relation": ann.island_relation.to_numpy(), "enhancer": ann.enhancer.to_numpy(),
                      "on_27k": z["hm27"], "on_450k": z["hm450"], "on_epic_v1": z["epic_v1"], "on_epic_v2": z["epic_v2"],
                      "n_ewas_catalog_traits": ew.n_ewas_traits.to_numpy()})
    v.to_csv(OUT / "methylgpt_vocabulary_2026-10-03.csv", index=False)
    print(len(prof), "profiles;", prof.split.value_counts().to_dict())
    print(len(v), "CpGs; on 27K/450K/EPICv1/EPICv2:", [int(v[c].sum()) for c in ("on_27k", "on_450k", "on_epic_v1", "on_epic_v2")],
          "; >=5 EWAS traits:", int((v.n_ewas_catalog_traits >= 5).sum()))


if __name__ == "__main__":
    main()
