"""Copy the gated figures, rendered without their provenance footer, into manuscript_v13/figures/.

Each figure script calls save_figure(), which writes the gated PDF with a grey provenance footer
and, when NATURE_FIGURE_NOFOOTER=1 is set, the same figure without the footer
(<stem>_nofooter.png at 300 dpi and <stem>_nofooter.pdf). This script checks that each no-footer
file is newer than its gated PDF and copies it under the manuscript's figure names.
Run: NATURE_FIGURE_NOFOOTER=1 MPLBACKEND=Agg python plot_*.py, then python export_manuscript_pngs_2026-10-02.py
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "manuscript_v13" / "figures"
FIGS = {"figure1": "figure1_2026-10-01", "figure2": "figure2_2026-10-02", "figure3": "figure3_2026-10-02",
        "figure4": "figure4_2026-10-02", "figure5": "figure5_2026-10-02", "figure6": "figure6_2026-10-02",
        "figure7": "figure7_2026-10-12",
        # manuscript numbering (order of first citation) -> plotting stems in this folder
        "supplementary_figure1": "supplementary_figure1_2026-10-02", "supplementary_figure2": "supplementary_figure2_2026-10-02",
        "supplementary_figure3": "supplementary_figure6_2026-10-03", "supplementary_figure4": "supplementary_figure3_2026-10-02",
        "supplementary_figure5": "supplementary_figure5_2026-10-02",
        # supplementary reruns of 18 Oct 2026 (Supplementary Notes 1-8)
        "supplementary_figure6": "supplementary_pretraining_curve_2026-10-18", "supplementary_figure7": "supplementary_reconstruction_diagnostics_2026-10-18",
        "supplementary_figure8": "supplementary_neighbouring_cpgs_2026-10-18", "supplementary_figure9": "supplementary_figure_cpg_dnase_2026-10-18",
        "supplementary_figure10": "si_sample_representations_2026-10-18", "supplementary_figure11": "si_embedding_pretraining_2026-10-18", "si2_figure1": "si2_embedding_designs_2026-10-19", "supplementary_figure14": "supplementary_attribution_2026-10-22", "supplementary_figure15": "supplementary_harmonization_2026-10-22",
        "supplementary_figure12": "supplementary_missing_cpgs_comparators_2026-10-18", "supplementary_figure13": "supplementary_reprogramming_2026-10-18"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, stem in FIGS.items():
        gated, png, pdf = HERE / f"{stem}.pdf", HERE / f"{stem}_nofooter.png", HERE / f"{stem}_nofooter.pdf"
        if not (gated.exists() and png.exists() and pdf.exists()) or png.stat().st_mtime < gated.stat().st_mtime - 60:
            print(f"{name}: missing or stale no-footer render for {stem}", file=sys.stderr)
            bad += 1
            continue
        shutil.copy2(png, OUT / f"{name}.png")
        shutil.copy2(pdf, OUT / f"{name}.pdf")
        print(f"{name}: {stem}_nofooter.png/.pdf")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
