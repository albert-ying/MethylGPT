#!/bin/bash
# ============================================================================
# MethylGPT Tutorials -- Unified Data Download Script
# ============================================================================
# Downloads all shared data to tutorials/shared_data/ and symlinks files
# into each tutorial directory that needs them.
#
# Usage:
#   cd tutorials
#   bash download_data.sh
#
# Requirements: gdown (pip install gdown), wget
# Idempotent: safe to run multiple times (skips existing files).
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$SCRIPT_DIR/shared_data"

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
PROBE_IDS_URL="https://www.dropbox.com/scl/fi/2n6bx7j8v0aon0kwfsghp/probe_ids_type3.csv?rlkey=ly133xlce1xxjiku6tiski6qq&st=pig4e41h&dl=1"
TINY_MODEL_GDRIVE_FOLDER="https://drive.google.com/drive/folders/1kWdmkkVQpU17uzUC6-wpNR_4UEdxGx6k"
MEDIUM_MODEL_GDRIVE_FOLDER="https://drive.google.com/drive/folders/14M4wdS83el9PAgh9TdfjSCeEcDPbz34f"
PARQUET_DATA_URL="https://www.dropbox.com/scl/fi/bbs6sxlkpbx11rhyvdfto/processed_type3_parquet_shuffled.tar.gz?rlkey=s73utmumq6xldmv3y6kh9bz75&st=8pslwy2a&dl=1"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' is not installed."
        if [ "$1" = "gdown" ]; then
            echo "  Install with: pip install gdown"
        fi
        exit 1
    fi
}

info()  { echo "[INFO]  $*"; }
ok()    { echo "[OK]    $*"; }
skip()  { echo "[SKIP]  $* (already exists)"; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
check_command gdown
check_command wget

info "Shared data directory: $SHARED_DIR"
mkdir -p "$SHARED_DIR"

# ============================= 1. probe_ids_type3.csv =====================
PROBE_IDS_FILE="$SHARED_DIR/probe_ids_type3.csv"
if [ -f "$PROBE_IDS_FILE" ]; then
    skip "$PROBE_IDS_FILE"
else
    info "Downloading probe_ids_type3.csv from Dropbox..."
    if wget -q --show-progress -O "$PROBE_IDS_FILE" "$PROBE_IDS_URL"; then
        ok "probe_ids_type3.csv downloaded"
    else
        echo "ERROR: Failed to download probe_ids_type3.csv"
        rm -f "$PROBE_IDS_FILE"
        exit 1
    fi
fi

# ============================= 2. Tiny pretrained model ===================
TINY_MODEL_DIR="$SHARED_DIR/pretrained_models/tiny"
if [ -d "$TINY_MODEL_DIR" ] && ls "$TINY_MODEL_DIR"/*.pt &>/dev/null 2>&1; then
    skip "Tiny model in $TINY_MODEL_DIR"
else
    info "Downloading tiny pretrained model from Google Drive..."
    mkdir -p "$TINY_MODEL_DIR"
    TEMP_DIR=$(mktemp -d)
    if gdown --folder "$TINY_MODEL_GDRIVE_FOLDER" -O "$TEMP_DIR" --quiet; then
        mv "$TEMP_DIR"/* "$TINY_MODEL_DIR/" 2>/dev/null || true
        rm -rf "$TEMP_DIR"
        ok "Tiny model downloaded to $TINY_MODEL_DIR"
    else
        rm -rf "$TEMP_DIR"
        echo "ERROR: Failed to download tiny model"
        exit 1
    fi
fi

# ============================= 3. Medium pretrained model ==================
MEDIUM_MODEL_DIR="$SHARED_DIR/pretrained_models/medium"
if [ -d "$MEDIUM_MODEL_DIR" ] && ls "$MEDIUM_MODEL_DIR"/*.pt &>/dev/null 2>&1; then
    skip "Medium model in $MEDIUM_MODEL_DIR"
else
    info "Downloading medium pretrained model from Google Drive..."
    mkdir -p "$MEDIUM_MODEL_DIR"
    TEMP_DIR=$(mktemp -d)
    if gdown --folder "$MEDIUM_MODEL_GDRIVE_FOLDER" -O "$TEMP_DIR" --quiet; then
        mv "$TEMP_DIR"/* "$MEDIUM_MODEL_DIR/" 2>/dev/null || true
        rm -rf "$TEMP_DIR"
        ok "Medium model downloaded to $MEDIUM_MODEL_DIR"
    else
        rm -rf "$TEMP_DIR"
        echo "ERROR: Failed to download medium model"
        exit 1
    fi
fi

# ============================= 4. Sample parquet data =====================
SAMPLE_DATA_DIR="$SHARED_DIR/sample_data"
PARQUET_EXTRACT_DIR="$SAMPLE_DATA_DIR/processed_type3_parquet_shuffled"
if [ -d "$PARQUET_EXTRACT_DIR" ] && ls "$PARQUET_EXTRACT_DIR"/*.parquet &>/dev/null 2>&1; then
    skip "Sample parquet data in $PARQUET_EXTRACT_DIR"
else
    info "Downloading processed_type3_parquet_shuffled.tar.gz from Dropbox..."
    mkdir -p "$SAMPLE_DATA_DIR"
    ARCHIVE="$SAMPLE_DATA_DIR/processed_type3_parquet_shuffled.tar.gz"
    if wget -q --show-progress -O "$ARCHIVE" "$PARQUET_DATA_URL"; then
        info "Extracting archive..."
        tar -xzf "$ARCHIVE" -C "$SAMPLE_DATA_DIR/"
        rm -f "$ARCHIVE"
        ok "Parquet data extracted to $SAMPLE_DATA_DIR/"
    else
        echo "ERROR: Failed to download parquet data"
        rm -f "$ARCHIVE"
        exit 1
    fi
fi

# ============================= 5. Symlink into tutorial directories ========
info "Creating symlinks in tutorial directories..."

# Tutorials that need probe_ids_type3.csv
PROBE_TUTORIALS=("quickstart" "get_embeddings" "imputation" "disease_prediction" "embedding_analysis")

for tutorial in "${PROBE_TUTORIALS[@]}"; do
    TUTORIAL_DIR="$SCRIPT_DIR/$tutorial"
    if [ ! -d "$TUTORIAL_DIR" ]; then
        continue
    fi

    # Create data/ subdirectory and symlink probe_ids
    mkdir -p "$TUTORIAL_DIR/data"
    LINK_TARGET="$TUTORIAL_DIR/data/probe_ids_type3.csv"
    if [ ! -e "$LINK_TARGET" ]; then
        ln -sf "$PROBE_IDS_FILE" "$LINK_TARGET"
        ok "Symlinked probe_ids_type3.csv -> $tutorial/data/"
    else
        skip "probe_ids_type3.csv in $tutorial/data/"
    fi

    # Symlink parquet data directory
    if [ -d "$PARQUET_EXTRACT_DIR" ]; then
        PARQUET_LINK="$TUTORIAL_DIR/data/processed_type3_parquet_shuffled"
        if [ ! -e "$PARQUET_LINK" ]; then
            ln -sf "$PARQUET_EXTRACT_DIR" "$PARQUET_LINK"
            ok "Symlinked parquet data -> $tutorial/data/"
        else
            skip "Parquet data link in $tutorial/data/"
        fi
    fi
done

# Symlink pretrained models into tutorials that use them
MODEL_TUTORIALS=("quickstart" "get_embeddings" "imputation" "disease_prediction" "embedding_analysis")

for tutorial in "${MODEL_TUTORIALS[@]}"; do
    TUTORIAL_DIR="$SCRIPT_DIR/$tutorial"
    if [ ! -d "$TUTORIAL_DIR" ]; then
        continue
    fi

    mkdir -p "$TUTORIAL_DIR/pretrained_models"

    # Tiny model (used by quickstart)
    if [ -d "$TINY_MODEL_DIR" ]; then
        LINK="$TUTORIAL_DIR/pretrained_models/methylgpt-base"
        if [ ! -e "$LINK" ]; then
            ln -sf "$TINY_MODEL_DIR" "$LINK"
            ok "Symlinked tiny model -> $tutorial/pretrained_models/methylgpt-base"
        fi
    fi

    # Medium model (used by get_embeddings, imputation, disease_prediction, embedding_analysis)
    if [ -d "$MEDIUM_MODEL_DIR" ]; then
        LINK="$TUTORIAL_DIR/pretrained_models/methylgpt-medium"
        if [ ! -e "$LINK" ]; then
            ln -sf "$MEDIUM_MODEL_DIR" "$LINK"
            ok "Symlinked medium model -> $tutorial/pretrained_models/methylgpt-medium"
        fi
    fi
done

# ============================= Done =======================================
echo ""
echo "========================================"
echo "  All downloads complete!"
echo "========================================"
echo ""
echo "Shared data location: $SHARED_DIR/"
echo ""
echo "Directory structure:"
echo "  shared_data/"
echo "    probe_ids_type3.csv"
echo "    pretrained_models/"
echo "      tiny/       (methylgpt-base, ~3M params)"
echo "      medium/     (methylgpt-medium, ~25M params)"
echo "    sample_data/"
echo "      processed_type3_parquet_shuffled/"
echo ""
echo "Symlinks created in: quickstart/, get_embeddings/, imputation/,"
echo "  disease_prediction/, embedding_analysis/"
echo ""
echo "You can now run any tutorial notebook."
