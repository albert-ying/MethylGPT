#!/bin/bash

# This script downloads the tiny model for the finetuning age prediction demo
# and the AltumAge dataset.
# It requires the 'gdown' tool for Google Drive and 'wget' or 'curl' for Dropbox.
# You can typically install gdown using: pip install gdown
# wget and curl are usually pre-installed on Linux/macOS.

# --- Configuration ---
# Google Drive folder ID for the tiny model
TINY_MODEL_FOLDER_ID="1kWdmkkVQpU17uzUC6-wpNR_4UEdxGx6k"
TINY_MODEL_OUTPUT_DIR="pretrained_models/tiny"

# Google Drive folder ID for the medium model
MEDIUM_MODEL_FOLDER_ID="1lt8SF9MvoytPN3DeaxIss_ED9zNpf_Le"
MEDIUM_MODEL_OUTPUT_DIR="pretrained_models/medium"

# --- Helper Functions ---
check_command() {
    if ! command -v $1 &> /dev/null
    then
        echo "Error: $1 is not installed. Please install it to continue."
        if [ "$1" == "gdown" ]; then
            echo "You can try: pip install gdown"
        fi
        exit 1
    fi
}

# --- Sanity Checks ---
check_command gdown
check_command wget # or curl, adjust if you prefer curl

# --- Download Tiny Model ---
echo "Downloading the tiny model from Google Drive..."
echo "Folder ID: $TINY_MODEL_FOLDER_ID"
echo "Output directory: $TINY_MODEL_OUTPUT_DIR"

# Create the model output directory if it doesn't exist
mkdir -p "$TINY_MODEL_OUTPUT_DIR"

# Download the folder contents using gdown
TEMP_TINY_DOWNLOAD_DIR="./temp_tiny_download"
mkdir -p "$TEMP_TINY_DOWNLOAD_DIR"

if gdown --folder "$TINY_MODEL_FOLDER_ID" -O "$TEMP_TINY_DOWNLOAD_DIR"; then
    echo "Tiny model download successful! Moving files to $TINY_MODEL_OUTPUT_DIR..."
    mv "$TEMP_TINY_DOWNLOAD_DIR"/* "$TINY_MODEL_OUTPUT_DIR/"
    rm -rf "$TEMP_TINY_DOWNLOAD_DIR"
    echo "Tiny model files moved to $TINY_MODEL_OUTPUT_DIR."
else
    echo "Tiny model download failed. Please ensure 'gdown' is installed and you have internet access."
    rm -rf "$TEMP_TINY_DOWNLOAD_DIR"
    # exit 1 # Optionally exit if model download fails
fi

# --- Download Medium Model ---
echo "Downloading the medium model from Google Drive..."
echo "Folder ID: $MEDIUM_MODEL_FOLDER_ID"
echo "Output directory: $MEDIUM_MODEL_OUTPUT_DIR"

# Create the model output directory if it doesn't exist
mkdir -p "$MEDIUM_MODEL_OUTPUT_DIR"

# Download the folder contents using gdown
TEMP_MEDIUM_DOWNLOAD_DIR="./temp_medium_download"
mkdir -p "$TEMP_MEDIUM_DOWNLOAD_DIR"

if gdown --folder "$MEDIUM_MODEL_FOLDER_ID" -O "$TEMP_MEDIUM_DOWNLOAD_DIR"; then
    echo "Medium model download successful! Moving files to $MEDIUM_MODEL_OUTPUT_DIR..."
    mv "$TEMP_MEDIUM_DOWNLOAD_DIR"/* "$MEDIUM_MODEL_OUTPUT_DIR/"
    rm -rf "$TEMP_MEDIUM_DOWNLOAD_DIR"
    echo "Medium model files moved to $MEDIUM_MODEL_OUTPUT_DIR."
else
    echo "Medium model download failed. Please ensure 'gdown' is installed and you have internet access."
    rm -rf "$TEMP_MEDIUM_DOWNLOAD_DIR"
    # exit 1 # Optionally exit if model download fails
fi

# --- Download Processed Type3 Parquet Data ---
echo "Downloading processed_type3_parquet_shuffled.tar.gz from Dropbox..."

# Create data directory if it doesn't exist
DATA_DIR="data"
mkdir -p "$DATA_DIR"

# Dropbox URL (modified to force download)
DROPBOX_URL="https://www.dropbox.com/scl/fi/bbs6sxlkpbx11rhyvdfto/processed_type3_parquet_shuffled.tar.gz?rlkey=s73utmumq6xldmv3y6kh9bz75&st=8s562gm6&dl=1"
ARCHIVE_NAME="processed_type3_parquet_shuffled.tar.gz"

# Download using wget
if wget -O "$DATA_DIR/$ARCHIVE_NAME" "$DROPBOX_URL"; then
    echo "Data archive download successful!"
    
    # Extract the archive
    echo "Extracting $ARCHIVE_NAME to $DATA_DIR..."
    if tar -xzf "$DATA_DIR/$ARCHIVE_NAME" -C "$DATA_DIR/"; then
        echo "Archive extracted successfully!"
        
        # Remove the archive file to save space
        rm "$DATA_DIR/$ARCHIVE_NAME"
        echo "Archive file removed to save space."
    else
        echo "Failed to extract $ARCHIVE_NAME. Archive file kept for manual extraction."
    fi
else
    echo "Data archive download failed. Please check your internet connection and try again."
fi

echo "Download script completed!"
