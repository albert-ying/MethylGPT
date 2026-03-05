#!/bin/bash

# This script downloads the tiny model for the finetuning age prediction demo
# and the AltumAge dataset.
# It requires the 'gdown' tool for Google Drive and 'wget' or 'curl' for Dropbox.
# You can typically install gdown using: pip install gdown
# wget and curl are usually pre-installed on Linux/macOS.

# --- Configuration ---
# Google Drive folder ID for the tiny model
TINY_MODEL_FOLDER_ID="1kWdmkkVQpU17uzUC6-wpNR_4UEdxGx6k"
MODEL_OUTPUT_DIR="pretrained_models/tiny"

# Dropbox shared link for the AltumAge dataset (ensure this is a direct download link or a link to a zip file)
# The provided link is a folder link. For scripted download, a direct link to a zip file is usually better.
# If this link directly downloads a zip when ?dl=1 is appended, this might work.
ALTUMAGE_DATASET_DROPBOX_URL="https://www.dropbox.com/scl/fo/6gajlca6kj7941q64icj9/AP2-EiDrAUxStjevwopRnhw?rlkey=f3n6zdajwokdbf27ylefqjk4g&dl=1" # Appended dl=1 for direct download attempt
ALTUMAGE_DATA_OUTPUT_DIR="data/altumage_metadata"
ALTUMAGE_ZIP_NAME="altumage_dataset.zip" # Assuming it downloads as a zip file

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
echo "Output directory: $MODEL_OUTPUT_DIR"

# Create the model output directory if it doesn't exist
mkdir -p "$MODEL_OUTPUT_DIR"

# Download the folder contents using gdown
# gdown typically downloads into the current directory, then we move it.
# Or, if gdown supports an output flag for folders that works reliably, use that.
# For simplicity, download to current and move.

TEMP_MODEL_DOWNLOAD_DIR="./temp_model_download"
mkdir -p "$TEMP_MODEL_DOWNLOAD_DIR"

if gdown --folder "$TINY_MODEL_FOLDER_ID" -O "$TEMP_MODEL_DOWNLOAD_DIR"; then
    echo "Model download successful! Moving files to $MODEL_OUTPUT_DIR..."
    # Move contents from temp_model_download/folder_name/* to MODEL_OUTPUT_DIR
    # The folder name gdown creates inside TEMP_MODEL_DOWNLOAD_DIR might be based on the GDrive folder name.
    # Assuming it creates a single folder inside, or downloads files directly.
    # This part might need adjustment based on gdown's exact behavior with --folder -O
    # For now, let's assume it downloads files directly into TEMP_MODEL_DOWNLOAD_DIR or a known subfolder.
    # If gdown creates a subfolder named after the Google Drive folder, you might need to adjust the source of mv.
    # Let's assume for now gdown downloads the *contents* into the specified -O path if it's a directory.
    mv "$TEMP_MODEL_DOWNLOAD_DIR"/* "$MODEL_OUTPUT_DIR/"
    rm -rf "$TEMP_MODEL_DOWNLOAD_DIR"
    echo "Model files moved to $MODEL_OUTPUT_DIR."
else
    echo "Model download failed. Please ensure 'gdown' is installed and you have internet access."
    rm -rf "$TEMP_MODEL_DOWNLOAD_DIR"
    # exit 1 # Optionally exit if model download fails
fi

# --- Download AltumAge Dataset from Dropbox ---
echo "\nDownloading the AltumAge dataset from Dropbox..."
echo "URL: $ALTUMAGE_DATASET_DROPBOX_URL"
echo "Output directory: $ALTUMAGE_DATA_OUTPUT_DIR"

# Create the data output directory if it doesn't exist
mkdir -p "$ALTUMAGE_DATA_OUTPUT_DIR"

# Download the dataset (assuming it's a zip file)
if wget -O "$ALTUMAGE_DATA_OUTPUT_DIR/$ALTUMAGE_ZIP_NAME" "$ALTUMAGE_DATASET_DROPBOX_URL"; then
    echo "AltumAge dataset ZIP downloaded successfully to $ALTUMAGE_DATA_OUTPUT_DIR/$ALTUMAGE_ZIP_NAME"
    # Unzip the dataset if it was a zip file
    echo "Unzipping dataset..."
    if unzip -o "$ALTUMAGE_DATA_OUTPUT_DIR/$ALTUMAGE_ZIP_NAME" -d "$ALTUMAGE_DATA_OUTPUT_DIR/"; then
        echo "Dataset unzipped successfully into $ALTUMAGE_DATA_OUTPUT_DIR/"
        # Optionally remove the zip file after extraction
        # rm "$ALTUMAGE_DATA_OUTPUT_DIR/$ALTUMAGE_ZIP_NAME"
    else
        echo "Failed to unzip dataset. Please check the file and ensure 'unzip' is installed."
    fi
else
    echo "AltumAge dataset download failed. Please check the URL, your internet connection, and ensure 'wget' is installed."
    # exit 1 # Optionally exit if data download fails
fi

echo "\nScript finished."
