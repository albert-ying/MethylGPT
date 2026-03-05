#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR_BASH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values - MODIFY THESE AS NEEDED
# Paths are now relative to the script's directory
CONFIG_FILE_NAME="train_methyGPT_altumage.yml"
DEFAULT_GPUS="1" # Specify the PHYSICAL GPU ID to use (e.g., "0", "1", "2", etc.)
DATA_DIR_NAME="data/altumage_metadata/"
OUTPUT_DIR_NAME="output_altumage/"
PROBE_ID_FILE_NAME="probe_ids_type3.csv"

# Construct paths relative to the script's directory
CONFIG_FILE_PATH="$SCRIPT_DIR_BASH/$CONFIG_FILE_NAME"
DATA_DIR_PATH="$SCRIPT_DIR_BASH/$DATA_DIR_NAME"
OUTPUT_DIR_PATH="$SCRIPT_DIR_BASH/$OUTPUT_DIR_NAME"
PROBE_ID_FILE_PATH="$SCRIPT_DIR_BASH/$PROBE_ID_FILE_NAME"

# Ensure the script exits on any error
set -e

# Pass the GPU device string directly to the Python script
PYTHON_SCRIPT_GPU_ARG="cuda:$DEFAULT_GPUS"

echo "GPU configuration: Using $PYTHON_SCRIPT_GPU_ARG directly"


PYTHON_EXECUTABLE="python" # Or specify full path to python in conda env if needed
MAIN_SCRIPT_PATH="$SCRIPT_DIR_BASH/finetuning_age_main.py"

# Construct the command, ensuring paths with spaces are quoted.
# Using an array for the command is safer than eval.
CMD_ARRAY=("$PYTHON_EXECUTABLE" \
    "$MAIN_SCRIPT_PATH" \
    --config_file "$CONFIG_FILE_PATH" \
    --gpus "$PYTHON_SCRIPT_GPU_ARG" \
    --data_dir "$DATA_DIR_PATH" \
    --output_dir "$OUTPUT_DIR_PATH" \
    --probe_id_file "$PROBE_ID_FILE_PATH")

# Echo the command for debugging
echo "Executing command array:"
printf "%q " "${CMD_ARRAY[@]}"
echo ""
echo ""

# Execute the command
"$PYTHON_EXECUTABLE" \
    "$MAIN_SCRIPT_PATH" \
    --config_file "$CONFIG_FILE_PATH" \
    --gpus "$PYTHON_SCRIPT_GPU_ARG" \
    --data_dir "$DATA_DIR_PATH" \
    --output_dir "$OUTPUT_DIR_PATH" \
    --probe_id_file "$PROBE_ID_FILE_PATH"

echo ""
echo "Finetuning finished."
