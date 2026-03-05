#!/bin/bash

RUN_PREPROCESSING=true # TOCHANGE
# Inputs (expected in the script's execution directory)
RAW_DATA_DIR="data_example"                 # TOCHANGE - Directory with raw CSV/CSV.gz files
PROBE_ID_REF="probe_ids_type3.csv"           # TOCHANGE - Probe ID reference for preprocessing & vocab

# Script and Config files (expected to be in the same directory as this script)
CONFIG_FILE="config_example.json"                 # TOCHANGE - Training configuration file
PREPROCESS_SCRIPT="preprocess_data.py" # TOCHANGE - Path to the preprocessing script
TRAIN_SCRIPT="pretraining.py"      # TOCHANGE - Path to the training script

# Training settings
DEVICE="cuda:0"                                              # TOCHANGE - Default CUDA device for training
SAVENAME_PREFIX="pretrain_run"                               # TOCHANGE - Prefix for the save name



# Outputs (will be created in the script's execution directory or a subdirectory)
PREPROCESSED_METADATA_FILE="QCed_samples_type3.csv" # Metadata file in current directory
PREPROCESSED_PARQUET_DIR="parquet_files"            # Parquet files in ./parquet_files/


# Generate a unique save name with a timestamp
SAVENAME="${SAVENAME_PREFIX}_$(date +'%Y%m%d_%H%M%S')"

echo "--- Configuration ---"
echo "Execution Directory: $(pwd)"
echo "Run Preprocessing: ${RUN_PREPROCESSING}"
echo "Raw Data Dir: ${RAW_DATA_DIR}"
echo "Probe ID Ref: ${PROBE_ID_REF}"
echo "Output Metadata File: ./${PREPROCESSED_METADATA_FILE}"
echo "Output Parquet Dir: ./${PREPROCESSED_PARQUET_DIR}"
echo "Config File: ${CONFIG_FILE}"
echo "Device: ${DEVICE}"
echo "Save Name: ${SAVENAME}"
echo "---------------------"


if [ "${RUN_PREPROCESSING}" = true ] ; then
  echo ""
  echo "--- Starting Preprocessing Step ---"
  # Create the specific directory for parquet files if it doesn't exist.
  # preprocess_data.py is expected to place parquet files into a 'parquet_files' subdir
  # under the 'output_parquet_base_dir'.
  mkdir -p "${PREPROCESSED_PARQUET_DIR}"

  echo "Executing: python ${PREPROCESS_SCRIPT} \\"
  echo "    --input_raw_csv_dir \"${RAW_DATA_DIR}\" \\"
  echo "    --probe_id_ref_path \"${PROBE_ID_REF}\" \\"
  # Output metadata (e.g., QCed_samples_type3.csv) to the current directory
  echo "    --output_metadata_dir \".\" \\"
  # Base directory for parquet output; preprocess_data.py should create PREPROCESSED_PARQUET_DIR under this.
  echo "    --output_parquet_base_dir \".\""

  python "${PREPROCESS_SCRIPT}" \
      --input_raw_csv_dir "${RAW_DATA_DIR}" \
      --probe_id_ref_path "${PROBE_ID_REF}" \
      --output_metadata_dir "." \
      --output_parquet_base_dir "."

  if [ $? -ne 0 ]; then
      echo "Preprocessing failed. Exiting."
      exit 1
  fi
  echo "--- Preprocessing Step Completed ---"
else
  echo ""
  echo "--- Skipping Preprocessing Step (RUN_PREPROCESSING is false) ---"
  echo "To enable, edit this script and set RUN_PREPROCESSING=true."
fi

echo ""
echo "--- Starting Training Step ---"
echo "Executing: python ${TRAIN_SCRIPT} \\"
  echo "    --config_file \"${CONFIG_FILE}\" \\"
  echo "    --device \"${DEVICE}\" \\"
  echo "    --savename \"${SAVENAME}\" \\"
  echo "    --probe_id_path \"${PROBE_ID_REF}\" \\"
  echo "    --parquet_data_dir \"${PREPROCESSED_PARQUET_DIR}\" \\"
  echo "    --metadata_file \"${PREPROCESSED_METADATA_FILE}\" \\"
  echo "    \"\$@\" (additional arguments passed to training script)"

python "${TRAIN_SCRIPT}" \
    --config_file "${CONFIG_FILE}" \
    --device "${DEVICE}" \
    --savename "${SAVENAME}" \
    --probe_id_path "${PROBE_ID_REF}" \
    --parquet_data_dir "${PREPROCESSED_PARQUET_DIR}" \
    --metadata_file "${PREPROCESSED_METADATA_FILE}" \
    "$@"

if [ $? -ne 0 ]; then
    echo "Training script exited with an error."
    exit 1
fi

echo "--- Training Step Script Invoked ---"
