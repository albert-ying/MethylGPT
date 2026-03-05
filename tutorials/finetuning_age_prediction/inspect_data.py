import pandas as pd
import os

def inspect_parquet_file(file_path):
    """Reads a Parquet file and prints basic information about its contents."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}\n")
        return

    try:
        df = pd.read_parquet(file_path)
        print(f"--- Inspecting File: {file_path} ---")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("Column names:")
        for col in df.columns:
            print(f"  - {col} (dtype: {df[col].dtype})")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}\n")

if __name__ == "__main__":
    base_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    altumage_files = [
        "altumage_metadata/altumage_train.parquet",
        "altumage_metadata/altumage_valid.parquet",
        "altumage_metadata/altumage_test.parquet"
    ]

    all_files = []
    for f_path in altumage_files: # Only use altumage files
        all_files.append(os.path.join(base_data_path, f_path))

    for file_path in all_files:
        inspect_parquet_file(file_path)

    print("Data inspection complete.")
