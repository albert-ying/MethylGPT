# Tutorial: Generating and Visualizing Embeddings

This folder contains scripts and notebooks for:
1. **Downloading required data and models** from Google Drive and Dropbox
2. **Generating cell embeddings** using a pretrained MethylGPT model
3. **Visualizing** those embeddings in UMAP space with biological metadata

## Files

### Scripts
- **`download_files.sh`**  
  Downloads pretrained models (tiny and medium) from Google Drive and processed parquet data from Dropbox.
  
- **`get_embeddings.py`**  
  Generates embeddings from a pretrained MethylGPT model using processed methylation data.
  - **Input**: Processed type3 parquet files and pretrained model
  - **Output**: Cell embeddings saved in `Embeddings/cell_emb.pt`

- **`plot_embeddings.py`**  
  Loads generated embeddings and creates UMAP visualizations colored by biological metadata.
  - **Input**: Cell embeddings and compiled metadata
  - **Output**: UMAP plots saved in `Figures/` folder

### Notebooks
- **`get_embeddings.ipynb`** (original notebook version)
- **`plot_embeddings.ipynb`** (original notebook version)  
- **`test_cell_emb_structure.ipynb`** (for inspecting embedding file structure)

## Quick Start

### 1. Download Required Data and Models
```bash
# Download pretrained models and data
bash download_files.sh
```

This will create:
- `pretrained_models/tiny/` - Small model for testing
- `pretrained_models/medium/` - Larger model for better performance
- `data/processed_type3_parquet_shuffled/` - Processed methylation data

### 2. Generate Cell Embeddings
```bash
# Generate embeddings using the pretrained model
python get_embeddings.py
```

**What it does:**
- Loads the pretrained MethylGPT model
- Processes methylation data from parquet files
- Generates high-dimensional cell embeddings (typically 256-512 dimensions)
- Saves embeddings and cell IDs to `Embeddings/cell_emb.pt`

**Configuration:**
The script uses the tiny model by default. To use the medium model, edit `get_embeddings.py`:
```python
MODEL_PATH_DIR = "pretrained_models/medium"  # Change from "tiny" to "medium"
```

### 3. Create UMAP Visualizations
```bash
# Generate UMAP plots colored by biological metadata
python plot_embeddings.py
```

**Output Figures:**
The script generates UMAP plots in the `Figures/` directory:
- `embeding_tissue_*.png` - Colored by tissue type
- `embeding_sex_*.png` - Colored by biological sex
- `embeding_race_*.png` - Colored by race/ethnicity
- `embeding_disease_*.png` - Colored by disease status
- `embeding_vivo_vitro_*.png` - Colored by in vivo vs in vitro samples
- `embeding_numeric_gsm_*.png` - Colored by GSM sample number

## Data Structure

### Cell Embeddings (`Embeddings/cell_emb.pt`)
A pickle file containing:
```python
{
    "cell_emb": numpy.ndarray,    # Shape: (n_cells, embedding_dim)
    "cell_list": numpy.ndarray    # Shape: (n_cells,) - GSM sample IDs
}
```

### Metadata
The metadata comes from DSV files in `metadata/gpt-4-1106-preview_output/` containing:
- **GSM_ID**: Gene Expression Omnibus sample identifier
- **PATIENT_ID**: Patient identifier
- **race**: Race/ethnicity information
- **sex**: Biological sex (male/female)
- **age**: Age in years
- **genetic_info**: Genetic background information
- **disease**: Disease status/type
- **tissue**: Tissue type (blood, brain, etc.)
- **cell_line**: Cell line information
- **vivo_vitro**: Whether sample is in vivo or in vitro
- **case_control**: Case or control status
- **group_name**: Experimental group
- **treatment**: Treatment information
- **perturbation_category**: Type of experimental perturbation

### UMAP Visualizations
- Each plot shows cell embeddings reduced to 2D using UMAP
- Points are colored by different biological/technical metadata
- Clustering patterns reveal biological relationships captured by the model
- Plots help assess model performance and biological validity

## Technical Details

### Model Configuration
- **Architecture**: Transformer-based model for methylation data
- **Input**: CpG methylation values (49,156 sites for type3 data)
- **Output**: Cell-level embeddings from `<bos>` token
- **Device**: Configured to use `cuda:1` by default

### Data Processing
- Processes up to 100 batches by default (configurable in code)
- Uses batch size of 32
- Applies masking with ratio 0.0 (no masking for embedding generation)
- Handles variable sequence lengths with padding

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size in `get_embeddings.py`
2. **Missing metadata**: Ensure `metadata/` directory contains DSV files
3. **Model loading errors**: Check that model files were downloaded correctly
4. **Matplotlib backend issues**: Script automatically uses 'Agg' backend for compatibility

### Dependencies
- torch
- numpy
- pandas
- matplotlib
- seaborn
- umap-learn
- pickle
- tqdm

## Notes

- Created on **Jan 22** by **Jinyeop Song** (yeopjin@mit.edu)
- Updated with Python scripts and enhanced documentation
- For large datasets, consider running on GPU-enabled systems
- The embedding quality depends on the pretrained model and input data quality
