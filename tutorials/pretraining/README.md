# Tutorial: Generating and Visualizing Embeddings

This folder contains Jupyter notebooks for:
1. **Generating cell embeddings** using a pretrained methylGPT model.
2. **Visualizing** those embeddings in UMAP space.

## Files

- **`get_embeddings.ipynb`**  
  Generates embeddings (output of the `<bos>` token) from a pretrained methylGPT model.  
  - **Output**: Embeddings are saved in the `Embeddings/` folder.

- **`plot_embeddings.ipynb`**  
  Loads the generated embeddings and applies UMAP for dimensionality reduction.  
  - **Output**: UMAP plots are saved in the `Figures/` folder.

## Usage

1. **Generate Embeddings**  
   - Run `get_embeddings.ipynb` to produce the embedding files in `Embeddings/`.

2. **Plot Embeddings**  
   - Run `plot_embeddings.ipynb` to visualize the embeddings in UMAP space and save the figures in `Figures/`.

## Notes

- Created on **Jan 22** by **Jinyeop Song** (yeopjin@mit.edu).
- Ensure the `Embeddings/` and `Figures/` directories exist or update paths accordingly before running the notebooks.
