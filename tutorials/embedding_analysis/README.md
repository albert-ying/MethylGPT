# Tutorial: Embedding Analysis

This tutorial demonstrates how to analyze MethylGPT embeddings through visualization, clustering, and CpG-level analysis.

## Files

- **`embedding_analysis.ipynb`** -- Interactive notebook with the full analysis pipeline

## What You'll Learn

1. **UMAP Visualization** -- Plot embeddings colored by tissue type, age, sex, batch, and other metadata
2. **Clustering Analysis** -- K-Means clustering with silhouette score evaluation
3. **CpG Embedding Analysis** -- Extract and visualize token-level CpG embeddings

## Requirements

- Pretrained MethylGPT model and data (run `bash download_data.sh` from tutorials/)
- Pre-computed embeddings file, or extract fresh embeddings in the notebook
- Metadata CSV with sample annotations (tissue, age, sex, etc.)
