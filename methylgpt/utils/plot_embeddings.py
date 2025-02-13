import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap

##

def plot_umap_categorical(feature, data, save_as=None, dpi=600):
    top_categories = data[feature].value_counts().nlargest(10).index
    # Only replace categories not in the top 10 and not NA/N/A
    data[f"{feature}_top_categories"] = data[feature].apply(
        lambda x: x if x in top_categories or pd.isna(x) else "Other"
    )
    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue=f"{feature}_top_categories",
        data=data,
        palette="tab20",
        alpha=0.6,
        s=3,
        edgecolor="none",
    )
    plt.title(f"UMAP of Sample Embeddings Colored by Top 10 {feature}")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    
    # Modify legend
    plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3)
    
    # Save the plot
    if save_as:
        plt.savefig(save_as, bbox_inches="tight", dpi=dpi)
    plt.show()

# Function to plot UMAP with a numerical feature
def plot_umap_numerical(feature, data, save_as=None, dpi=600):
    # Convert to numeric and handle NA/N/A
    data[feature] = pd.to_numeric(data[feature], errors="coerce")

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        data["UMAP1"],
        data["UMAP2"],
        c=data[feature],
        cmap="viridis",
        alpha=0.6,
        s=3,
        edgecolor="none",
    )
    plt.colorbar(label=feature)
    plt.title(f"UMAP of Sample Embeddings Colored by {feature}")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")

    # Save the plot
    if save_as:
        plt.savefig(save_as, bbox_inches="tight", dpi=dpi)
    plt.show()


##
if __name__ == "__main__":
    compiled_data = pd.read_csv("compiled_metadata.csv.gz")

    # Read cell_emb.pt pickle file
    with open("output/embedding_Aug01-10-12/valid_cell_emb.pt", "rb") as f:
        valid_cell_emb = pickle.load(f)

    # Run UMAP on valid_cell_emb
    valid_cell_emb_np = valid_cell_emb["cell_emb"]
    valid_umap_model = umap.UMAP(n_components=2, random_state=42)
    valid_umap_emb = valid_umap_model.fit_transform(valid_cell_emb_np)
    cell_list = valid_cell_emb["cell_list"]
    umap_emb = valid_umap_emb


    cell_emb_df = pd.DataFrame(umap_emb, index=cell_list)
    cell_emb_df.reset_index(inplace=True)
    cell_emb_df.columns = ["GSM_ID", "UMAP1", "UMAP2"]

    merged_data = pd.merge(compiled_data, cell_emb_df, on="GSM_ID")

    plot_umap_categorical(
        "tissue",
        merged_data,
        save_as="Figures/embeding_tissue_Aug01-10-12.png",
    )

    plot_umap_categorical(
        "sex",
        merged_data,
        save_as="Figures/embeding_sex_Aug01-10-12.png",
    )

    plot_umap_categorical(
        "race",
        merged_data,
        save_as="Figures/embeding_race_Aug01-10-12.png",
    )

    plot_umap_categorical(
        "disease",
        merged_data,
        save_as="Figures/embeding_disease_Aug01-10-12.png",
    )

