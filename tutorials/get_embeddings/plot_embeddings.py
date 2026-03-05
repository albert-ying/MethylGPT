import os
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap


def plot_umap_categorical(feature, data, save_as=None, dpi=600, alpha=0.6, s=3):
    """
    Plot UMAP with categorical feature coloring.
    
    Args:
        feature (str): Column name for categorical feature
        data (pd.DataFrame): DataFrame containing UMAP coordinates and feature
        save_as (str, optional): Path to save the figure
        dpi (int): Resolution for saved figure
        alpha (float): Transparency of points
        s (int): Size of points
    """
    top_categories = data[feature].value_counts().nlargest(10).index
    # Only replace categories not in the top 10 and not NA/N/A
    data[f"{feature}_top_categories"] = data[feature].apply(
        lambda x: x if x in top_categories or pd.isna(x) else "Other"
    )
    # Create the plot
    plt.figure(figsize=(7, 6))
    scatter = sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue=f"{feature}_top_categories",
        data=data,
        palette="tab20",
        alpha=alpha,
        s=s,
        edgecolor="none",
    )
    plt.title(f"UMAP of Sample Embeddings Colored by Top 10 {feature}")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    
    # Modify legend
    plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3)
    
    # Save the plot
    if save_as:
        try:
            plt.savefig(save_as, bbox_inches="tight", dpi=dpi)
            print(f"Saved plot to {save_as}")
        except Exception as e:
            print(f"Warning: Failed to save plot to {save_as}. Error: {e}")
            # Try alternative save method
            try:
                plt.savefig(save_as, bbox_inches="tight", dpi=dpi, format='png')
                print(f"Saved plot to {save_as} using alternative method")
            except Exception as e2:
                print(f"Failed to save plot with alternative method: {e2}")
    plt.close()  # Close the figure to free memory


def plot_umap_numerical(feature, data, save_as=None, dpi=600, alpha=0.6, s=3):
    """
    Plot UMAP with numerical feature coloring.
    
    Args:
        feature (str): Column name for numerical feature
        data (pd.DataFrame): DataFrame containing UMAP coordinates and feature
        save_as (str, optional): Path to save the figure
        dpi (int): Resolution for saved figure
        alpha (float): Transparency of points
        s (int): Size of points
    """
    # Convert to numeric and handle NA/N/A
    data[feature] = pd.to_numeric(data[feature], errors="coerce")

    # Create the plot
    plt.figure(figsize=(7, 6))
    plt.scatter(
        data["UMAP1"],
        data["UMAP2"],
        c=data[feature],
        cmap="viridis",
        alpha=alpha,
        s=s,
        edgecolor="none",
    )
    plt.colorbar(label=feature)
    plt.title(f"UMAP of Sample Embeddings Colored by {feature}")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")

    # Save the plot
    if save_as:
        try:
            plt.savefig(save_as, bbox_inches="tight", dpi=dpi)
            print(f"Saved plot to {save_as}")
        except Exception as e:
            print(f"Warning: Failed to save plot to {save_as}. Error: {e}")
            # Try alternative save method
            try:
                plt.savefig(save_as, bbox_inches="tight", dpi=dpi, format='png')
                print(f"Saved plot to {save_as} using alternative method")
            except Exception as e2:
                print(f"Failed to save plot with alternative method: {e2}")
    plt.close()  # Close the figure to free memory


def load_and_compile_metadata():
    """
    Load and compile metadata from multiple DSV files.
    
    Returns:
        pd.DataFrame: Compiled metadata
    """
    metadata_path = "metadata/gpt-4-1106-preview_output"
    column_names = [
        "GSM_ID",
        "PATIENT_ID",
        "race",
        "sex",
        "age",
        "genetic_info",
        "disease",
        "tissue",
        "cell_line",
        "vivo_vitro",
        "case_control",
        "group_name",
        "treatment",
        "perturbation_category",
    ]

    # Initialize an empty list to hold dataframes
    dataframes = []

    # Iterate over each file in the directory
    for filename in os.listdir(metadata_path):
        if filename.endswith(".dsv"):
            file_path = os.path.join(metadata_path, filename)
            # Read the file and add it to the list
            df = pd.read_csv(file_path, sep="|", names=column_names, header=None)
            dataframes.append(df)

    # Concatenate all dataframes into a single dataframe
    compiled_data = pd.concat(dataframes, ignore_index=True)
    compiled_data = compiled_data[compiled_data["tissue"] != "NA"]
    
    # Save compiled data
    write_path = "compiled_metadata.csv.gz"
    compiled_data.to_csv(write_path, index=False, compression="gzip")
    
    return compiled_data


def load_embeddings_and_create_umap():
    """
    Load cell embeddings and create UMAP visualization.
    
    Returns:
        tuple: (merged_data, umap_embedding)
    """
    # Read valid_cell_emb.pt pickle file
    with open("Embeddings/cell_emb.pt", "rb") as f:
        valid_cell_emb = pickle.load(f)

    # Run UMAP on valid_cell_emb
    valid_cell_emb_np = valid_cell_emb["cell_emb"]
    valid_umap_model = umap.UMAP(n_components=2, random_state=42)
    valid_umap_emb = valid_umap_model.fit_transform(valid_cell_emb_np)
    cell_list = valid_cell_emb["cell_list"]

    # Create DataFrame with UMAP coordinates
    cell_emb_df = pd.DataFrame(valid_umap_emb, index=cell_list)
    cell_emb_df.reset_index(inplace=True)
    cell_emb_df.columns = ["GSM_ID", "UMAP1", "UMAP2"]

    return cell_emb_df, valid_umap_emb


def generate_all_plots(merged_data):
    """
    Generate all UMAP plots for different features.
    
    Args:
        merged_data (pd.DataFrame): Merged data with UMAP coordinates and metadata
    """
    # Create Figures directory if it doesn't exist
    if not os.path.exists("Figures"):
        os.mkdir("Figures")

    # Generate categorical plots
    plot_umap_categorical(
        "tissue",
        merged_data,
        save_as="Figures/embeding_tissue_Aug29-11-38.png",
        s=10,
        alpha=0.8,
    )

    plot_umap_categorical(
        "sex",
        merged_data,
        save_as="Figures/embeding_sex_Aug29-11-38.png",
        s=10,
        alpha=0.8,
    )

    plot_umap_categorical(
        "race",
        merged_data,
        save_as="Figures/embeding_race_Aug29-11-38.png",
        s=10,
        alpha=0.8,
    )

    plot_umap_categorical(
        "disease",
        merged_data,
        save_as="Figures/embeding_disease_Aug29-11-38.png",
        s=10,
        alpha=0.8,
    )

    plot_umap_categorical(
        "vivo_vitro",
        merged_data,
        save_as="Figures/embeding_vivo_vitro_Aug29-11-38.png",
        s=10,
        alpha=0.8,
    )

    # Generate numerical plot
    plot_umap_numerical(
        "numeric_gsm",
        merged_data,
        save_as="Figures/embeding_numeric_gsm_Aug29-11-38.png",
        s=10,
        alpha=0.8,
    )


def main():
    """
    Main function to execute the full pipeline.
    """
    print("Loading and compiling metadata...")
    
    # Try to load existing compiled metadata, otherwise compile it
    try:
        compiled_data = pd.read_csv("compiled_metadata.csv.gz")
        print("Loaded existing compiled metadata.")
    except FileNotFoundError:
        print("Compiling metadata from DSV files...")
        compiled_data = load_and_compile_metadata()
    
    print("Loading embeddings and creating UMAP...")
    cell_emb_df, umap_emb = load_embeddings_and_create_umap()
    
    print("Merging data...")
    merged_data = pd.merge(compiled_data, cell_emb_df, on="GSM_ID")
    merged_data["numeric_gsm"] = merged_data["GSM_ID"].apply(lambda x: int(x[3:]))
    
    print("Generating plots...")
    generate_all_plots(merged_data)
    
    print("All plots generated successfully!")


# Commented out training set code from the original notebook
"""
# Code for training set (commented out from original notebook)
def plot_training_embeddings():
    with open("results/embeddings/train_cell_emb.pt", "rb") as f:
        train_cell_emb = pickle.load(f)

    # Run UMAP on train_cell_emb
    train_cell_emb_np = train_cell_emb["cell_emb"]
    train_umap_model = umap.UMAP(n_neighbors=100, n_components=2, random_state=42)
    train_umap_emb = train_umap_model.fit_transform(train_cell_emb_np)
    cell_list = train_cell_emb["cell_list"]
    umap_emb = train_umap_emb

    cell_emb_df = pd.DataFrame(umap_emb, index=cell_list)
    cell_emb_df.reset_index(inplace=True)
    cell_emb_df.columns = ["GSM_ID", "UMAP1", "UMAP2"]

    merged_data = pd.merge(compiled_data, cell_emb_df, on="GSM_ID")

    plot_umap_categorical(
        "disease",
        merged_data,
        save_as="figures/embeding_disease_train_Aug29-11-38.png",
    )

    plot_umap_categorical(
        "tissue",
        merged_data,
        save_as="figures/embeding_tissue_train_Aug29-11-38.png",
    )
"""


if __name__ == "__main__":
    main()
