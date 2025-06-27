import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openTSNE import TSNE as OpenTSNE
from sentence_transformers import LoggingHandler, SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# 1. Configure logging to show progress
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def get_or_create_dataframe_with_embeddings(args):
    """
    Loads a DataFrame with embeddings from a parquet cache file if available.
    Otherwise, it loads the raw data, generates embeddings, adds them to the
    DataFrame, and saves it to the cache.

    Args:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        pd.DataFrame: A DataFrame with an 'embedding' column.
    """
    # Construct a dynamic cache filename based on sample size
    sample_size_str = f"sample_{args.sample_size}" if args.sample_size else "full"
    cache_filename = f"df_with_embeddings_{sample_size_str}.parquet"
    cache_filepath = os.path.join(args.cache_dir, cache_filename)

    # Check for and load the cached DataFrame if it exists
    if os.path.exists(cache_filepath):
        logging.info(f"Loading cached DataFrame from {cache_filepath}...")
        df = pd.read_parquet(cache_filepath)
        logging.info("Cached DataFrame with embeddings loaded successfully.")
        return df

    # --- If cache not found, proceed with data loading and embedding generation ---
    logging.info(f"No cache found at {cache_filepath}. Starting data processing.")

    # Load raw data
    logging.info(f"Loading raw data from {args.input_path}...")

    # Validate the input path to prevent downstream errors and provide a clear message.
    if not isinstance(args.input_path, str) or not os.path.exists(args.input_path):
        raise FileNotFoundError(
            f"The --input_path '{args.input_path}' is invalid. "
            "It is either not a string or the path does not exist. "
            "Please provide a correct path to your parquet file or directory.",
        )

    df = pd.read_parquet(args.input_path)
    logging.info(f"Initial data shape: {df.shape}")

    # Sample the data if requested
    if args.sample_size:
        if args.sample_size > len(df):
            logging.warning(
                f"Sample size {args.sample_size} is larger than the number of rows {len(df)}. Using all rows.",
            )
        else:
            logging.info(f"Sampling {args.sample_size} rows from the dataframe.")
            df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    logging.info(f"Data shape for processing: {df.shape}")

    # Generate embeddings
    logging.info(f"Loading SentenceTransformer model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    logging.info("Starting embedding generation with multi-process pool...")
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(
        df["text"].tolist(),
        pool=pool,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    model.stop_multi_process_pool(pool)
    logging.info("Embedding generation complete.")

    # Add embeddings to the DataFrame
    df["embedding"] = embeddings.tolist()

    # Save the new DataFrame with embeddings to the cache
    os.makedirs(args.cache_dir, exist_ok=True)
    logging.info(f"Saving DataFrame with embeddings to cache at {cache_filepath}...")
    df.to_parquet(cache_filepath)
    logging.info("DataFrame cached successfully.")

    return df


def perform_dimensionality_reduction(embeddings):
    """
    Performs PCA for initial dimensionality reduction and then a fast,
    multiprocessing-enabled t-SNE on the given embeddings using openTSNE.

    Args:
        embeddings (np.ndarray): The high-dimensional input embeddings.

    Returns:
        tuple: A tuple containing the PCA results and the optimized t-SNE results.
    """
    # 1. Perform PCA. This step is fast and remains unchanged.
    logging.info("Performing PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(embeddings)

    # 2. Perform fast, parallelized t-SNE using openTSNE.
    #    The `n_jobs=-1` parameter enables multiprocessing to use all available CPU cores.
    logging.info("Performing optimized t-SNE with multiprocessing...")

    # For very large datasets (like 330k points), running PCA before t-SNE
    # can further accelerate the process without sacrificing quality.
    # We first reduce the data to 50 dimensions, a common practice.
    if embeddings.shape[1] > 50:
        logging.info(
            "Applying initial PCA to reduce dimensions to 50 for t-SNE efficiency.",
        )
        pca_for_tsne = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca_for_tsne.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings

    # Initialize openTSNE with multiprocessing enabled
    tsne_optimizer = OpenTSNE(
        n_components=2,
        n_jobs=-1,  # Use all available CPU cores
        random_state=42,
        verbose=True,  # Provides progress updates, which is helpful for long runs
    )

    tsne_result = tsne_optimizer.fit(embeddings_reduced)

    logging.info("Dimensionality reduction complete.")
    return pca_result, tsne_result


def plot_unclustered_visuals(pca_result, tsne_result, output_dir):
    """
    Plots and saves the unclustered PCA and t-SNE results.

    Args:
        pca_result (np.ndarray): The PCA-reduced data.
        tsne_result (np.ndarray): The t-SNE-reduced data.
        output_dir (str): The directory to save the plot.
    """
    plt.figure(figsize=(14, 6))
    plt.suptitle("Dimensionality Reduction Visualization", fontsize=16)

    # PCA Plot
    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title("PCA Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)

    # t-SNE Plot
    plt.subplot(1, 2, 2)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, "dimensionality_reduction_plots.png")
    plt.savefig(output_path)
    logging.info(f"Unclustered visualization plot saved to {output_path}")
    plt.close()


def run_elbow_analysis(embeddings, k_min, k_max, output_dir):
    """
    Runs the elbow method for KMeans and saves the plot.

    Args:
        embeddings (np.ndarray): The input embeddings.
        k_min (int): The minimum number of clusters to test.
        k_max (int): The maximum number of clusters to test.
        output_dir (str): The directory to save the plot.
    """
    logging.info(f"Running Elbow Method for k from {k_min} to {k_max}...")
    inertia = []
    k_range = range(k_min, k_max + 1)

    for k in tqdm(k_range, desc="Calculating Inertia"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker="o", linestyle="--")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-cluster sum of squares)")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(k_range)
    plt.grid(True)

    output_path = os.path.join(output_dir, "elbow_analysis_plot.png")
    plt.savefig(output_path)
    logging.info(f"Elbow analysis plot saved to {output_path}")
    plt.close()


def cluster_and_visualize(df, embeddings, pca_result, tsne_result, k, output_dir):
    """
    Performs KMeans clustering, adds results to the DataFrame, and visualizes.

    Args:
        df (pd.DataFrame): The DataFrame to add cluster labels to.
        embeddings (np.ndarray): The input embeddings.
        pca_result (np.ndarray): The PCA-reduced data.
        tsne_result (np.ndarray): The t-SNE-reduced data.
        k (int): The number of clusters to use.
        output_dir (str): The directory to save the plots.
    """
    logging.info(f"Performing KMeans clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    cluster_col_name = f"cluster_index"
    df[cluster_col_name] = labels
    df["pca_x"] = pca_result[:, 0]
    df["pca_y"] = pca_result[:, 1]
    df["tsne_x"] = tsne_result[:, 0]
    df["tsne_y"] = tsne_result[:, 1]

    logging.info("Cluster distribution:")
    logging.info(f"\n{df[cluster_col_name].value_counts()}")

    # --- Visualize the Results with Legends ---
    logging.info("Generating clustered visualizations...")
    plt.figure(figsize=(20, 8))
    plt.suptitle(f"Dimensionality Reduction with k={k} Clusters", fontsize=18)

    # Use a color palette with enough distinct colors
    palette = sns.color_palette("hsv", k)

    # PCA Plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca_x",
        y="pca_y",
        hue=cluster_col_name,
        palette=palette,
        data=df,
        legend="full",
        alpha=0.7,
    )
    plt.title(f"PCA with k={k} Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)

    # t-SNE Plot
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne_x",
        y="tsne_y",
        hue=cluster_col_name,
        palette=palette,
        data=df,
        legend="full",
        alpha=0.7,
    )
    plt.title(f"t-SNE with k={k} Clusters")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, f"clustered_plots_k{k}.png")
    plt.savefig(output_path)
    logging.info(f"Clustered visualization plots saved to {output_path}")
    plt.close()


def main():
    """Main function to run the clustering analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Perform sentence embedding, clustering, and visualization.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- File I/O and Cache Arguments ---
    parser.add_argument(
        "--input_path",
        type=str,
        default="/rhome/sawale/indus_traning/sentense_transformers/gen_data_stage3/filtered_sde_data/",
        help="Path to the input .parquet file. This argument is required.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output .parquet file with cluster labels. If not provided, a name will be generated.",
    )
    parser.add_argument(
        "--output_plot_dir",
        type=str,
        default="./plots",
        help="Directory to save the output plots.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory to save/load the cached DataFrame with embeddings.",
    )

    # --- Model and Embedding Arguments ---
    parser.add_argument(
        "--model_name",
        type=str,
        default="nasa-impact/nasa-smd-ibm-st-v2",
        help="Name of the SentenceTransformer model.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of rows to sample from the input data for faster processing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for sentence embedding.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Chunk size for multi-process embedding.",
    )

    # --- Clustering Control Arguments ---
    parser.add_argument(
        "--selected_k",
        type=int,
        default=None,
        help="If set, performs clustering with this value of k. If None, runs elbow analysis.",
    )
    parser.add_argument(
        "--k_min",
        type=int,
        default=4,
        help="Minimum k for elbow analysis.",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=21,
        help="Maximum k for elbow analysis.",
    )

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.input_path is None:
        parser.error("--input_path is a required argument.")

    if args.selected_k is not None and args.output_path is None:
        sample_size_str = f"sample_{args.sample_size}" if args.sample_size else "full"
        args.output_path = (
            f"clustered_data_{sample_size_str}_k{args.selected_k}.parquet"
        )

    os.makedirs(args.output_plot_dir, exist_ok=True)

    # --- Main Pipeline ---
    try:
        df = get_or_create_dataframe_with_embeddings(args)
    except FileNotFoundError as e:
        logging.error(f"Error initializing DataFrame: {e}")
        sys.exit(1)  # Exit the script if the initial data can't be loaded.

    # Extract embeddings from the dataframe
    embeddings = np.array(df["embedding"].tolist())

    pca_result, tsne_result = perform_dimensionality_reduction(embeddings)

    if args.selected_k is None:
        # --- Mode 1: Full Analysis (No K provided) ---
        logging.info("No selected_k provided. Running full analysis pipeline.")
        plot_unclustered_visuals(pca_result, tsne_result, args.output_plot_dir)
        run_elbow_analysis(embeddings, args.k_min, args.k_max, args.output_plot_dir)
        logging.info("Analysis complete. Review the elbow plot to choose an optimal k.")

    else:
        # --- Mode 2: Specific K Clustering ---
        logging.info(
            f"selected_k={args.selected_k} provided. Running clustering and saving results.",
        )

        # Create a copy to avoid modifying the original DataFrame in place before saving
        df_to_cluster = df.copy()

        cluster_and_visualize(
            df_to_cluster,
            embeddings,
            pca_result,
            tsne_result,
            args.selected_k,
            args.output_plot_dir,
        )

        # Save the final dataframe (dropping the temporary visualization columns)
        df_to_cluster.drop(columns=["pca_x", "pca_y", "tsne_x", "tsne_y"], inplace=True)
        df_to_cluster.to_parquet(args.output_path)
        logging.info(f"Final DataFrame with cluster labels saved to {args.output_path}")


if __name__ == "__main__":
    main()
