# Import libraries and directory
from array import array
from ahl_food_reformulation import PROJECT_DIR
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_samples, silhouette_score


def hierarchical_clustering(X_umap: array, method: str):
    """
    Reduces dimensions of df using PCA and Umap

    Args:
        X_umap (array): Umap representation of household representations
        method (str): Method to calculate the distance

    Returns:
        Z matrix and cophenet score
    """

    Z = linkage(X_umap, method)
    c, _ = cophenet(Z, pdist(X_umap))
    return Z, c


def fcluster_score(Z: array, k: int, X_umap: array):
    """
    Gives silhoutte score from flattened hieracical clusters

    Args:
        Z: linked matrix from hierarcical clustering
        k (int): Number of k
        X_umap (array): Umap representation of household representations

    Returns:
        silhoutte score
    """
    labels = fcluster(Z, k, criterion="maxclust")
    return silhouette_score(X_umap, labels), labels


def get_s_scores(no_clusters: list, X_umap: array, methods: list):
    """
    Gives lists of silhoutte scores from k-means and hierarcical clustering across numbers of k and methods

    Args:
        no_clusters (list): list of cluster numbers to test
        X_umap (array): Umap representation of household representations
        methods (list): List of methods to test (hierarcical clustering)

    Returns:
        silhoutte scores lists
    """
    fclust_scores = []
    for method in methods:
        Z, _ = hierarchical_clustering(X_umap, method)
        method_score = [fcluster_score(Z, k, X_umap)[0] for k in no_clusters]
        fclust_scores.append(method_score)
    kmeans_scores = [kmeans_score(k, X_umap)[0] for k in no_clusters]

    return kmeans_scores, fclust_scores


def plot_clusters(n_clusters: int, df: array, cluster_labels: list, filename: str):
    """
    Plot of clustering based on number of k and labels

    Args:
        no_clusters (int): k
        df (array): Umap representation of household representations
        cluster_labels (list): List of assigned labels
        filename (str): Name to use in filename

    Returns:
        Silhoutte score
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.6, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(df, cluster_labels)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        df[:, 0],
        df[:, 1],
        marker=".",
        s=30,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k",
    )

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for clustering with n_clusters = %d" % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    # Create path if doesn't exist and save file
    Path(f"{PROJECT_DIR}/outputs/figures/kmeans/").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"{PROJECT_DIR}/outputs/figures/kmeans/kmeans_"
        + str(n_clusters)
        + "_"
        + filename
        + ".png",
        bbox_inches="tight",
    )
    plt.show(block=False)

    return silhouette_avg


def test_methods(methods: list, X_umap: array, k: int):
    """
    Plots results from hierarcical methods

    Args:
        methods (list): List of methods
        X_umap (array): Umap representation of household representations
        k (int): k number of clusters

    Returns:
        Plots of clusters
    """
    s_scores = []
    c_coef = []
    # Get results for each method
    for method in methods:
        Z, c = hierarchical_clustering(X_umap, method)
        labels = list(fcluster(Z, k, criterion="maxclust"))
        s_score = silhouette_score(X_umap, labels)
        print(method)
        print("----")
        print("Cophenetic Correlation Coefficient " + str(c))
        print("Silhouette score at " + str(k) + " clusters: " + str(s_score))

        # Add scores to list
        s_scores.append(s_score)
        c_coef.append(c)

        # Plot dendrograms
        plt.figure(figsize=(15, 6))
        plt.title(method + ": Hierarchical Clustering Dendrogram (full)", fontsize=18)
        plt.xlabel("sample index", fontsize=16)
        plt.ylabel("distance", fontsize=16)
        dendrogram(
            Z,
        )
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        plt.show()

        plt.title(
            method + ": Hierarchical Clustering Dendrogram (truncated)", fontsize=14
        )
        plt.xlabel("sample index or (cluster size)", fontsize=12)
        plt.ylabel("distance", fontsize=12)
        dendrogram(
            Z,
            truncate_mode="lastp",  # show only the last p merged clusters
            p=k,  # show only the last p merged clusters
            leaf_rotation=90.0,
            leaf_font_size=12.0,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
        # Create path if doesn't exist and save file
        Path(f"{PROJECT_DIR}/outputs/figures/hierarchical/").mkdir(
            parents=True, exist_ok=True
        )
        plt.savefig(
            f"{PROJECT_DIR}/outputs/figures/hierarchical/dendrogram_" + method + ".png",
            bbox_inches="tight",
        )
        plt.show(block=False)

    return s_scores, c_coef
