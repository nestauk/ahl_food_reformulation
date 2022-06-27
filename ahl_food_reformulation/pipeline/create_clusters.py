# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: ahl_food_reformulation
#     language: python
#     name: ahl_food_reformulation
# ---

# %%
# Import libraries
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

# Import project directory
from ahl_food_reformulation import PROJECT_DIR


# %%
def pca_variance(pca, filename):
    """
    Plots the sum of the variance explained for all components up to 90% variance explained.
    """
    plt.figure(figsize=(5, 5))
    plt.plot(
        range(1, pca.explained_variance_ratio_.cumsum().shape[0] + 1),
        pca.explained_variance_ratio_.cumsum(),
        marker="o",
        linestyle="--",
    )
    plt.title("Explained variance by components", fontsize=12)
    plt.xlabel("Number of components", fontsize=10)
    plt.ylabel("Cumulative explained variance", fontsize=10)
    # Add folder if not already created
    Path(f"{PROJECT_DIR}/outputs/figures/pca/").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"{PROJECT_DIR}/outputs/figures/pca/pca_variance_explained_"
        + filename
        + ".png",
        bbox_inches="tight",
    )
    plt.show()


# %%
def k_means_disp(range_n_clusters, X, filename):
    """
    For a given number of k produces silhouette and cluster plots and returns the silhouette scores.
    """
    silhouettes = {}
    # Add folder if not already created
    Path(f"{PROJECT_DIR}/outputs/figures/kmeans/").mkdir(parents=True, exist_ok=True)
    # K-means for each number of k
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.6, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        # Apply k-means with cluster number
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        # Avg silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouettes[n_clusters] = silhouette_avg  # add to dict
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]
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
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for k-means clustering with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

        plt.savefig(
            f"{PROJECT_DIR}/outputs/figures/kmeans/kmeans_"
            + str(n_clusters)
            + "_"
            + filename
            + ".png",
            bbox_inches="tight",
        )
        plt.show()
    return silhouettes


# %%
def test_clusters(df, filename, range_n_clusters):
    """
    Produces silhouette and cluster plots using the k_means_disp and then re-runs k-means for the
    best number of clusters. Returns the labels and silhouette scores.
    """
    pca = PCA(n_components=0.90)
    X = pca.fit_transform(df)
    print(
        "90% of variance explained by "
        + str(pca.explained_variance_ratio_.cumsum().shape[0])
        + " components."
    )
    pca_variance(pca, filename)
    silhouettes = k_means_disp(range_n_clusters, X, filename)
    optimum_cluster_num = max(silhouettes, key=silhouettes.get)
    print("Score: " + str(silhouettes[optimum_cluster_num]))
    print("Optimum number of clusters is " + str(optimum_cluster_num))
    # Re-assign for best n_clusters
    kmeans = KMeans(n_clusters=optimum_cluster_num)
    labels = kmeans.fit_predict(X)
    return labels, silhouettes
