# Import libraries and directory
from array import array
from ahl_food_reformulation import PROJECT_DIR
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap
from sklearn.metrics import silhouette_score


def dimension_reduction(df: pd.DataFrame, components: float):
    """
    Reduces dimensions of df using PCA and Umap

    Args:
        df (pd.DataFrame): Pandas dataframe / household representations
        components (float): Percent of variance explained to be retained

    Returns:
        Array: Umap matrix
    """
    pca = PCA(n_components=components)
    s_reducer = umap.UMAP(n_components=2, random_state=1)
    X = pca.fit_transform(df)
    return s_reducer.fit_transform(X)


def kmeans_score(k: int, X_umap: array):
    """
    Gives silhoutte score from applying kmeans

    Args:
        X_umap (array): Umap representation of household representations
        k (int): Number of k

    Returns:
        silhoutte score
    """
    kmeans = KMeans(n_clusters=k, random_state=2)
    labels = kmeans.fit_predict(X_umap)
    return silhouette_score(X_umap, labels), labels


def kmeans_score_list(no_clusters: list, X_umap: array):
    """
    Gives silhoutte scores from numbers of k

    Args:
        no_clusters (list): List of k
        X_umap (array): Umap representation of household representations

    Returns:
        Silhoutte scores
    """
    scores = []
    for k in no_clusters:
        score, _ = kmeans_score(k, X_umap)
        scores.append(score)
    return scores


def centroids_cluster(
    umap: array, k_broad: int, k_gran: int, hh_df: pd.DataFrame, filename: str
):
    """
    Creates and saved df of clusters assignments

    Args:
        umap (array): Umap representation of household representations
        k_broad (int): K size for broader clusters
        k_gran (int): K size for more granular clusters
        hh_df (pd.DataFrame): Dataframe of hh representations
        filename (str): File extension to save


    Returns:
        Df of cluster assignments
    """
    # Kmeans to get labels
    kmeans = KMeans(n_clusters=k_gran, random_state=1)
    labels = kmeans.fit_predict(umap)
    # Get centroids
    centroids = kmeans.cluster_centers_
    # Kmeans on centroids
    kmeans = KMeans(n_clusters=k_broad, random_state=1)  # Broader number
    cent_labels = kmeans.fit_predict(centroids)

    # Create df of labels to centroid labels
    label_df = pd.DataFrame({"centroid_labels": list(cent_labels)})
    label_df.reset_index(inplace=True)
    label_df.rename(columns={"index": "clusters"}, inplace=True)
    assigned_labels = pd.DataFrame(
        {"clusters": list(labels)}, index=hh_df.index
    ).reset_index()
    assigned_labels = assigned_labels.merge(label_df, how="left", on="clusters")
    assigned_labels.set_index("Panel Id", inplace=True)

    # Save dataframe
    # Create path if doesn't exist and save file
    Path(f"{PROJECT_DIR}/outputs/data/alternative_clusters/").mkdir(
        parents=True, exist_ok=True
    )
    assigned_labels.to_csv(
        f"{PROJECT_DIR}/outputs/data/alternative_clusters/" + filename + ".csv"
    )
    return assigned_labels
