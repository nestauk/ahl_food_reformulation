# Import libraries and directory
from pyclbr import Function
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from array import array
from ahl_food_reformulation import PROJECT_DIR
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score


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
    kmeans = KMeans(n_clusters=k, random_state=1)
    labels = kmeans.fit_predict(X_umap)
    return silhouette_score(X_umap, labels), labels, kmeans


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
        score, _, _ = kmeans_score(k, X_umap)
        scores.append(score)
    return scores


def combine_files(
    val_fields: pd.DataFrame,
    pur_recs: pd.DataFrame,
    prod_codes: pd.DataFrame,
    prod_vals: pd.DataFrame,
    att_num: int,
):
    """
    Performs multiple merges and a few cleaning functions to combine the following files into one:
    val_fields, pur_records, prod_mast, uom, prod_codes, prod_vals
    Args:
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        prod_mast (pd.DataFrame): Pandas dataframe unique product list
        uom (pd.DataFrame): Panadas dataframe contains product measurement information
        prod_codes (pd.DataFrame): Pandas dataframe contains the codes to link products to category information
        prod_vals (pd.DataFrame): Pandas dataframe contains the product category information
        att_num (int): Product category type code number
    Returns:
        pur_recs (pandas.DateFrame): Merged pandas dataframe
    """
    val_fields.drop_duplicates(inplace=True)  # Remove duplicates
    pur_recs = pur_recs[
        [
            "PurchaseId",
            "Panel Id",
            "Period",
            "Product Code",
            "Volume",
            "Quantity",
            "Reported Volume",
            "Gross Up Weight",
        ]
    ]  # .merge(
    rst_4_ext = prod_codes[prod_codes["Attribute Number"] == att_num].copy()
    prod_code_vals = rst_4_ext.merge(prod_vals, on="Attribute Value", how="left")
    pur_recs = pur_recs.merge(
        prod_code_vals[["Product Code", "Attribute Value Description"]],
        on="Product Code",
        how="left",
    )
    pur_recs = pur_recs[
        pur_recs["Reported Volume"].notna()
    ]  # Remove purchases with no volume
    pur_recs["att_vol"] = pur_recs["Attribute Value Description"]
    return pur_recs


def vol_for_purch(
    pur_recs: pd.DataFrame,
    val_fields: pd.DataFrame,
    prod_mast: pd.DataFrame,
    uom: pd.DataFrame,
):
    """Adds volume meausrement to purchase records

    Args:
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        prod_mast (pd.DataFrame): Pandas dataframe unique product list
        uom (pd.DataFrame): Panadas dataframe contains product measurement information

    Returns:
        (pandas.DateFrame): Merged pandas dataframe
    """
    val_fields.drop_duplicates(inplace=True)  # Remove duplicates
    return (
        pur_recs.merge(
            prod_mast[["Product Code", "Validation Field"]],
            on="Product Code",
            how="left",
        )
        .merge(
            val_fields[["VF", "UOM"]],
            left_on="Validation Field",
            right_on="VF",
            how="left",
        )
        .merge(uom[["UOM", "Reported Volume"]], on="UOM", how="left")
        .drop(["Validation Field", "VF", "UOM"], axis=1)
    )


def scale_hh(df: pd.DataFrame, scaler: Function):
    """
    Applies a scaler to each row of household purchases.

    Args:
        df (pd.DataFrame): Pandas dataframe household purchases by food category
        scaler (function): Sklearn scaler to apply

    Returns:
        pd.DateFrame: Household totals scaled by rows.
    """
    return pd.DataFrame(
        scaler.fit_transform(df.T).T, columns=list(df.columns), index=df.index
    )


def nutrition_merge(nutrition: pd.DataFrame, purch_recs: pd.DataFrame, cols: list):
    """Merges the purchase records and nutrition file

    Args:
        nutrition (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        cols (list): List of columns names to merge from the nutrition dataset

    Returns:
        (pandas.DateFrame): Merged pandas dataframe
    """
    # Add unique purchase ID
    nutrition["pur_id"] = (
        nutrition["Purchase Number"].astype(str)
        + "_"
        + nutrition["Purchase Period"].astype(str)
    )
    purch_recs["pur_id"] = (
        purch_recs["PurchaseId"].astype(str) + "_" + purch_recs["Period"].astype(str)
    )
    # Merge datasets
    return purch_recs.merge(nutrition[["pur_id"] + cols], on="pur_id", how="left")


def total_product_hh_purchase(purch_recs: pd.DataFrame, cols):
    """Groups by household, measurement and product and sums the volume and kcal content.
    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        cols (list): List of cols to group (different for kcal and volume representations)
    Returns:
        (pandas.DateFrame): groupby pandas dataframe
    """
    # Remove cases where volume is zero (8 cases)
    purch_recs = purch_recs[purch_recs["Volume"] != 0].copy()
    purch_recs["Gross_up_kcal"] = (
        purch_recs["Energy KCal"] * purch_recs["Gross Up Weight"]
    )
    return (
        purch_recs.groupby(["Panel Id"] + cols)[
            ["Volume", "Energy KCal", "Quantity", "Gross Up Weight", "Gross_up_kcal"]
        ]
        .sum()
        .reset_index()
    )


def make_purch_records(
    nutrition: pd.DataFrame, purchases_comb: pd.DataFrame, cols: list
):
    """
    Merges dataframes to create purchase records df with food category and nutrition information
    Args:
        nutrition (pd.DataFrame): Pandas dataframe of purchase level nutritional information
        purchases_comb (pd.DataFrame): Combined files to give product informaion to purchases
        cols (list): Columns to use for groupby
    Returns:
        pd.DateFrame: Household totals per food category
    """
    purchases_nutrition = nutrition_merge(nutrition, purchases_comb, ["Energy KCal"])
    return total_product_hh_purchase(purchases_nutrition, cols)


def hh_kcal_per_prod(pur_recs: pd.DataFrame, kcal_col: str):
    """
    Unstacks df to show total kcal per product per household then normalises by household (rows)

    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        kcal_col (str): Energy Kcal column (weighted or unweighted)

    Returns:
        (pd.DateFrame): Kcal totals per product per household
    """
    prod_kcal = (
        pur_recs.set_index(["Panel Id", "att_vol"])[[kcal_col]]
        .unstack(["att_vol"])
        .fillna(0)
    )
    prod_kcal.columns = prod_kcal.columns.droplevel()
    prod_kcal.drop(list(prod_kcal.filter(regex="Oil")), axis=1, inplace=True)
    prod_kcal.drop(list(prod_kcal.filter(regex="Rice")), axis=1, inplace=True)
    return prod_kcal


def hh_kcal_per_category(
    nut: pd.DataFrame,
    scaler_type: Function,
    comb_files: pd.DataFrame,
):
    """
    Unstacks df to show total kcal per product per household then normalises by household (rows)
    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut (pd.DataFrame): Pandas dataframe contains nutritional information per purchase record
        scaler_type: Scaler function to apply to normalise data
        cat (int): Number ID of product category
        comb_files (pd.DataFrame): Combined purchase and product info
    Returns:
        (pd.DateFrame): Kcal totals per product per household normalised by total household kcal
    """
    purch_recs_comb = make_purch_records(nut, comb_files, ["att_vol"])
    return scale_hh(
        hh_kcal_per_prod(purch_recs_comb, "Energy KCal"), scaler_type
    )  # Scale the hh purchases 0 to 1


def plot_clusters(
    title: str, n_clusters: int, df: array, cluster_labels: list, clusterer
):
    """
    Plot of clustering based on number of k and labels
    Args:
        title (str): Description of representation to add to start of plot title
        no_clusters (int): k
        df (array): Umap representation of household representations
        cluster_labels (list): List of assigned labels
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
        title + " Silhouette analysis for clustering with n_clusters = %d" % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    plt.show()
