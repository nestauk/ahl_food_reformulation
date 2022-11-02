# Import libraries and directory
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.pipeline import cluster_methods as cluster
import logging
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

if __name__ == "__main__":
    # Get data
    logging.info("loading data")
    purch_recs_subset = kantar.purchase_subsets(202111)
    nut_subset = kantar.nutrition_subsets(202111)
    pan_ind = kantar.household_ind()
    prod_mast = kantar.product_master()
    val_fields = kantar.val_fields()
    uom = kantar.uom()
    prod_codes = kantar.product_codes()
    prod_vals = kantar.product_values()

    logging.info("Create representations")
    # Scaler
    scaler = MinMaxScaler()
    # Converted household size
    pan_conv = transform.hh_size_conv(pan_ind)
    # Add volume measurement
    pur_vol = transform.vol_for_purch(purch_recs_subset, val_fields, prod_mast, uom)
    # Purchase and product info combined
    comb_files = transform.combine_files(
        val_fields, pur_vol, prod_codes, prod_vals, 2907
    )
    # Household kcal per category adjusted for size - Make representation
    kcal_adj_subset = transform.hh_kcal_volume_converted(
        nut_subset, pan_conv, scaler, comb_files
    )
    # Share of household kcal per category - Make representation
    kcal_share_subset = transform.hh_kcal_per_category(nut_subset, scaler, comb_files)

    logging.info("Dimensionality reduction")
    # Using PCA and UMAP to reduce dimensions
    umap_adj_sub = cluster.dimension_reduction(kcal_adj_subset, 0.97)
    umap_share_sub = cluster.dimension_reduction(kcal_share_subset, 0.97)

    logging.info("Apply kmeans to representations")
    # Get silhoutte scores for different numbers of k
    no_clusters = [10, 20, 30, 40, 50, 60, 70]
    scores_adj = cluster.kmeans_score_list(no_clusters, umap_adj_sub)
    scores_share = cluster.kmeans_score_list(no_clusters, umap_share_sub)
    # Plot results
    fig = plt.figure(figsize=(7, 5))
    sns.scatterplot(x=no_clusters, y=scores_share)
    sns.scatterplot(x=no_clusters, y=scores_adj)
    sns.lineplot(x=no_clusters, y=scores_share)
    sns.lineplot(x=no_clusters, y=scores_adj)
    fig.legend(
        labels=["Kcal share", "Kcal adjusted for household size"],
        bbox_to_anchor=(0.86, 0.35, 0.5, 0.5),
        fontsize=11,
    )
    plt.xlabel("Number of clusters", fontsize=12)
    plt.ylabel("Silhoutte score", fontsize=12)
    plt.title("Silhoutte score for different no of k", fontsize=14)
    Path(f"{PROJECT_DIR}/outputs/figures/kmeans/").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"{PROJECT_DIR}/outputs/figures/kmeans/silhoutte_scores_k_methods.png",
        bbox_inches="tight",
    )
    plt.show(block=False)

    logging.info("Cluster with centroids with optimum k")
    # Create dfs and save files with cluster assignments
    df_adj_size = cluster.centroids_cluster(
        umap_adj_sub, 6, 40, kcal_adj_subset, "panel_clusters_adj_size"
    )
    df_kcal_share = cluster.centroids_cluster(
        umap_share_sub, 6, 50, kcal_share_subset, "panel_clusters_kcal_share"
    )
