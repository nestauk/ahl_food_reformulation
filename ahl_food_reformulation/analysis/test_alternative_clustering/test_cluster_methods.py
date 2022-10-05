# Import libraries and directory
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.pipeline import cluster_methods as cluster
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


if __name__ == "__main__":
    # Get data
    logging.info("loading data")

    purch_recs_subset = kantar.purchase_subsets(202111)
    nut_subset = kantar.nutrition_subsets(202111)
    nut_year = kantar.nutrition()
    purch_recs_year = kantar.purchase_records()
    pan_ind = kantar.household_ind()

    # Scalers
    scalers = [MinMaxScaler(), StandardScaler()]

    logging.info("Creating household representations")
    logging.info("Volume of kcal")
    ## Volume of household kcal per category - Make representations
    # Converted household size
    pan_conv = transform.hh_size_conv(pan_ind)
    # Broader category
    hh_kcal_subset_broad = [
        transform.hh_kcal_volume_converted(
            purch_recs_subset, nut_subset, pan_conv, 2829, scaler
        )
        for scaler in scalers
    ]
    hh_kcal_year_broad = [
        transform.hh_kcal_volume_converted(
            purch_recs_year, nut_year, pan_conv, 2829, scaler
        )
        for scaler in scalers
    ]
    # More granular category
    hh_kcal_subset = [
        transform.hh_kcal_volume_converted(
            purch_recs_subset, nut_subset, pan_conv, 2907, scaler
        )
        for scaler in scalers
    ]
    hh_kcal_year = [
        transform.hh_kcal_volume_converted(
            purch_recs_year, nut_year, pan_conv, 2907, scaler
        )
        for scaler in scalers
    ]
    logging.info("Share of kcal")
    ## Share of kcal
    kcal_share_subset = [
        transform.hh_kcal_per_category(purch_recs_subset, nut_subset, scaler, 2907)
        for scaler in scalers
    ]
    kcal_share_year = [
        transform.hh_kcal_per_category(purch_recs_year, nut_year, scaler, 2907)
        for scaler in scalers
    ]

    logging.info("Dimensionality reduction")
    # Using PCA and UMAP to reduce dimensions
    logging.info("Volume of kcal")
    umap_vol_sub_broad = [
        cluster.dimension_reduction(df, 0.97) for df in hh_kcal_subset_broad
    ]
    umap_vol_year_broad = [
        cluster.dimension_reduction(df, 0.97) for df in hh_kcal_year_broad
    ]
    umap_vol_sub = [cluster.dimension_reduction(df, 0.97) for df in hh_kcal_subset]
    umap_vol_year = [cluster.dimension_reduction(df, 0.97) for df in hh_kcal_year]
    logging.info("Share of kcal")
    umap_share_sub = [cluster.dimension_reduction(df, 0.97) for df in kcal_share_subset]
    umap_share_year = [cluster.dimension_reduction(df, 0.97) for df in kcal_share_year]

    ### Testing different representations using kmeans
    logging.info("Plot kmeans representations")
    n_clusters = 50  # Picking 50 clusters as best performing on previous method
    ### Testing different representations using kmeans

    logging.info("Volume of kcal")
    # Oct min-max - broader product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_vol_sub_broad[0])
    cluster.plot_clusters(
        n_clusters, umap_vol_sub_broad[0], kmeans_labels, "vol-oct-minmax-broad"
    )
    # Oct standard scaler - broader product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_vol_sub_broad[1])
    cluster.plot_clusters(
        n_clusters, umap_vol_sub_broad[1], kmeans_labels, "vol-oct-standscal-broad"
    )
    # 1 year min-max - broader product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_vol_year_broad[0])
    cluster.plot_clusters(
        n_clusters, umap_vol_year_broad[0], kmeans_labels, "vol-year-minmax-broad"
    )
    # 1 year standard scaler - broader product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_vol_year_broad[1])
    cluster.plot_clusters(
        n_clusters, umap_vol_year_broad[1], kmeans_labels, "vol-year-standscal-broad"
    )
    # Oct min-max - granular product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_vol_sub[0])
    cluster.plot_clusters(n_clusters, umap_vol_sub[0], kmeans_labels, "vol-oct-minmax")
    # Oct standard scaler - granular product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_vol_sub[1])
    cluster.plot_clusters(
        n_clusters, umap_vol_sub[1], kmeans_labels, "vol-oct-standscal"
    )
    # 1 year min-max - granular product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_vol_year[0])
    cluster.plot_clusters(
        n_clusters, umap_vol_year[0], kmeans_labels, "vol-year-minmax"
    )
    # 1 year standard scaler - granular product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_vol_year[1])
    cluster.plot_clusters(
        n_clusters, umap_vol_year[1], kmeans_labels, "vol-year-standscal"
    )

    logging.info("Share of kcal")
    # Oct min-max - granular product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_share_sub[0])
    cluster.plot_clusters(
        n_clusters, umap_share_sub[0], kmeans_labels, "share-oct-minmax"
    )
    # Oct standard scaler - granular product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_share_sub[1])
    cluster.plot_clusters(
        n_clusters, umap_share_sub[1], kmeans_labels, "share-oct-stand"
    )
    # 1 year min-max - granular product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_share_year[0])
    cluster.plot_clusters(
        n_clusters, umap_share_year[0], kmeans_labels, "share-year-minmax"
    )
    # 1 year standard scaler - granular product category
    _, kmeans_labels = cluster.kmeans_score(n_clusters, umap_share_year[1])
    cluster.plot_clusters(
        n_clusters, umap_share_year[1], kmeans_labels, "share-year-stand"
    )

    ## Testing hierarcical methods
    logging.info("Testing hierarchical methods on October subset and plotting results")
    # List of methods for calculating distance
    methods = [
        "ward",
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
    ]
    print("Ocotber Subset, share of kcal, minmax scaler")
    s_scores_sub, c_coef_sub = cluster.test_methods(
        methods, umap_share_sub[0], n_clusters
    )

    # Number of K to test
    no_clusters = [10, 20, 30, 40, 50, 60, 70]
    logging.info("Getting silhoutte scores across methods and number of k")
    kmeans_scores, fclust_scores = cluster.get_s_scores(
        no_clusters, umap_share_sub[0], methods
    )

    # Print results
    print(kmeans_scores)
    print(fclust_scores)

    # Visualise best result for hierarcical clustering
    Z, _ = cluster.hierarchical_clustering(umap_share_sub[0], "ward")
    hier_score, hier_labels = cluster.fcluster_score(Z, n_clusters, umap_share_sub[0])
    cluster.plot_clusters(n_clusters, umap_share_sub[0], hier_labels)

    logging.info("Cluster with centroids")
    # Kmeans to get labels
    kmeans = KMeans(n_clusters=n_clusters)
    labels = cluster.kmeans.fit_predict(umap_share_sub[0])
    print(silhouette_score(umap_share_sub[0], labels))
    cluster.plot_clusters(n_clusters, umap_share_sub[0], labels)
    # Get centroids
    centroids = kmeans.cluster_centers_
    # Kmeans on centroids
    kmeans = KMeans(n_clusters=6)
    cent_labels = kmeans.fit_predict(centroids)

    # Create df of labels to centroid labels
    label_df = pd.DataFrame({"centroid_labels": list(cent_labels)})
    label_df.reset_index(inplace=True)
    label_df.rename(columns={"index": "clusters"}, inplace=True)
    assigned_labels = pd.DataFrame(
        {"clusters": list(labels)}, index=kcal_share_subset[0].index
    ).reset_index()
    assigned_labels = assigned_labels.merge(label_df, how="left", on="clusters")
    assigned_labels.set_index("Panel Id", inplace=True)

    # Plot 6 clusters
    cluster.plot_clusters(6, umap_share_sub[0], assigned_labels.centroid_labels.values)

    # Save dataframe
    # Create path if doesn't exist and save file
    Path(f"{PROJECT_DIR}/outputs/data/alternative_clusters/").mkdir(
        parents=True, exist_ok=True
    )
    assigned_labels.to_csv(
        f"{PROJECT_DIR}/outputs/data/alternative_clusters/kmeans_50_6_kcal_share.csv"
    )
