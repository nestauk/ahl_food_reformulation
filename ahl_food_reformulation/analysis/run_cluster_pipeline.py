# Import libraries and directory
from sklearn.preprocessing import MinMaxScaler
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar as get_k
from ahl_food_reformulation.pipeline import transform_data as transform
import logging
import ahl_food_reformulation.analysis.clustering_interpretation as cluster_interp
from ahl_food_reformulation.pipeline.robust_clustering import (
    extract_clusters,
    clustering_params,
)

if __name__ == "__main__":

    # Get data
    logging.info("loading data")
    purch_recs_subset = get_k.purchase_subsets(202111)  # 1 month lag
    nut_subset = get_k.nutrition_subsets(202111)  # 1 month lag
    prod_mast = get_k.product_master()
    val_fields = get_k.val_fields()
    uom = get_k.uom()
    prod_codes = get_k.product_codes()
    prod_vals = get_k.product_values()
    panel_weights = get_k.panel_weights().query("purchase_period==202111")

    logging.info("Merging dataframes and getting totals per product and household")
    # Oct 2021
    purch_recs_subset = transform.make_purch_records(
        purch_recs_subset, nut_subset, val_fields, prod_mast, uom, prod_codes, prod_vals
    )

    # Get household kcal for Oct 2021
    logging.info("Creating representations - household kcal")
    hh_kcal_subset = transform.hh_kcal_per_category(purch_recs_subset)

    # Apply robust clustering (reduced parameters)
    cluster_df, cluster_lookup = extract_clusters(
        hh_kcal_subset, 20, 5, 0.8, clustering_params
    )

    # Plot cluster sizes (weighted and un-weighted)
    # Use this to inform exclusion of clusters in demographic/product analysis
    cluster_interp.plot_cluster_counts(
        cluster_df[["Panel Id", "clusters"]], panel_weights
    )

    # Save clusters (you may want ot rename existing saved file if this is an update)
    cluster_df[["Panel Id", "clusters"]].to_csv(
        f"{PROJECT_DIR}/outputs/data/panel_clusters.csv", index=False
    )
