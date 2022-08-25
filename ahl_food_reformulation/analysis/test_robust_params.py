# Import libraries and directory
from sklearn.preprocessing import MinMaxScaler
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar as get_k
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.pipeline import create_clusters as cluster
import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import silhouette_score
import ahl_food_reformulation.analysis.clustering_interpretation as cluster_interp
from ahl_food_reformulation.pipeline.robust_clustering import (
    extract_clusters,
    clustering_params,
)

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

logging.info("Create household representation")
purch_recs_subset = transform.make_purch_records(
    purch_recs_subset, nut_subset, val_fields, prod_mast, uom, prod_codes, prod_vals
)
# Get household kcal for Oct 2021
hh_kcal_subset = transform.hh_kcal_per_category(purch_recs_subset)

logging.info("Testing clustering at different resolutions")
# Testing different sizes of communutity resolution
comm_res = 0.2
# Apply robust clustering (reduced parameters)
cluster_df, cluster_lookup = extract_clusters(
    hh_kcal_subset, 20, 5, comm_res, clustering_params
)

# Create path if doesn't exist and save file
Path(f"{PROJECT_DIR}/outputs/data/robust_params/").mkdir(parents=True, exist_ok=True)
# Save clusters (you may want ot rename existing saved file if this is an update)
cluster_df.to_csv(
    f"{PROJECT_DIR}/outputs/data/robust_params/panel_clusters"
    + str(comm_res).replace(".", "_")
    + ".csv",
    index=False,
)

# Get the number of clusters and silhouette score
print("Number of clusters =", cluster_df["clusters"].nunique())

silhouette_avg = silhouette_score(cluster_df[["x", "y"]], cluster_df["clusters"])
print(
    "For community resolution =",
    comm_res,
    "The average silhouette_score is :",
    silhouette_avg,
)

# Log
# ------
# Tested:
# Comm res | silh score  | No of clusters |
# -----------------------------------------
#   0.2    |  0.400      |      21        |
#   0.4    |  0.429      |      22        |
#   0.6    |    -        |      12        |
#   0.8    |    -        |      12        |
#   1.0    |  0.407      |      13        |


# Returning to K-means

labels = cluster.k_means(hh_kcal_subset, 30)
# Create household cluster labels df and save as a csv file in outputs/data
panel_clusters = pd.DataFrame(labels, columns=["clusters"], index=hh_kcal_subset.index)
panel_clusters.to_csv(f"{PROJECT_DIR}/outputs/data/panel_clusters_30_hhkcal_sub.csv")

panel_weights = get_k.panel_weights().query("purchase_period==202111")

# Plot cluster sizes (weighted and un-weighted)
# Use this to inform exclusion of clusters in demographic/product analysis
cluster_interp.plot_cluster_counts(panel_clusters.reset_index(), panel_weights)
