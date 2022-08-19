# Import libraries and directory
from sklearn.preprocessing import MinMaxScaler
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar as get_k
from ahl_food_reformulation.pipeline import transform_data as transform
import logging
from pathlib import Path
import pandas as pd
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


logging.info("Merging dataframes and getting totals per product and household")
# Oct 2021
purch_recs_subset = transform.make_purch_records(
    purch_recs_subset, nut_subset, val_fields, prod_mast, uom, prod_codes, prod_vals
)

# Get household kcal for Oct 2021
logging.info("Creating representations - household kcal")
hh_kcal_subset = transform.hh_kcal_per_category(purch_recs_subset)

# Apply robust clustering
cluster_df, cluster_lookup = extract_clusters(
    hh_kcal_subset, 20, 5, 0.8, clustering_params
)
# Save clusters
cluster_df[["Panel Id", "clusters"]].to_csv(
    f"{PROJECT_DIR}/outputs/data/panel_clusters_robust_hh_kcal_subset.csv", index=False
)
