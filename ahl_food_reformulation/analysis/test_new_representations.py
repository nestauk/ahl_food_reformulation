### Representations to test
# Testing new ways to represent the households purchasing.

# Representations
# - [x] Household kcal contribution to volume
# - [x] Household kcal purchased
# - [x] 1 year of household purchases
# - [ ] Absolute kcal / household size (with some weighting for children)

# Import libraries and directory
from sklearn.preprocessing import MinMaxScaler
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar as get_k
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.pipeline import create_clusters as cluster
import logging

# from pathlib import Path
# import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Get data
logging.info("loading data")
purch_recs_subset = get_k.purchase_subsets(
    202111
)  # Creating one month subset or reads file if exists - 202111
nut_subset = get_k.nutrition_subsets(
    202111
)  # Creating one month subset or reads file if exists - 202111
prod_mast = get_k.product_master()
val_fields = get_k.val_fields()
uom = get_k.uom()
prod_codes = get_k.product_codes()
prod_vals = get_k.product_values()
nut_year = get_k.nutrition()
purch_recs_year = get_k.purchase_records()

logging.info("Merging dataframes and getting totals per product and household")
# October 2021
purch_recs_subset = transform.make_purch_records(
    purch_recs_subset, nut_subset, val_fields, prod_mast, uom, prod_codes, prod_vals
)
# Whole time (1 year)
purch_recs_year = transform.make_purch_records(
    purch_recs_year, nut_year, val_fields, prod_mast, uom, prod_codes, prod_vals
)

# Testing also removing units as looks like its not a consistent form of measurement (need to confirm this with Kantar).
# It shouldn't affect the household kcal representation as it doesn't use measurements so only testing it on the kcal / volume contribution.
# Removing units
purch_recs_year_units_rem = purch_recs_year[
    purch_recs_year["Reported Volume"] != "Units"
].copy()

purch_recs_sub_units_rem = purch_recs_subset[
    purch_recs_subset["Reported Volume"] != "Units"
].copy()

# Get kcal contribution to volume for each time period (and units removed)
logging.info("Creating representations - kcal contribution to volume")
kcal_vol_subset, kcal_vol_year, kcal_vol_year_nu, kcal_vol_sub_nu = [
    transform.kcal_contribution(purch_rec)
    for purch_rec in [
        purch_recs_subset,
        purch_recs_year,
        purch_recs_year_units_rem,
        purch_recs_sub_units_rem,
    ]
]
kcal_vol_subset, kcal_vol_year, kcal_vol_year_nu, kcal_vol_sub_nu = [
    transform.scale_hh(purch_rec, MinMaxScaler())
    for purch_rec in [kcal_vol_subset, kcal_vol_year, kcal_vol_year_nu, kcal_vol_sub_nu]
]


# Get household kcal for each time period
logging.info("Creating representations - household kcal")
hh_kcal_subset, hh_kcal_year = [
    transform.hh_kcal_per_category(purch_rec)
    for purch_rec in [purch_recs_subset, purch_recs_year]
]

### Test clustering representations
logging.info(
    "Running clustering algorthm on representations - testing different numbers of k"
)
k_numbers = [10, 15, 20, 25, 30, 35, 40, 50]
(
    silh_hh_k_subset,
    silh_hh_k_year,
    silh_kv_subset,
    silh_kv_year,
    silh_kv_year_nu,
    silh_kv_sub_nu,
) = [
    cluster.test_fit_clusters(hh_rep, file_name, k_numbers)
    for hh_rep, file_name in zip(
        [
            hh_kcal_subset,
            hh_kcal_year,
            kcal_vol_subset,
            kcal_vol_year,
            kcal_vol_year_nu,
            kcal_vol_sub_nu,
        ],
        [
            "hh_kcal_sub",
            "hh_kcal_year",
            "kcal_vol_sub",
            "kcal_vol_year",
            "kcal_vol_year_nu",
            "kcal_vol_sub_nu",
        ],
    )
]

# Save cluster IDs for chosen representation (kcal contribution to volume - 1 year)
# Get labels for optimum k and representation
# labels = cluster.k_means(hh_kcal_subset, 20)

# Create household cluster labels df and save as a csv file in outputs/data
# panel_clusters = pd.DataFrame(labels, columns=["clusters"], index=hh_kcal_subset.index)

# Create path if doesn't exist and save file
# Path(f"{PROJECT_DIR}/outputs/data/").mkdir(parents=True, exist_ok=True)
# panel_clusters.to_csv(f"{PROJECT_DIR}/outputs/data/panel_clusters.csv")
