# This script clusters households based on different normalised transformations.
# The output is:
# - Silloutte scores from each cluster
# - Plots depicting the results from each cluster (silloutte scores and clusters visualised in 2D)
# - CSV file with the cluster labels for each household

# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

# Import project libraries and directory
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar as get_k
from ahl_food_reformulation.pipeline import transform_data as td
from ahl_food_reformulation.pipeline import create_clusters as cc

# k numbers to test
range_n_clusters = [2, 3, 4, 5, 6, 8, 10, 15, 20]

if __name__ == "__main__":
    # Read in datasets
    purch_recs = get_k.purchase_subsets(
        202110
    )  # Creating one month subset or reads file if exists - 202108
    prod_mast = get_k.product_master()
    val_fields = get_k.val_fields()
    uom = get_k.uom()
    prod_codes = get_k.product_codes()
    prod_vals = get_k.product_values()

    # Merge files
    purch_recs = td.combine_files(
        val_fields, purch_recs, prod_mast, uom, prod_codes, prod_vals, 2907
    )

    # Scale measures from 0-1
    purch_recs["gross_up_vol"] = purch_recs.groupby("Reported Volume")[
        "gross_up_vol"
    ].apply(td.norm_variable)

    # Groupby to get category totals per household
    hh_totals = td.hh_total_categories(purch_recs)

    # Testing different ways to normalise household purchases
    hh_totals_prop = td.proportion_hh(hh_totals)  # proportion of the hh purchases
    hh_totals_ss = td.scale_hh(
        hh_totals, StandardScaler()
    )  # Standardised the hh purchases -1 to 1
    hh_totals_mm = td.scale_hh(
        hh_totals, MinMaxScaler()
    )  # Scale the hh purchases 0 to 1
    hh_totals_fcr = td.food_cat_represent(
        hh_totals
    )  # How over/under represented a category is for the hh

    # Test k-means and save the optimum labels and scores for each hh representation
    labels_pp, silh_pp = cc.test_clusters(hh_totals_prop, "prop_hh", range_n_clusters)
    labels_ss, silh_ss = cc.test_clusters(
        hh_totals_ss, "standard_scaler", range_n_clusters
    )
    labels_mm, silh_mm = cc.test_clusters(
        hh_totals_mm, "min_max_scale", range_n_clusters
    )
    labels_fcr, silh_fcr = cc.test_clusters(
        hh_totals_fcr, "over_under_rep", range_n_clusters
    )

    # Create household cluster labels df and save as a csv file in outputs/data
    panel_clusters = pd.DataFrame(
        list(zip(labels_pp, labels_ss, labels_mm, labels_fcr)),
        columns=["clusters_pp", "clusters_ss", "clusters_mm", "clusters_fcr"],
        index=hh_totals.index,
    )

    # Create path if doesn't exist
    Path(f"{PROJECT_DIR}/outputs/data/").mkdir(parents=True, exist_ok=True)

    panel_clusters.to_csv(f"{PROJECT_DIR}/outputs/data/panel_clusters.csv")
