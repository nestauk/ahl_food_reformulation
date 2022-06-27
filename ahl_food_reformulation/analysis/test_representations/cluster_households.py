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
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

# Import project libraries and directory
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.pipeline import transform_data as td
from ahl_food_reformulation.pipeline import create_clusters as cc

# %%
# Read in datasets
pur_recs = td.create_subsets(
    202110
)  # Creating one month subset or reads file if exists - 202108
prod_mast = pd.read_csv(
    PROJECT_DIR / "inputs/data/product_master.csv", encoding="ISO-8859-1"
)
val_fields = pd.read_csv(PROJECT_DIR / "inputs/data/validation_field.csv")
uom = pd.read_csv(
    PROJECT_DIR / "inputs/data/uom.csv",
    header=0,
    names=["UOM", "Measure Description", "Factor", "Reported Volume"],
)
prod_codes = pd.read_csv(PROJECT_DIR / "inputs/data/product_attribute_coding.csv")
prod_vals = pd.read_csv(
    PROJECT_DIR / "inputs/data/product_attribute_values.csv", encoding="ISO-8859-1"
)

# %%
# Merge files
pur_recs = td.combine_files(val_fields, pur_recs, prod_mast, uom, prod_codes, prod_vals)
# Scale measures from 0-1
pur_recs[["Reported Volume", "gross_up_vol"]] = td.GroupByScaler(
    by="Reported Volume"
).fit_transform(pur_recs[["Reported Volume", "gross_up_vol"]])
# Groupby to get category totals per household
pur_pan = td.hh_total_categories(pur_recs)

# %%
# Testing different ways to normalise household purchases
pur_pan_prop = td.proportion_hh(pur_pan)
pur_pan_ss = td.scale_hh(pur_pan, StandardScaler())
pur_pan_mm = td.scale_hh(pur_pan, MinMaxScaler())

# %%
# k numbers to test
range_n_clusters = [2, 3, 4, 5, 6, 8, 10, 15, 20]

# Test k-means and save the optmum labels and scores for each hh representation
labels_pp, silh_pp = cc.test_clusters(pur_pan_prop, "prop_hh", range_n_clusters)
labels_ss, silh_ss = cc.test_clusters(pur_pan_ss, "standard_scaler", range_n_clusters)
labels_mm, silh_mm = cc.test_clusters(pur_pan_mm, "min_max_scale", range_n_clusters)

# %%
# Create household cluster labels df and save as a csv file in outputs/data
panel_clusters = pd.DataFrame(
    list(zip(labels_pp, labels_ss, labels_mm)),
    columns=["clusters_pp", "clusters_mm", "clusters_ss"],
    index=pur_pan.index,
)
# Create path if doesn't exist
Path(f"{PROJECT_DIR}/outputs/data/").mkdir(parents=True, exist_ok=True)
panel_clusters.to_csv(f"{PROJECT_DIR}/outputs/data/panel_clusters.csv")
