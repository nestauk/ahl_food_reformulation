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
from ahl_food_reformulation.pipeline import transform_data as td
from matplotlib import pyplot as plt

# Import project libraries and directory
from ahl_food_reformulation import PROJECT_DIR

# %%
# Read in files
pur_recs = td.create_subsets(
    202108
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

panel_clusters = pd.read_csv(f"{PROJECT_DIR}/outputs/data/panel_clusters.csv")

nutrition = pd.read_csv(
    PROJECT_DIR / "inputs/data/nutrition_data.csv", encoding="ISO-8859-1"
)

# %%
# Combine and merge files
pur_recs = td.combine_files(val_fields, pur_recs, prod_mast, uom, prod_codes, prod_vals)
nut_subset = nutrition[nutrition["Purchase Period"] == 202108].copy()
nut_subset["pur_id"] = (
    nut_subset["Purchase Number"].astype(str)
    + "_"
    + nut_subset["Purchase Period"].astype(str)
)
pur_recs["pur_id"] = (
    pur_recs["PurchaseId"].astype(str) + "_" + pur_recs["Period"].astype(str)
)
nut_subset = nut_subset[
    [
        "pur_id",
        "Energy KJ",
        "Energy KCal",
        "Protein KG",
        "Carbohydrate KG",
        "Sugar KG",
        "Fat KG",
        "Saturates KG",
        "Fibre KG Flag",
        "Sodium KG",
    ]
].copy()
pur_recs = pur_recs.merge(nut_subset, on="pur_id", how="left")
pur_recs = pur_recs.merge(panel_clusters, on="Panel Id", how="left")

# %%
# Split into clusters
cluster_mm_1 = pur_recs[pur_recs["clusters_mm"] == 0]
cluster_mm_2 = pur_recs[pur_recs["clusters_mm"] == 1]
cluster_mm_3 = pur_recs[pur_recs["clusters_mm"] == 2]
cluster_mm_4 = pur_recs[pur_recs["clusters_mm"] == 3]
cluster_mm_5 = pur_recs[pur_recs["clusters_mm"] == 4]
cluster_mm_6 = pur_recs[pur_recs["clusters_mm"] == 5]

# %%
# Split into clusters
cluster_pp_1 = pur_recs[pur_recs["clusters_pp"] == 0]
cluster_pp_2 = pur_recs[pur_recs["clusters_pp"] == 1]
cluster_pp_3 = pur_recs[pur_recs["clusters_pp"] == 2]
cluster_pp_4 = pur_recs[pur_recs["clusters_pp"] == 3]


# %%
# Get total nutritional volume per category per cluster
def total_nutrition_intake(cluster):
    c_total = cluster.groupby(by=["Attribute Code Description"]).sum()[
        [
            "gross_up_vol",
            "Energy KJ",
            "Energy KCal",
            "Protein KG",
            "Carbohydrate KG",
            "Sugar KG",
            "Fat KG",
            "Saturates KG",
            "Fibre KG Flag",
            "Sodium KG",
        ]
    ]
    return c_total.loc[:, c_total.columns != "gross_up_vol"].multiply(
        c_total["gross_up_vol"], axis="index"
    )


# %%
# Apply function to cluster subsets
c_1_total = total_nutrition_intake(cluster_pp_1)
c_2_total = total_nutrition_intake(cluster_pp_2)
c_3_total = total_nutrition_intake(cluster_pp_3)
c_4_total = total_nutrition_intake(cluster_pp_4)

# %%
# Apply function to cluster subsets
c_1_total = total_nutrition_intake(cluster_mm_1)
c_2_total = total_nutrition_intake(cluster_mm_2)
c_3_total = total_nutrition_intake(cluster_mm_3)
c_4_total = total_nutrition_intake(cluster_mm_4)
c_5_total = total_nutrition_intake(cluster_mm_5)
c_6_total = total_nutrition_intake(cluster_mm_6)

# %% [markdown]
# Looking at the top 10 categories for energy kcal total per cluster group

# %%
total_cats = total_nutrition_intake(pur_recs)

# %%
total_cats.head(1)

# %%
total_cats = pd.concat(
    [
        total_cats["Energy KJ"],
        c_1_total["Energy KJ"],
        c_2_total["Energy KJ"],
        c_3_total["Energy KJ"],
        c_4_total["Energy KJ"],
        c_5_total["Energy KJ"],
        c_6_total["Energy KJ"],
    ],
    axis=1,
)

# %%
total_cats.columns = ["total KJ", "c1 KJ", "c2 KJ", "c3 KJ", "c4 KJ", "c5 KJ", "c6 KJ"]

# %%
for col in total_cats.columns:
    total_cats[col] / total_cats[col].sum()

# %%
total_cats.sort_values(by="total KJ", ascending=False).head(15)

# %%
total_cats.sort_values(by="total KJ", ascending=False).head(20).loc[
    :, total_cats.columns != "total KJ"
].plot(kind="barh", figsize=(10, 20))
plt.title(
    "Percentage kcal purchased in a cluster per category (top 15 kcal purchased categories)",
    fontsize=14,
    pad=15,
)
plt.xlabel("Percentage purchased per cluster", fontsize=12)
plt.ylabel("Category", fontsize=12)
plt.savefig(f"{PROJECT_DIR}/outputs/figures/cluster-kcal-purchased per category")
plt.show()
