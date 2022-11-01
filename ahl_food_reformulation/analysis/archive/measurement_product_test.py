# Import libraries
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.utils import lookups as lps
from ahl_food_reformulation import PROJECT_DIR
import logging
import pandas as pd
import numpy as np
import seaborn as sns

logging.info("loading data")
# Reading in data
pur_recs = kantar.purchase_records()
nut_recs = kantar.nutrition()
prod_mast = kantar.product_master()
prod_codes = kantar.product_codes()
prod_vals = kantar.product_values()
prod_att = kantar.product_attribute()
val_fields = kantar.val_fields()
uom = kantar.uom()
conv_meas = lps.measure_table(kantar.product_measurement())

# Adding volume measurement
pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)

conv_meas.head(5)  # Look at a few examples

# Measurements to convert
measures = ["Units", "Litres", "Servings"]

# Convert selected measures and combine with existing kilos
pur_recs_full = lps.conv_kilos(pur_rec_vol, conv_meas, measures)

# A few examples
pur_recs_full.head(4)

# Check results
print(pur_recs_full["Reported Volume"].value_counts())

print("------")
# Compare to previous
print(pur_rec_vol["Reported Volume"].value_counts())

# ### Unique product table

logging.info("Slice by Kilos")
# Slicing by only kilos
pur_rec_kilos = pur_rec_vol[pur_rec_vol["Reported Volume"] == "Kilos"].copy()

logging.info("Getting unique product list")
# Applying function to get unique product list with selected per 100g nutritional info
unique_prods_nut = lps.products_per_100g(
    ["Energy KJ", "Protein KG"], pur_rec_kilos, nut_recs
)

# Checking a few examples
unique_prods_nut.head(5)

# Check for duplicates
unique_prods_nut["Product Code"].is_unique

### Analysis of products with missing volume weight (and reported volume as units)

# %% [markdown]
#
# About 20% of products have missing weights (no grams or litres)
#
# - [ ]  Analysis to understand what these products are and if they are over/underrepresented in certain categories
# - [ ]  Analysis to understand if dropping these products may create bias when describing different clusters

# %%
unique_prods_nut.head(2)

# %%
pur_recs_full.head(1)

# %%
product_list = lps.product_table(
    val_fields, prod_mast, uom, prod_codes, prod_vals, prod_att
)  # [['Product Code', 'Reported Volume']]

# %%
product_list.head(1)

# %%
prods_kilo_converted = pur_recs_full.copy()
prods_kilo_converted["converted"] = "yes"
prods_kilo_converted = prods_kilo_converted[["Product Code", "converted"]]

# %%
prods_kilo_converted.drop_duplicates(inplace=True)

# %%
product_list.head(3)

# %%
prods_kilo_converted.head(1)

# %%
product_list.shape

# %%
pur_rec_vol.columns

# %%
pur_rec_vol.drop_duplicates(subset=["Product Code"]).shape

# %%
pur_prod_list = pur_rec_vol.drop_duplicates(subset=["Product Code"])[
    ["Product Code", "Reported Volume"]
].copy()

# %%
pur_prod_list = pur_prod_list.merge(prods_kilo_converted, how="left", on="Product Code")

# %%
pur_prod_list.converted.fillna("no", inplace=True)

# %%
pur_prod_list.shape[0]

# %%
(
    pur_prod_list.groupby(["converted"])["Reported Volume"].value_counts()
    / (pur_prod_list.shape[0])
) * 100

# %%
pur_prod_list = pur_prod_list.merge(
    product_list[["Product Code", "RST 4 Extended", "RST 4 Market Sector"]],
    how="left",
    on="Product Code",
)

# %%
# pur_prod_list[(pur_prod_list['Reported Volume'] =='Litres')&(pur_prod_list['converted'] =='no')]['RST 4 Extended'].value_counts().head(20)

# %%
not_converted = pur_prod_list[pur_prod_list.converted == "no"]

# %%
rst_not_converted = (
    not_converted.groupby(["Reported Volume"])["RST 4 Market Sector"]
    .value_counts()
    .reset_index(name="count")
)

# %%
rst_not_converted["percent"] = (
    rst_not_converted["count"] / rst_not_converted["count"].sum()
) * 100

# %%
rst_not_converted.sort_values(by="percent", ascending=False)

# %%
num_prod_cat = (
    pur_prod_list.groupby(["RST 4 Market Sector", "converted"])["Product Code"]
    .size()
    .reset_index()
)

# %%
num_prod_cat = (
    num_prod_cat.pivot_table(
        values="Product Code",
        index="RST 4 Market Sector",
        columns="converted",
        aggfunc="first",
    )
    .fillna(0)
    .reset_index()
)

# %%
num_prod_cat["percent_missing"] = (
    num_prod_cat["no"] / (num_prod_cat["no"] + num_prod_cat["yes"])
) * 100

# %%
num_prod_cat

# %%
num_prod_cat.sort_values(by="percent_missing", ascending=False).set_index(
    "RST 4 Market Sector"
)["percent_missing"].sort_values().plot(kind="barh")

# %%
num_prod_cat.sort_values(by="percent_missing", ascending=False).head(10)

# %%
pur_prod_list.head(1)

# %%
pur_rec_vol.head(1)

# %%
pur_vol_conv = pur_rec_vol[
    [
        "Panel Id",
        "Product Code",
        "Reported Volume",
        "Gross Up Weight",
        "Volume",
        "Quantity",
    ]
].merge(
    pur_prod_list[["Product Code", "converted", "RST 4 Market Sector"]],
    how="left",
    on="Product Code",
)

# %%
(
    pur_vol_conv.groupby(["converted"])["Reported Volume"].value_counts()
    / (pur_vol_conv.shape[0])
) * 100

# %%
pur_vol_conv["RST 4 Market Sector"].value_counts().shape

# %%
pur_vol_conv.columns

# %%
percent_cat_kilos = (
    pur_vol_conv.groupby(["converted", "RST 4 Market Sector"])["Quantity"].sum()
    / pur_vol_conv.groupby(["RST 4 Market Sector"])["Quantity"].sum()
) * 100
percent_cat_kilos = percent_cat_kilos.reset_index()
percent_cat_non_kilos = percent_cat_kilos[percent_cat_kilos.converted == "no"].copy()
percent_cat_non_kilos.sort_values(by="Quantity", ascending=False)
percent_cat_non_kilos[percent_cat_non_kilos.Quantity > 10]["RST 4 Market Sector"]

# %%
pur_prod_list.columns

# %%
pur_vol_conv = pur_rec_vol[
    [
        "Panel Id",
        "Product Code",
        "Reported Volume",
        "Gross Up Weight",
        "Volume",
        "Quantity",
    ]
].merge(
    pur_prod_list[["Product Code", "converted", "RST 4 Extended"]],
    how="left",
    on="Product Code",
)
percent_cat_kilos = (
    pur_vol_conv.groupby(["converted", "RST 4 Extended"])["Quantity"].sum()
    / pur_vol_conv.groupby(["RST 4 Extended"])["Quantity"].sum()
) * 100
percent_cat_kilos = percent_cat_kilos.reset_index()
percent_cat_non_kilos = percent_cat_kilos[percent_cat_kilos.converted == "no"].copy()
percent_cat_non_kilos.sort_values(by="Quantity", ascending=False)
remove_list = percent_cat_non_kilos[percent_cat_non_kilos.Quantity > 10][
    "RST 4 Extended"
]

# %%
remove_list[remove_list == "Butter Slightly Sal Spreadable"]

# %% [markdown]
# ### Findings

# %%
100 - ((102680 / 128432) * 100)

# %% [markdown]
# - 20% of all products (in products master) are not in the purchase records
# - Of all purchased products:
#    - 78% of products have kilo weighting
#    - 22% can't be converted (16% litres, 6% units, less than 0.01% servings
# - Non-converted products:
#     - 36% alcohol
#     - 24% bakery
#     - 20% soft drinks
#     - 7% frozen
#     - 5% dairy
# - Percent of products missing
#     - 100% of alcohol
#     - 96% take-home soft drinks
#     - 85% chilled drinks
#     - 77% of bakery

# %%
## To Do
# Percent of category
# Percent of purchases
