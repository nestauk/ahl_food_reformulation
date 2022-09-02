# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Creates the dataset behind the visualisation

# %% [markdown]
# ## Preliminaries

# %%
# Load libraries
from ahl_food_reformulation import get_yaml_config, Path, PROJECT_DIR
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import math
import csv
from itertools import chain
import matplotlib.colors
import numpy as np
from scipy.special import comb
import json


# %%
# Load config file
mp_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/ahl_food_reformulation/config/base.yaml")
)

# Paths - inputs
path_purchase_records = str(PROJECT_DIR) + mp_config["PURCHASE_RECORDS"]
path_product_attribute_coding = str(PROJECT_DIR) + mp_config["PRODUCT_ATTRIBUTE_CODING"]
path_product_attribute_values = str(PROJECT_DIR) + mp_config["PRODUCT_ATTRIBUTE_VALUES"]
path_nutrition_data = str(PROJECT_DIR) + mp_config["NUTRITION_DATA"]
path_product_master = str(PROJECT_DIR) + mp_config["PRODUCT_MASTER"]
path_validation_field = str(PROJECT_DIR) + mp_config["VALIDATION_FIELD"]
path_uom = str(PROJECT_DIR) + mp_config["UOM"]
path_old_new_labels = str(PROJECT_DIR) + mp_config["OLD_NEW_LABELS"]
path_grams_products = str(PROJECT_DIR) + mp_config["GRAMS_PRODUCTS"]
path_panel_household_master = str(PROJECT_DIR) + mp_config["PANEL_HOUSEHOLD_MASTER"]

# Paths - outputs
path_data = str(PROJECT_DIR) + mp_config["DATA"]

# %%
# Parameters

# Product groups
BROAD_PRODUCT_GROUPING_LEVEL = 2828
MID_PRODUCT_GROUPING_LEVEL = 2829
NARROW_PRODUCT_GROUPING_LEVEL = 2827
VERY_NARROW_PRODUCT_GROUPING_LEVEL = 2907

# Month on which to focus (Oct 2021)
# Note: This month includes two 'purchase periods' as defined by Kantar
PURCHASE_DATE = "10/2021"
PURCHASE_PERIOD_A = 202110
PURCHASE_PERIOD_B = 202111

# The number of regulars to show in the inner circle
# (those goods that are most likely to be purchased
# on a single shopping trip)
NO_REGULARS = 20

# If more than 50% of products in a group exceed the sugar/sodium/saturates threshold
# then they are given a flag
MIN_PERCENT_PRODUCTS_FLAG = 50

# Size of the inner and outer circle in the data visualisation
RADIUS_INNER_CIRCLE = 10
RADIUS_OUTER_CIRCLE = 40

# Number of lines to show in the visualisation
# (the 100 pairs of products that were most frequently
# purchased together)
NO_LINKS_SHOW = 100

# In instances where the volume is in 'Units',
# we check for the percentage of occassions where 'gram' information
# is available. If it's available for over 90% of purchases then we use the gram information.
# Otherwise we say that the product group is missing nutritional information.
THRESHOLD_USING_GRAMS = 90

# Display options for dataframes
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500


# %%
# Thresholds for indicating that a product has high sugar, saturates or salt
# These are the lower bounds for the amber colour on the front-of-package labels
# They are from https://www.food.gov.uk/sites/default/files/media/document/fop-guidance_0.pdf
# See pages 19 and 20

dict_nutritional_thresholds = {
    "Sugar": {"grams": 5 / 100, "millilitres": 2.5 / 100},
    "Saturates": {"grams": 1.5 / 100, "millilitres": 0.75 / 100},
    "Sodium": {"grams": 0.3 / 100, "millilitres": 0.3 / 100},
}


# %% [markdown]
# ### Check location of households
# ### (to find that it's GB only)

# %%
# Load product master
df_phm = pd.read_csv(path_panel_household_master)  # , encoding="ISO-8859-1")


# %%
# Take a look
df_phm.head(1)

# %%
# Correct spelling mistake in column name
df_phm.rename(columns={"Postocde District": "Postcode District"}, inplace=True)


# %%
# Check for Edinburgh (EH)
df_phm[df_phm["Postcode District"].str.contains("EH")].head(1)

# %%
# Check for Belfast (BT)
df_phm[df_phm["Postcode District"].str.contains("BT")].head(1)


# %%
# Check for Cardiff (CF)
df_phm[df_phm["Postcode District"].str.contains("CF")].head(1)

# %%
# Check most popular postcodes
df_phm["Postcode District"].value_counts()

# %% [markdown]
# # Combine all information about purchases and create a dataframe containing one row per purchase

# %% [markdown]
# ## Load purchase data

# %%
# Load purchase data (in chunks)
iter_csv_1 = pd.read_csv(path_purchase_records, iterator=True, chunksize=1000000)
df_pr = pd.concat(
    [chunk[chunk["Purchase Date"].str.contains(PURCHASE_DATE)] for chunk in iter_csv_1]
)

# %%
# Take a look
# Note: Multiple purchases of the same product are in one row (see Quantity column)
df_pr.head(1)

# %%
# Remove the couple of rows which report a Quantity of 0 or a Volume of 0
# (to prevent division by zero later on)

print(
    "No. of purchase records with Quantity = 0 or Volume = 0: "
    + str(df_pr[df_pr["Quantity"] == 0].shape[0] + df_pr[df_pr["Volume"] == 0].shape[0])
)
df_pr.drop(df_pr[df_pr["Quantity"] == 0].index, inplace=True)
df_pr.drop(df_pr[df_pr["Volume"] == 0].index, inplace=True)


# %% [markdown]
# ## Add in nutritional information

# %% [markdown]
# ### Load nutritional information

# %%
# Load nutrition data (in chunks)
iter_csv_2 = pd.read_csv(path_nutrition_data, iterator=True, chunksize=1000000)
df_nu = pd.concat(
    [
        chunk[
            (chunk["Purchase Period"] == PURCHASE_PERIOD_A)
            | (chunk["Purchase Period"] == PURCHASE_PERIOD_B)
        ]
        for chunk in iter_csv_2
    ]
)


# %%
# Take a look
df_nu.head(1)


# %%
# Rename a couple of columns to make merging easier
# and correct 'Fibre KG' and 'Fibre KG Flag' which have been swapped
df_nu.rename(
    columns={
        "Purchase Number": "PurchaseId",
        "Purchase Period": "Period",
        "Fibre KG Flag": "Fibre KG",
        "Fibre KG": "Fibre KG Flag",
    },
    inplace=True,
)


# %% [markdown]
# ### Add nutritional information to purchase data

# %%
# Add nutrition data into purchase records
df_pr = pd.merge(df_pr, df_nu, on=["PurchaseId", "Period"], how="left")


# %%
# Take a look
df_pr.head(1)

# %% [markdown]
# ### Scale nutritional values by Quantity

# %%
# Scale each nutritional value by quantity
# (The original nutritional information is inclusive of Quantity)

df_pr["Energy KJ"] = df_pr["Energy KJ"] / df_pr["Quantity"]
df_pr["Energy KCal"] = df_pr["Energy KCal"] / df_pr["Quantity"]
df_pr["Protein KG"] = df_pr["Protein KG"] / df_pr["Quantity"]
df_pr["Carbohydrate KG"] = df_pr["Carbohydrate KG"] / df_pr["Quantity"]
df_pr["Sugar KG"] = df_pr["Sugar KG"] / df_pr["Quantity"]
df_pr["Fat KG"] = df_pr["Fat KG"] / df_pr["Quantity"]
df_pr["Saturates KG"] = df_pr["Saturates KG"] / df_pr["Quantity"]
df_pr["Fibre KG"] = df_pr["Fibre KG"] / df_pr["Quantity"]
df_pr["Sodium KG"] = df_pr["Sodium KG"] / df_pr["Quantity"]


# %% [markdown]
# ## Expand purchase data (so that one row = one item)

# %%
# Expand out df_pr so that every row is just one purchase of an item
# (i.e. multiple purchases of the same item have been expanded)
df_pr_ex = df_pr.reindex(df_pr.index.repeat(df_pr.Quantity))


# %%
# Create a 'Volume per product' variable
df_pr_ex["Volume per product"] = df_pr_ex["Volume"] / df_pr_ex["Quantity"]


# %%
# Only keep relevant columns
# (particularly important to get rid of the Volume, Quantity, Weight & Spend columns
# they're no longer accurate)
df_pr_ex = df_pr_ex[
    [
        "Panel Id",
        "Product Code",
        "Store Code",
        "Promo Code",
        "PurchaseId",
        "Purchase Date",
        "Period",
        "Volume per product",
        "Gross Up Weight",
        "Energy KJ",
        "Energy KCal",
        "Protein KG",
        "Carbohydrate KG",
        "Sugar KG",
        "Fat KG",
        "Saturates KG",
        "Fibre KG",
        "Sodium KG",
    ]
]


# %%
# Number of purchases
print("Number of purchases: " + str(len(df_pr_ex)))

# %% [markdown]
# ## Gram and unit data for some products

# %% [markdown]
# ### Load the file that contains gram and unit data for some products

# %%
# Not available for all products (so still need to load the original unit data below)

# %%
# Load file that contains the number of grams in each product
df_grams_products = pd.read_csv(path_grams_products, sep=",", header=None)
df_grams_products.rename(
    columns={0: "Product Code", 1: "UnitsAndGrams", 2: "Value"}, inplace=True
)

# %%
# Take a look
df_grams_products.head(1)


# %%
# Create dummy variables for units and gram values
df_grams_products["Units"] = np.where(
    df_grams_products["UnitsAndGrams"] == "UNITS", df_grams_products["Value"], None
)

df_grams_products["Grams"] = np.where(
    df_grams_products["UnitsAndGrams"] == "GRAMS", df_grams_products["Value"], None
)


# %%
# Collapse df_grams_products so that each row corresponds to one product
df_grouped_grams_products = df_grams_products.groupby(
    ["Product Code"], as_index=False
).agg({"Grams": "first", "Units": "first"})


# %%
# Take a look
df_grouped_grams_products.head(1)


# %%
# CORRECT A MISTAKE IN THE FILE
# There are several thousand products that reportedly are available in 'Grams' but the value is '1'
# These should be set to none

# Extent of problem
print(
    "No of mistakes: "
    + str(df_grouped_grams_products[df_grouped_grams_products["Grams"] == 1].shape[0])
)


# %%
# Correction (set grams to None in these instances)
df_grouped_grams_products[df_grouped_grams_products["Grams"] == 1] = None


# %%
# Check it has been fixed
print(
    "No of mistakes: "
    + str(df_grouped_grams_products[df_grouped_grams_products["Grams"] == 1].shape[0])
)


# %% [markdown]
# ### Merge the unit and gram information into the purchases data

# %%
# Add in unit and gram information
df_pr_ex = pd.merge(df_pr_ex, df_grouped_grams_products, on="Product Code", how="left")


# %%
# Check number of purchases hasn't changed
print("Number of purchases: " + str(len(df_pr_ex)))


# %% [markdown]
# ## Volume units for products

# %% [markdown]
# ### Load the relevant files (product master, validation fields, units of measurement)

# %%
# Load product master
df_pm = pd.read_csv(path_product_master, encoding="ISO-8859-1")


# %%
# Take a look
df_pm.head(1)

# %%
# Drop Nones from the Validation Field (to help with merging)
df_pm.dropna(subset=["Validation Field"], inplace=True)


# %%
# Change type of Validation Field to match with df_vf
df_pm["Validation Field"] = df_pm["Validation Field"].astype(int)


# %%
# Load validation fields
df_vf = pd.read_csv(path_validation_field)


# %%
# Rename VF in df_vf to match its name in df_pm
df_vf.rename(columns={"VF": "Validation Field"}, inplace=True)


# %%
# Change type of validation field to match df_pm
df_vf["Validation Field"] = df_vf["Validation Field"].astype(int)


# %%
# Take a look
df_vf.head(1)


# %%
# Load units
df_uom = pd.read_csv(path_uom)


# %%
# Rename uom_id in df_uom to match its name in df_vf
df_uom.rename(columns={"uom_id": "UOM"}, inplace=True)


# %%
# Take a look
df_uom.head(1)


# %% [markdown]
# ### Merge unit of measurement into validation field, and then merge validation field into product master

# %%
# Add column to df_vf that gives the actual units (e.g. litres)
df_vf = pd.merge(df_vf, df_uom, on="UOM", how="left")


# %%
# Take a look
df_vf.head(1)


# %%
# Merge df_vf into df_pm to give the actual units (e.g. litres)
df_pm = pd.merge(df_pm, df_vf, on="Validation Field", how="left")


# %%
# Drop duplicates
df_pm.drop_duplicates(
    subset=["Product Code"], keep="first", inplace=True, ignore_index=True
)


# %%
# Take a look
df_pm.head(1)


# %% [markdown]
# ### Merge product master into purchase data

# %%
# Merge df_pm into df_pr_ex to give the actual units (e.g. litres)
df_pr_ex = pd.merge(df_pr_ex, df_pm, on=["Product Code"], how="left")


# %%
# Check number of purchases hasn't changed
print("Number of purchases: " + str(len(df_pr_ex)))


# %% [markdown]
# ## Calculate grams per product and millilitres per product (where available)

# %%
# Function to return grams per product (where available)
def grams_per_product(x):

    # For products originally expressed in kilos
    if x["reported_volume"] == "Kilos":
        grams_pp = 1000 * x["Volume per product"]

    # For all other products (expressed in litres, units, servings or if it's NA)
    # (Note: this value will be NA if grams are not available)
    else:
        grams_pp = x["Grams"]

    return grams_pp


# %%
# Calculate grams per product (not available for all products) (a little slow)
df_pr_ex["Grams per product"] = df_pr_ex.apply(lambda x: grams_per_product(x), axis=1)


# %%
# Calculate percentage of products where grams are missing
# (typically due to product being expressed in litres)
percent_missing = df_pr_ex["Grams per product"].isnull().sum() * 100 / len(df_pr_ex)
print(
    "Percent of purchases where grams per product is missing: " + str(percent_missing)
)


# %%
# Function to return millilitres per product (where available)
def millilitres_per_product(x):

    # For products originally expressed in Litres
    if x["reported_volume"] == "Litres":
        millil_pp = 1000 * x["Volume per product"]

    # For all other products
    # (we don't any other source)
    else:
        millil_pp = None

    return millil_pp


# %%
# Calculate millilitres per product (not available for all products) (a little slow)
df_pr_ex["Millilitres per product"] = df_pr_ex.apply(
    lambda x: millilitres_per_product(x), axis=1
)


# %%
# Start to calculate percentage of products where millilitres and grams are missing
df_pr_ex["Missing grams and millilitres"] = df_pr_ex.apply(
    lambda x: 1
    if pd.isna(x["Grams per product"]) and pd.isna(x["Millilitres per product"])
    else 0,
    axis=1,
)


# %%
# Percentage of products where grams and millilitres are missing
percent_missing = 100 * (
    df_pr_ex[df_pr_ex["Missing grams and millilitres"] == 1].shape[0] / len(df_pr_ex)
)

print(
    "Percent of purchases where grams & millitres are missing: " + str(percent_missing)
)


# %% [markdown]
# ## Find the broader categories for products

# %% [markdown]
# ### Load product attributes & values

# %%
# Load product attribute coding
df_pac = pd.read_csv(path_product_attribute_coding)


# %%
# Only keep attribute numbers for PRODUCT GROUPING LEVELS
df_pac = df_pac[
    (df_pac["Attribute Number"] == BROAD_PRODUCT_GROUPING_LEVEL)
    | (df_pac["Attribute Number"] == MID_PRODUCT_GROUPING_LEVEL)
    | (df_pac["Attribute Number"] == NARROW_PRODUCT_GROUPING_LEVEL)
    | (df_pac["Attribute Number"] == VERY_NARROW_PRODUCT_GROUPING_LEVEL)
]


# %%
# Take a look
df_pac.head(1)

# %%
# Load product attribute values
df_pav = pd.read_csv(path_product_attribute_values, encoding="ISO-8859-1")


# %%
# Take a look
df_pav.head(1)


# %% [markdown]
# ### Merge product values into attributes

# %%
# Add Attribute Value Descriptions (in df_pav) into df_pac
df_pac = pd.merge(
    df_pac,
    df_pav[["Attribute Value", "Attribute Value Description"]],
    on="Attribute Value",
    how="left",
)


# %%
# Take a look
df_pac.head(1)

# %%
# There are a very small number (108) of products whose 'very narrow groups' are missing
# their attribute values and descriptions
# Replace their descriptions (which are set to NaNs) with 'Missing'
# (this prevents the 'groupby' below from throwing an error)
df_pac["Attribute Value Description"].fillna("Missing", inplace=True)


# %%
# Group by Product Code to create the full name of the product
# (combining the Broad, Mid, Narrow & Very Narrow labels)
df_pac = df_pac.groupby(["Product Code"], as_index=False).agg(
    {"Attribute Value Description": " - ".join, "Attribute Value": " - ".join}
)


# %%
# The line above mixes up the order of the groups
# and so we need to rearrange the order so that
# their descriptions and values go from broadest to narrowest
df_pac["Attribute Value Description"] = df_pac["Attribute Value Description"].apply(
    lambda x: x.split("-")[1].strip()
    + " - "
    + x.split("-")[0].strip()
    + " - "
    + x.split("-")[2].strip()
    + " - "
    + x.split("-")[3].strip()
)

df_pac["Attribute Value"] = df_pac["Attribute Value"].apply(
    lambda x: x.split("-")[1].strip()
    + " - "
    + x.split("-")[0].strip()
    + " - "
    + x.split("-")[2].strip()
    + " - "
    + x.split("-")[3].strip()
)


# %%
# Take a look
df_pac.head(1)

# %% [markdown]
# ### Add product values to purchase data

# %%
# Add in product information
df_pr_ex = pd.merge(
    df_pr_ex,
    df_pac[["Product Code", "Attribute Value", "Attribute Value Description"]],
    on="Product Code",
    how="left",
)


# %%
# Check that the number of purchases hasn't changed
print("Number of purchases: " + str(len(df_pr_ex)))


# %% [markdown]
# ## Calculate nutritional ratios (e.g. sugar per gram) for each product group

# %% [markdown]
# ### Calculate sugar, saturates, and sodium per gram, per millilitre for each purchase

# %%
# Calculate sugar per gram and per millilitre

df_pr_ex["Sugar per gram"] = 1000 * df_pr_ex["Sugar KG"] / df_pr_ex["Grams per product"]
df_pr_ex["Sugar per millilitre"] = (
    1000 * df_pr_ex["Sugar KG"] / df_pr_ex["Millilitres per product"]
)


# %%
# Calculate saturates per gram and per millilitre

df_pr_ex["Saturates per gram"] = (
    1000 * df_pr_ex["Saturates KG"] / df_pr_ex["Grams per product"]
)
df_pr_ex["Saturates per millilitre"] = (
    1000 * df_pr_ex["Saturates KG"] / df_pr_ex["Millilitres per product"]
)


# %%
# Calculate sodium per gram and per millilitre

df_pr_ex["Sodium per gram"] = (
    1000 * df_pr_ex["Sodium KG"] / df_pr_ex["Grams per product"]
)
df_pr_ex["Sodium per millilitre"] = (
    1000 * df_pr_ex["Sodium KG"] / df_pr_ex["Millilitres per product"]
)


# %% [markdown]
# ### Drop purchases where the nutritional information appears incorrect

# %%
# We first exclude any products that have in excess of 1 gram of sugar/saturates/sodium
# per gram of product which implies a mistake in the recording

df_pr_ex_no_mistakes = df_pr_ex[
    # If product is in grams
    (
        (df_pr_ex["Grams per product"].notnull() == True)
        & (df_pr_ex["Sugar per gram"] <= 1)
        & (df_pr_ex["Saturates per gram"] <= 1)
        & (df_pr_ex["Sodium per gram"] <= 1)
    )
    |
    # If product is in millilitres
    (
        (df_pr_ex["Millilitres per product"].notnull() == True)
        & (df_pr_ex["Sugar per millilitre"] <= 1)
        & (df_pr_ex["Saturates per millilitre"] <= 1)
        & (df_pr_ex["Sodium per millilitre"] <= 1)
    )
    |
    # If the product is not available in grams or millilitres (keep these)
    (
        (df_pr_ex["Grams per product"].notnull() == False)
        & (df_pr_ex["Millilitres per product"].notnull() == False)
    )
]


# %%
# Percentages of purchases removed

percent_removed = 100 * (len(df_pr_ex) - len(df_pr_ex_no_mistakes)) / len(df_pr_ex)
print("Percent of purchases removed: " + str(percent_removed))


# %%
# Group purchases into product groups (using medians)
# We're considering all products that were purchased at least one in the month of October
# We're using medians to minimise the impact of potential errors in nutritional information

df_products = df_pr_ex_no_mistakes.groupby(["Product Code"])[
    [
        "Sodium per gram",
        "Sodium per millilitre",
        "Sugar per gram",
        "Sugar per millilitre",
        "Saturates per gram",
        "Saturates per millilitre",
        "Gross Up Weight",
        "Product Code",
        "Product Long Description",
        "Attribute Value",
        "Attribute Value Description",
    ]
].agg(
    {
        "Sodium per gram": "median",
        "Sodium per millilitre": "median",
        "Sugar per gram": "median",
        "Sugar per millilitre": "median",
        "Saturates per gram": "median",
        "Saturates per millilitre": "median",
        "Gross Up Weight": "sum",
        "Attribute Value": "first",
        "Product Code": "count",
        "Attribute Value Description": "first",
        "Product Long Description": "first",
    }
)


# %%
# Rename the 'product code' variables as it's now capturing the count of each product
df_products.rename(columns={"Product Code": "Product Count"}, inplace=True)


# %% [markdown]
# # Extract key information for the most frequently purchased product groups

# %% [markdown]
# ## Identify frequently purchased product groups & create a dictionary of these products

# %%
# Group purchases by Attribute Value (narrowest grouping available)
df_product_groups = df_pr_ex.groupby(["Attribute Value"])[
    [
        "Attribute Value Description",
        "Gross Up Weight",
    ]
].agg({"Attribute Value Description": "first", "Gross Up Weight": "sum"})

# %%
# Most frequently purchased product groups
most_frequently_purchased = (
    df_product_groups.sort_values(by="Gross Up Weight", ascending=False)
    .head(100)
    .index.tolist()
)


# %%
# Percentage of all purchases (weighted) that feature within the 100 most frequently purchased
# product groups

top_100_dominance = (
    100
    * df_product_groups.sort_values(by="Gross Up Weight", ascending=False)
    .head(100)["Gross Up Weight"]
    .sum()
    / df_product_groups["Gross Up Weight"].sum()
)
print(
    "Percentage of all purchases (weighted) of items that are in the 100 most frequently purchased product groups: "
    + str(top_100_dominance)
)

# %%
# Form a dictionary to hold information about the most popular products
# (This will house all the information for the visualisation)
dict_products = {value: {} for value in most_frequently_purchased}


# %% [markdown]
# ## Add in labels

# %%
# The label is made up of the concatenated attribute value descriptions (broad to narrow)
for key, value in dict_products.items():
    value["label"] = df_pac[df_pac["Attribute Value"] == key][
        "Attribute Value Description"
    ].values[0]


# %% [markdown]
# ## Add in the total purchases for each product group

# %%
# The total number of items (weighted) purchased across all products in this group
for key, value in dict_products.items():
    value["count"] = df_product_groups.loc[key]["Gross Up Weight"]


# %% [markdown]
# # Find shopping trip regulars

# %%
# Group purchases by Panel Id, Purchase Date and Store Code
# to give a dataframe of shopping trips (df_st)
# Note: Separate trips to the same store on the same day by the same household cannot be separately identified
# (Quite slow)
df_st = df_pr_ex.groupby(["Panel Id", "Purchase Date", "Store Code"])[
    ["Attribute Value", "Gross Up Weight"]
].agg({"Attribute Value": lambda x: list(x), "Gross Up Weight": lambda x: list(x)})


# %%
# Take a look
df_st.head(1)

# %% [markdown]
# ## Basic exploration of shopping trips

# %%
# Average and median number of items purchased per shopping trip
# (multiple purchases of the same product are counted separately)

ave_items = np.mean([len(value) for value in df_st["Attribute Value"].tolist()])
print("Average number of items purchased per shopping trips: " + str(ave_items))

med_items = np.median([len(value) for value in df_st["Attribute Value"].tolist()])
print("Median number of items purchased per shopping trips: " + str(med_items))


# %% [markdown]
# ## Calculate weighted shopping trip propensities

# %%
# Calculate the average weight for each shopping trip
df_st["Average Gross Up Weight"] = df_st["Gross Up Weight"].apply(lambda x: np.mean(x))


# %%
# Find the weighted proportion of shopping trips where the item was purchased (at least once)

total_weight = df_st["Average Gross Up Weight"].sum()
average_weights = df_st["Average Gross Up Weight"].tolist()
shopping_lists = df_st["Attribute Value"].tolist()

# Loop over most frequently purchased product groups
for key, value in dict_products.items():

    # Weighted number of shopping trips in which item was purchased
    one_product_sts = sum(
        [
            average_weights[index]
            for index, value in enumerate(shopping_lists)
            if key in value
        ]
    )

    # Weighted proportion of shopping trips where items was purchased
    value["shopping_trip_propensity"] = 100 * (one_product_sts / total_weight)


# %% [markdown]
# ## Calculate shopping trip propensities and find 'regulars'

# %%
# Sort items by which are most likely to be purchased in a single shopping trip (weighted)
sorted_propensity = sorted(
    dict_products.values(), key=lambda x: x["shopping_trip_propensity"], reverse=True
)


# %%
# Identify 'regulars' (those 20 items that are most frequently purchased in a given shopping trip)
regulars = [
    key
    for key, value in dict_products.items()
    if value in sorted_propensity[0:NO_REGULARS]
]


# %% [markdown]
# # Find the most frequently co-purchased items

# %% [markdown]
# ## Form a dict of edges (containing pairs of products that were purchased on the same trip)

# %%
# Create a dictionary to hold the edges
dict_edges = {}
progress = 0

# Loop over shopping trips
for index, row in df_st.iterrows():

    # One trip
    items_one_trip = row["Attribute Value"]
    weights_one_trip = row["Gross Up Weight"]
    unique_items_one_trip = list(set(row["Attribute Value"]))

    # Trips where are least 2 different items were purchased
    if len(items_one_trip) > 1:

        # Loop over first item in pair
        for index_first, first_item in enumerate(unique_items_one_trip[:-1]):

            # Loop over the second item in pair
            for index_second, second_item in enumerate(
                unique_items_one_trip[(index_first + 1) :]
            ):

                # If both items are in the 'most-purchased list' (i.e. dict_products)
                if first_item in dict_products and second_item in dict_products:

                    # Alphabetise combo (so the same combo is recorded consistently)
                    one_combo = (first_item, second_item)
                    if second_item < first_item:
                        one_combo = (second_item, first_item)

                    # Create space for combo
                    if one_combo not in dict_edges:
                        dict_edges[one_combo] = {"name": one_combo, "purchased_both": 0}

                    # Calculate the total weight for both items in the combo
                    # (that item may have been purchased multiple times in the same trip and so we gather
                    # all the weights attached to those purchases from the shopping trip)
                    weight = sum(
                        [
                            weight_one_item
                            for index_weight, weight_one_item in enumerate(
                                weights_one_trip
                            )
                            if items_one_trip[index_weight] == first_item
                        ]
                    ) + sum(
                        [
                            weight_one_item
                            for index_weight, weight_one_item in enumerate(
                                weights_one_trip
                            )
                            if items_one_trip[index_weight] == second_item
                        ]
                    )

                    # Add to dictionary
                    dict_edges[one_combo]["purchased_both"] += weight

    # Progress
    progress += 1
    if progress % 20000 == 0:
        print(progress)


# %% [markdown]
# ## Scale the 'joint purchase weight' for a pair by the total weight attached to all individual purchases of the two products

# %%
# The result is a type of weighted probability that both products are purchased when at least one is purchased
# (This is 'prob_co_purchase').

# Loop over pairs
for one_combo, value in dict_edges.items():

    # Total weight where either was purchased
    value["purchased_either"] = (
        df_product_groups.loc[one_combo[0]]["Gross Up Weight"]
        + df_product_groups.loc[one_combo[1]]["Gross Up Weight"]
    )

    # Scaled weight where both were purchased
    value["prob_co_purchase"] = (
        100 * value["purchased_both"] / value["purchased_either"]
    )


# %% [markdown]
# ## Find the most common pairs of products (amongst the most frequently purchased products)

# %%
# Sort the pairs with most likely to be purchased together at the top
sorted_edges = sorted(
    dict_edges.values(), key=lambda x: x["prob_co_purchase"], reverse=True
)


# %%
# Identify the most frequent co-purchases (weighted)
frequent_copurchases = [
    value for index, value in enumerate(sorted_edges) if index < NO_LINKS_SHOW
]

print("Number of top edges: " + str(len(frequent_copurchases)))


# %% [markdown]
# # Replace labels

# %% [markdown]
# ## Load new labels for product groups (that are easier to understand)

# %%
# Load new data
df_old_new_labels = pd.read_csv(path_old_new_labels)


# %%
# Change index
df_old_new_labels.index = df_old_new_labels["code"]


# %%
# Add labels to dictionary for viz
for key, value in dict_products.items():
    value["label_show"] = df_old_new_labels.loc[key]["new_label"]
    value["broad_label"] = df_old_new_labels.loc[key]["broad_label"]
    value["index"] = str(df_old_new_labels.loc[key]["index"])


# %% [markdown]
# ## Create a dictionary of broad labels & their counts

# %%
# List of unique broad labels
broad_labels = list(df_old_new_labels["broad_label"].unique())


# %%
# Create a dictionary of the broad labels and the number of product groups inside each
# (this is to assist with the visualisation)

dict_broad_labels = {}
for one_code, value in dict_products.items():

    # Broad label for product group
    broad_label = value["broad_label"]

    # Exclude 'regular' product groups (which sit in the inner circle)
    if one_code not in regulars:

        # New entries
        if broad_label not in dict_broad_labels:
            dict_broad_labels[broad_label] = {
                "broad_label": broad_label,
                "order": broad_labels.index(broad_label),
                "count": 0,
            }

        # Add to count
        dict_broad_labels[broad_label]["count"] += 1


# %%
# Sorted
broad_labels_sorted = sorted(dict_broad_labels.values(), key=lambda x: x["order"])


# %% [markdown]
# # Add in the x and y locations of products

# %%
# Running count variables
count_regular = 0
count_non_regular = 0

# Loop over each broad category of products
for index_broad, one_broad_label in enumerate(reversed(broad_labels)):

    # Extract all the product groups in this category
    products = df_old_new_labels[df_old_new_labels["broad_label"] == one_broad_label][
        "code"
    ].tolist()

    ## Set the x and y positions, as well as the angle and radius, of the points around the chart
    ## for each product group

    # Regular product groups
    one_set_regulars = [value for value in products if value in regulars]
    # Loop over regular product groups
    for index_regular, one_regular in enumerate(one_set_regulars):
        angle_regular = 360 * (count_regular / len(regulars)) + 90
        count_regular += 1
        dict_products[one_regular]["angle"] = angle_regular
        dict_products[one_regular]["x_pos"] = 50 + (
            RADIUS_INNER_CIRCLE * math.cos(angle_regular * math.pi / 180)
        )
        dict_products[one_regular]["y_pos"] = 50 + (
            RADIUS_INNER_CIRCLE * math.sin(angle_regular * math.pi / 180)
        )
        dict_products[one_regular]["radius"] = RADIUS_INNER_CIRCLE

    # Non-regulars
    one_set_non_regulars = [value for value in products if value not in regulars]
    # Angle of non-regulars
    for index_non_regular, one_non_regular in enumerate(one_set_non_regulars):
        angle_non_regular = (
            360 * (count_non_regular / (len(dict_products) - len(regulars))) + 90
        )
        count_non_regular += 1
        dict_products[one_non_regular]["angle"] = angle_non_regular
        dict_products[one_non_regular]["x_pos"] = 50 + (
            RADIUS_OUTER_CIRCLE * math.cos(angle_non_regular * math.pi / 180)
        )
        dict_products[one_non_regular]["y_pos"] = 50 + (
            RADIUS_OUTER_CIRCLE * math.sin(angle_non_regular * math.pi / 180)
        )
        dict_products[one_non_regular]["radius"] = RADIUS_OUTER_CIRCLE


# %% [markdown]
# # Add in nutritional flags

# %%
# A function that calculates the percentage of products in a product group whose
# sugar/saturates/sodium is above a partciular threshold from dict_nutritional_thresholds
def percentage_above_threshold(
    product, df_products, nutritional_factor, units, dict_nutritional_thresholds
):

    # Establish units
    if units == "Kilos":

        # Relevant threshold (i.e. sugar/saturates/sodium)
        threshold = dict_nutritional_thresholds[nutritional_factor]["grams"]

        # Appropriate column
        nutritional_column = nutritional_factor + " per gram"

    elif units == "Litres":

        # Relevant threshold (i.e. sugar/saturates/sodium)
        threshold = dict_nutritional_thresholds[nutritional_factor]["millilitres"]

        # Appropriate column
        nutritional_column = nutritional_factor + " per millilitre"

    # Total number of products in that product group (excluding products with no nutritional info)
    total_products = df_products[
        (df_products["Attribute Value"] == product)
        & (df_products[nutritional_column].notnull())
    ].shape[0]

    # Total number of products (still excluding NaNs) whose sugar/saturates/sodium content is above the threshold
    no_above_threshold = df_products[
        (df_products["Attribute Value"] == product)
        & (df_products[nutritional_column] > threshold)
    ].shape[0]

    # Percentage of products above threshold
    perc_products_over_threshold = 100 * no_above_threshold / total_products

    return total_products, perc_products_over_threshold


# %%
# Add in flags that indicate high sugar/saturates/sodium (a little slow)

progress = 0

# Loop over product groups in data viz
for key, value in dict_products.items():

    # Initialize flags
    value["sugar_flag"] = False
    value["sodium_flag"] = False
    value["saturates_flag"] = False

    # Which type of reported volume (e.g. litres/units/kgs) is the most common for this product
    count_units = df_pr_ex[df_pr_ex["Attribute Value"] == key][
        "reported_volume"
    ].value_counts()
    most_common_unit = count_units.to_frame().index[0]

    # If the most common metric is 'Units', check whether gram information is
    # available for many of the purchases of the products in this group
    if most_common_unit == "Units":

        # Percentage of purchases where 'sugar per gram' information was available
        perc_available_grams = (
            (
                df_pr_ex[
                    (df_pr_ex["Attribute Value"] == key)
                    & (df_pr_ex["Sugar per gram"].notnull())
                ].shape[0]
            )
            / df_pr_ex[(df_pr_ex["Attribute Value"] == key)].shape[0]
        ) * 100

        # If the grams for this product was frequently available
        if perc_available_grams > THRESHOLD_USING_GRAMS:
            # Use grams (kilos)
            most_common_unit = "Kilos"
        else:
            # Report the product group as missing nutritional information
            value["sugar_flag"] = "Missing"
            value["sodium_flag"] = "Missing"
            value["saturates_flag"] = "Missing"

    # If the most products in the group are expressed in Kilos or Litres
    if most_common_unit != "Units":

        # Percentage of products in each category that are above the relevant thresholds
        # for sugar, saturates and sodium
        total_products, perc_above_sugar_threshold = percentage_above_threshold(
            key, df_products, "Sugar", most_common_unit, dict_nutritional_thresholds
        )
        total_products, perc_above_saturates_threshold = percentage_above_threshold(
            key, df_products, "Saturates", most_common_unit, dict_nutritional_thresholds
        )
        total_products, perc_above_sodium_threshold = percentage_above_threshold(
            key, df_products, "Sodium", most_common_unit, dict_nutritional_thresholds
        )

        # Add flags
        if perc_above_sugar_threshold > MIN_PERCENT_PRODUCTS_FLAG:
            value["sugar_flag"] = True
        if perc_above_saturates_threshold > MIN_PERCENT_PRODUCTS_FLAG:
            value["saturates_flag"] = True
        if perc_above_sodium_threshold > MIN_PERCENT_PRODUCTS_FLAG:
            value["sodium_flag"] = True

    progress += 1
    if progress % 10 == 0:
        print(progress)


# %% [markdown]
# # Add in the lines between points

# %%
# A space to store lines between products
paths = []

# Loop over frequent copurchases
for one_link in frequent_copurchases:

    # One product
    start = one_link["name"][0]

    # Another product
    end = one_link["name"][1]

    # Weighted type of probability that both are purchased together
    weight = one_link["prob_co_purchase"]

    # Broader groups for both products
    broad_group_start = dict_products[start]["broad_label"]
    broad_group_end = dict_products[end]["broad_label"]

    # Determine the start and end of the path
    if dict_products[start]["angle"] < dict_products[end]["angle"]:

        paths.append(
            [
                {
                    "x": dict_products[start]["x_pos"],
                    "y": dict_products[start]["y_pos"],
                    "weight": weight,
                    "code": start,
                    "broad_group_start": broad_group_start,
                },
                {
                    "x": dict_products[end]["x_pos"],
                    "y": dict_products[end]["y_pos"],
                    "weight": weight,
                    "code": end,
                    "broad_group_end": broad_group_end,
                },
            ]
        )
    else:
        paths.append(
            [
                {
                    "x": dict_products[end]["x_pos"],
                    "y": dict_products[end]["y_pos"],
                    "weight": weight,
                    "code": end,
                    "broad_group_start": broad_group_start,
                },
                {
                    "x": dict_products[start]["x_pos"],
                    "y": dict_products[start]["y_pos"],
                    "weight": weight,
                    "code": start,
                    "broad_group_end": broad_group_end,
                },
            ]
        )


# %% [markdown]
# # Save

# %%
# Save
with open(path_data, "w") as f:
    json.dump(
        {
            "nodes": [value for index, value in dict_products.items()],
            "edges": paths,
            "broad_labels": broad_labels_sorted,
        },
        f,
    )
