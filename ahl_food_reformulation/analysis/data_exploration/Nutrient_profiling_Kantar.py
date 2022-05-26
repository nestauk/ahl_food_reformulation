# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Overview
#
# In this notebook you can find some utility functions for calculating "healthiness" according to 2004-2005 Nutrient Profiling System (NPS).

# %%
import boto3
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
from ahl_food_reformulation import PROJECT_DIR


# %% [markdown]
# ## 1. Define utility functions

# %%
# Utility functions to assign NPS scores


def assign_nps_energy_score(df, column_name):
    """Calculates NPS energy density score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350, 5000]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(df[column_name], bins=thresholds, labels=scores, right=True)
    return binned_cats


def assign_nps_satf_score(df, column_name):
    """Calculates NPS saturated fats score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_nps_sugars_score(df, column_name):
    """Calculates NPS sugars score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45, 100]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_nps_sodium_score(df, column_name):
    """Calculates NPS sodium score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 90, 180, 270, 360, 450, 540, 630, 720, 810, 900, 2400]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_nps_protein_score(df, column_name):
    """Calculates NPS proteins score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 1.6, 3.2, 4.8, 6.4, 8, 42]
    scores = [0, 1, 2, 3, 4, 5]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_nps_fiber_score(df, column_name):
    """Use 2004-2005 NPS to assign C points for fiber (AOAC)."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [
        0,
        0.9,
        1.9,
        2.8,
        3.7,
        4.7,
        24,
    ]  # NSP thresholds [0, 0.7, 1.4, 2.1, 2.8, 3.5]
    scores = [0, 1, 2, 3, 4, 5]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_nps_fvn_score(df, column_name):
    """Use 2004-2005 NPS to assign C points for fruit, vegetable and nut content."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    fvn_scores = df[column_name].apply(lambda x: calculate_nps_fvn_score(x))
    return fvn_scores


def calculate_total_nps_score(df, a_points, fiber_col, protein_col):
    """Calculate total points using 2004-2005 guidelines. Note, we currently
    don't have data on fruit, veg and nuts (fvn). So provided calculations will
    inflate nps scores for products with A points higher than 11, but which
    may contain a large proportion of fvn. Such products could may claim points
    for protein.

    Also, interpretation of the score depends on whether product is food or drink.
    """
    totals = []
    for ix, row in df.iterrows():
        a_points_sum = row[a_points].sum()
        if a_points_sum < 11:
            total = a_points_sum - row[fiber_col] - row[protein_col]
        else:
            total = a_points_sum - row[fiber_col]
        totals.append(total)
    return totals


# %% [markdown]
# ## 2. Read in nutrition_data

# %% [markdown]
# Accessing data from s3 may time out so try this [solution](https://stackoverflow.com/questions/41263304/s3-connection-timeout-when-using-boto3) or download locally

# %%
# obj = s3.get_object(Bucket='ahl-private-data', Key='kantar/data/nutrition_data.csv')
# nutrition = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding = "ISO-8859-1")

# %%
nutrition = pd.read_csv(
    PROJECT_DIR / "inputs/data/nutrition_data.csv", nrows=10000, encoding="ISO-8859-1"
)

# %%
nutrition.info()

# %%
nutrition.head()

# %%
# Looks like there are some issues with 'Fibre KG' field,
# where the values seem to be erroneously swapped with 'Fibre KG Flag'

# %%
nutrition.columns

# %%
nutrition.describe()

# %% [markdown]
# ## 3. Assign NPS scores

# %%
nps_scores = nutrition.copy()
nps_scores = nps_scores[["Purchase Number", "Purchase Period"]]

# %%
nps_scores["energy_scores"] = assign_nps_energy_score(nutrition, "Energy KCal")
nps_scores["satf_scores"] = assign_nps_satf_score(nutrition, "Saturates KG")
nps_scores["sugar_scores"] = assign_nps_sugars_score(nutrition, "Sugar KG")
nps_scores["sodium_scores"] = assign_nps_sodium_score(nutrition, "Sodium KG")
nps_scores["protein_scores"] = assign_nps_protein_score(nutrition, "Protein KG")
nps_scores["fiber_scores"] = assign_nps_fiber_score(nutrition, "Fibre KG Flag")

# %%
# Calculate total NPS scores
a_point_cols = ["energy_scores", "satf_scores", "sugar_scores", "sodium_scores"]
nps_scores["NPS_total"] = calculate_total_nps_score(
    nps_scores, a_point_cols, "fiber_scores", "protein_scores"
)

# %%
nps_scores.head()

# %% [markdown]
# ## 4. A few exploratory plots

# %%
# Distribution plots for nutrient content

fig, axs = plt.subplots(2, 3, figsize=(20, 12))

sns.histplot(
    data=nutrition,
    x="Energy KCal",
    label="Energy density",
    kde=True,
    color="olive",
    ax=axs[0, 1],
)
sns.histplot(
    data=nutrition,
    x="Saturates KG",
    label="Saturates",
    kde=True,
    color="gold",
    ax=axs[0, 2],
)
sns.histplot(
    data=nutrition, x="Sugar KG", label="Sugars", kde=True, color="teal", ax=axs[1, 0]
)
sns.histplot(
    data=nutrition, x="Sodium KG", label="Sodium", kde=True, color="teal", ax=axs[1, 1]
)
sns.histplot(
    data=nutrition,
    x="Protein KG",
    label="Protein",
    kde=True,
    color="teal",
    ax=axs[1, 2],
)
sns.histplot(
    data=nutrition,
    x="Fibre KG Flag",
    label="Fiber",
    kde=True,
    color="skyblue",
    ax=axs[0, 0],
)

plt.show()

# %%
# Box plots of nutrient content

fig, axs = plt.subplots(2, 3, figsize=(20, 12))

sns.boxplot(data=nutrition, x="Energy KCal", color="olive", ax=axs[0, 1])
sns.boxplot(data=nutrition, x="Saturates KG", color="gold", ax=axs[0, 2])
sns.boxplot(data=nutrition, x="Sugar KG", color="teal", ax=axs[1, 0])
sns.boxplot(data=nutrition, x="Sodium KG", color="teal", ax=axs[1, 1])
sns.boxplot(data=nutrition, x="Protein KG", color="teal", ax=axs[1, 2])
sns.boxplot(data=nutrition, x="Fibre KG Flag", color="skyblue", ax=axs[0, 0])

plt.show()

# %%
# Distribution plot for NPS scores

fig, axs = plt.subplots(figsize=(9, 5))

sns.histplot(
    data=nps_scores, x="NPS_total", label="Total NPS points", kde=True, color="olive"
)

plt.legend()
plt.show()

# %%
core_nutrients = [
    "Energy KCal",
    "Saturates KG",
    "Sugar KG",
    "Sodium KG",
    "Protein KG",
    "Fibre KG Flag",
]

# %%
# Produce a correlation matrix
nutrient_corr = nutrition[core_nutrients].corr()

# %%
# Correlations matrix for nutrients

# Set background color / chart style
# sns.set_style(style = 'white')

# Set up  matplotlib figure
f, ax = plt.subplots(figsize=(10, 10))

# Add diverging colormap from red to blue
cmap = sns.diverging_palette(250, 10, as_cmap=True)

sns.heatmap(
    nutrient_corr,
    #             mask=mask,
    cmap=cmap,
    square=True,
    linewidth=0.5,
    cbar_kws={"shrink": 0.5},
    ax=ax,
)

# %%
