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

# %%
import boto3
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import collections

# %%
from ahl_food_reformulation import PROJECT_DIR

# %% [markdown]
# ## 1. Read in data and relevant lookups

# %%
# Subset of purchases
subset = pd.read_csv(
    PROJECT_DIR / "inputs/data/purchasing_records_subset.csv", encoding="utf-8"
)

# %%
# Product attributes
combined_prod_att = pd.read_csv(PROJECT_DIR / "lookups/product_attribute_mapping.csv")

# %%
# Household info
combined_demog = pd.read_csv(PROJECT_DIR / "lookups/panel_demog_char.csv")


# %%
# Kantar product taxonomy
# I processed the original file to be more machine readable

# %% [markdown]
# ## 2. Trial measuring universality

# %%
def get_segments(df, column_name, n=4):
    """Group the values from the specified column in a df into n-segments.

    Args:
        df: A dataframe.
        column_name: A string that specifies the name of the variable in the dataframe.
        n: An int indicating number of segments. Default value is 4 (equivalent to quartiles).

    Returns:
        An array with segment labels.

    Raises:
        TypeError: If incorrect data type was passed in args.
    """
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    labelled_series = pd.qcut(df[column_name], n, labels=False)
    return labelled_series.values


# %%
def gini_discrete(df, column_name):
    value_counts = df[column_name].value_counts(normalize=True)
    value_counts = sorted(value_counts)
    # see this post for calculations https://stackoverflow.com/questions/31416664/python-gini-coefficient-calculation-using-numpy
    value_counts.insert(0, 0)
    shares_cumsum = np.cumsum(a=value_counts, axis=None)
    # perfect equality area
    pe_line = np.linspace(start=0.0, stop=1.0, num=len(shares_cumsum))
    area_under_pe = np.trapz(y=pe_line, dx=1 / len(shares_cumsum))
    # lorenz area
    area_under_lorenz = np.trapz(y=shares_cumsum, dx=1 / len(shares_cumsum))
    gini = (area_under_pe - area_under_lorenz) / area_under_pe
    return gini


# %%
# Subset a particular narrow category
combined_prod_att.head(3)

# %%
combined_prod_att_man = combined_prod_att[
    combined_prod_att["Attribute"] == "Manufacturer"
]

# %%
select_cat = combined_prod_att[combined_prod_att["Value"] == "Granola"]

# %%
len(select_cat)

# %%
# Select only purchases within selected category
subset_granola = subset[subset["Product Code"].isin(select_cat["Product Code"].values)]

# %%
# Need a better way to define the average price
subset_granola["Avg_price"] = subset_granola["Spend"] / subset_granola["Quantity"]

# %%
price_segs = get_segments(subset_granola, "Avg_price")

# %%
subset_granola["Price_segment"] = price_segs

# %% [markdown]
# ### Gini index for price segments

# %% [markdown]
# NB: You may want to check if the current way the Gini index is calculated is appropriate and explore it can be weighted by sales volume.

# %%
subset_granola = subset_granola.merge(
    combined_prod_att_man, left_on="Product Code", right_on="Product Code", how="left"
)

# %%
subset_granola.head()

# %%
# Measure Gini index
gini_discrete(subset_granola, "Price_segment")

# %%
# Plot of equality
value_counts = subset_granola["Price_segment"].value_counts(normalize=True)
value_counts = sorted(value_counts)
# see this post for calculations https://stackoverflow.com/questions/31416664/python-gini-coefficient-calculation-using-numpy
value_counts.insert(0, 0)
shares_cumsum = np.cumsum(a=value_counts, axis=None)
# perfect equality area
pe_line = np.linspace(start=0.0, stop=1.0, num=len(shares_cumsum))
area_under_pe = np.trapz(y=pe_line, dx=1 / len(shares_cumsum))
# lorenz area
area_under_lorenz = np.trapz(y=shares_cumsum, dx=1 / len(shares_cumsum))
gini = (area_under_pe - area_under_lorenz) / area_under_pe


plt.plot(pe_line, shares_cumsum, label="lorenz_curve")
plt.plot(pe_line, pe_line, label="perfect_equality")
plt.fill_between(pe_line, shares_cumsum)
plt.title("Gini: {}".format(gini), fontsize=20)
plt.ylabel("Cummulative price segment share", fontsize=15)
plt.xlabel("Price segment share (Lowest to Highest)", fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Gini index for manufacturer

# %%
# Measure Gini index
gini_discrete(subset_granola, "Value")

# %%
collections.Counter(subset_granola["Value"])

# %%
# Plot of equality
value_counts = subset_granola["Value"].value_counts(normalize=True)
value_counts = sorted(value_counts)
# see this post for calculations https://stackoverflow.com/questions/31416664/python-gini-coefficient-calculation-using-numpy
value_counts.insert(0, 0)
shares_cumsum = np.cumsum(a=value_counts, axis=None)
# perfect equality area
pe_line = np.linspace(start=0.0, stop=1.0, num=len(shares_cumsum))
area_under_pe = np.trapz(y=pe_line, dx=1 / len(shares_cumsum))
# lorenz area
area_under_lorenz = np.trapz(y=shares_cumsum, dx=1 / len(shares_cumsum))
gini = (area_under_pe - area_under_lorenz) / area_under_pe


plt.plot(pe_line, shares_cumsum, label="lorenz_curve")
plt.plot(pe_line, pe_line, label="perfect_equality")
plt.fill_between(pe_line, shares_cumsum)
plt.title("Gini: {}".format(gini), fontsize=20)
plt.ylabel("Cummulative manufacturer share", fontsize=15)
plt.xlabel("Manufacturer share (Lowest to Highest)", fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()

# %%
