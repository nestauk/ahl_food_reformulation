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

# %% [markdown]
# ## EDA of household demographics shopping baskets

# %%
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ahl_food_reformulation import PROJECT_DIR

# %% [markdown]
# Expanding on the [initial EDA performed](https://docs.google.com/document/d/1MT-p6gV9HpHJ0_4X0ayC24GYWzp-fapipwujSiuFS5o/edit?usp=sharing). Explore and display the distributions of purchases across household demographic groups.
#
# - [ ] Define a set of groups from the household demographic information
# - [ ] Display the following distributions across the demographic groups
#   - [ ] Purchasing volume
#   - [ ] Volume of food categories purchases
#   - [ ] Contribution to food category in annual / monthly purchases
#   - [ ] Contribution to food category compared to total of households

# %%
# Info needed
# Purchasing records
# Household demographic information

# %%
# Testing first with purchase subset - August 2021
pur_recs = pd.read_csv(PROJECT_DIR / "inputs/data/purchasing_records_subset.csv")

# %%
# pur_recs = pd.read_csv(
#   PROJECT_DIR / "inputs/data/purchase_records.csv")

# %%
# Household information
pan_mast = pd.read_csv(
    PROJECT_DIR / "inputs/data/panel_household_master.csv", encoding="ISO-8859-1"
)
pan_ind_mast = pd.read_csv(
    PROJECT_DIR / "inputs/data/panel_individual_master.csv", encoding="ISO-8859-1"
)
pan_dem = pd.read_csv(PROJECT_DIR / "inputs/data/panel_demographics.csv")
pan_dem_cod = pd.read_csv(PROJECT_DIR / "inputs/data/panel_demographic_coding.csv")
pan_dem_val = pd.read_csv(
    PROJECT_DIR / "inputs/data/panel_demographic_values.csv", encoding="ISO-8859-1"
)

# %% [markdown]
# #### Data quality checks

# %%
# Check for duplicate panel values - none
print(pan_mast.duplicated().any())
print(pan_mast["Panel Id"].duplicated().any())
print(pan_ind_mast.duplicated().any())

# %%
# Check if there are values in one df but not the other (on all purchase records)
unique_pan_pur = pur_recs.drop_duplicates(subset=["Panel Id"])
df_diff = pd.concat([unique_pan_pur, pan_mast]).drop_duplicates(
    subset=["Panel Id"], keep=False
)

# None in purchase records but not product master
# print(df_diff['Product Code'].value_counts())
# 11 in master but not purchase records
# print(df_diff.shape)

# %% [markdown]
# #### Create panel demographic dataset

# %%
## BMI 25 or over (ind records)
pan_ind_mast["high_bmi"] = np.where(pan_ind_mast["BMI"] >= 25, 1, 0)
pan_ind_mast["bmi_missing"] = np.where(pan_ind_mast["BMI"] == 0, 1, 0)
pan_ind_bmi = pan_ind_mast[["Panel Id", "high_bmi", "bmi_missing"]].copy()
pan_ind_bmi = pan_ind_bmi.groupby(by=["Panel Id"]).sum()
pan_mast_full = pan_mast.merge(
    pan_ind_bmi, left_on="Panel Id", right_on="Panel Id", how="left"
)

# %%
pan_mast_full.head(1)

# %%
# Demographic codes

# %%
pan_dem_val.columns = ["Demog Id", "Demog Value", "Demog Description"]

# %%
dem_cod = pan_dem_cod.merge(
    pan_dem_val,
    left_on=["Demog Id", "Demog Value"],
    right_on=["Demog Id", "Demog Value"],
    how="left",
)
dem_cod.drop(["Demog Value"], axis=1, inplace=True)


# %%
def map_codes(df, col):
    di = {
        2: "Urban-Rural",
        3: "Social Class",
        4: "Council Tax Band",
        5: "Region",
        7: "Newspaper Read",
        8: "Life Stage",
        9: "Household Income",
        11: "Ethnicity",
        12: "Education Level",
    }
    df[col] = df[col].map(di)
    return df


# %%
dem_cod = map_codes(dem_cod, "Demog Id")
dem_cod.set_index("Panel Id", inplace=True)
pan_dem = dem_cod.pivot_table(
    values="Demog Description", index=dem_cod.index, columns="Demog Id", aggfunc="first"
)

# %%
pan_mast_full = pan_mast_full.merge(
    pan_dem, left_on="Panel Id", right_index=True, how="left"
)

# %%
pan_mast_full.head(1)

# %%
pan_mast_full.to_csv(PROJECT_DIR / "outputs/data/panel_demographic_table_202108.csv")

# %% [markdown]
# #### Groups (add to this)
# - Postcode District
# - Council Tax Band
# - Household size (bins)
# - Children Y/N
# - High BMI in hh (Y/N)
# - Percent high BMI

# %% [markdown]
# #### Link data

# %%
pur_recs.shape

# %%
ctx_pur_recs = pur_recs.merge(
    pan_mast_full[["Panel Id", "Council Tax Band"]], on="Panel Id", how="left"
)

# %% [markdown]
# #### Council tax

# %%
ctx_pur_recs.head(1)

# %%
ctax_total_pur = (
    ctx_pur_recs["Council Tax Band"]
    .value_counts()
    .rename_axis("Council Tax Band")
    .reset_index(name="purchases")
)
ctx_pur_pan = (
    ctx_pur_recs.groupby(["Council Tax Band", "Purchase Date"])["Panel Id"]
    .value_counts()
    .rename("panel_purchases")
    .reset_index()
)
ctx_pans = ctx_pur_pan.groupby(["Council Tax Band"])["Panel Id"].nunique().reset_index()
ctax_total_pur = ctax_total_pur.merge(ctx_pans, how="inner", on="Council Tax Band")
ctax_total_pur["avg_panel_puchases"] = (
    ctax_total_pur["purchases"] / ctax_total_pur["Panel Id"]
)

# %%
ctax_total_pur.set_index("Council Tax Band")["avg_panel_puchases"].sort_values().plot(
    kind="barh"
)

# %%
plt.figure(figsize=(15, 8))

# Daily purchases per panel - ctax band
sns.boxplot(x="Council Tax Band", y="panel_purchases", data=ctx_pur_pan)
plt.xticks(rotation=90)

# %%
