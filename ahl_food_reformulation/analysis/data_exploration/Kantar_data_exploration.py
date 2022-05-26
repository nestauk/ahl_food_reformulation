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
# ## 1. Access data

# %%
import boto3
import pandas as pd
import io
import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt

# %%
from ahl_food_reformulation import PROJECT_DIR

# %%
# You can find info on relationships between various tables/files in data_schema.pdf in the root folder

# %% [markdown]
# ### Quick exploration of the purchase_records

# %%
purchase_records = pd.read_csv(
    PROJECT_DIR / "inputs/data/purchase_records.csv"
)  # nrows = 1000000

# %%
purchase_records.head(3)

# %%
purchase_records.info(null_counts=True)

# %%
purchase_records.columns

# %%
# Unique records
len(purchase_records)

# %%
# Unique households
len(purchase_records["Panel Id"].unique())

# %%
# Unique purchase IDs
len(purchase_records["PurchaseId"].unique())

# %%
# Unique products
len(purchase_records["Product Code"].unique())

# %%
# Unique retailers
len(purchase_records["Store Code"].unique())

# %%
# Total spend
purchase_records["Spend"].sum()

# %%
# Total volume
purchase_records["Volume"].sum()

# %%
# Unique promo codes
len(purchase_records["Promo Code"].unique())

# %%
# Distribution of spend
fig, axs = plt.subplots(figsize=(12, 7))

ax = sns.kdeplot(
    purchase_records["Spend"], color="teal", label="Spend"
)  # use weightes arg when we get this info

ax.set_xlabel("Spend")
# ax.set_ylabel("Density")
plt.legend()

# %%
# Products with the highest spend
purchase_records.sort_values(by="Spend", ascending=False).head(25)

# %%
# Distribution of volume
fig, axs = plt.subplots(figsize=(12, 7))

ax = sns.kdeplot(
    purchase_records["Volume"], color="olive", label="Volume"
)  # use weightes arg when we get this info

ax.set_xlabel("Volume")
# ax.set_ylabel("Density")
plt.legend()

# %%
# Products with the highest volume
purchase_records.sort_values(by="Volume", ascending=False).head(25)

# %%
# Typical basket
purchases = purchase_records.groupby("PurchaseId")["Spend", "Volume"].sum()

# %%
# Distribution of basket spend
fig, axs = plt.subplots(figsize=(12, 7))

ax = sns.kdeplot(
    purchases["Spend"], color="teal", label="Basket Spend"
)  # use weightes arg when we get this info

ax.set_xlabel("Basket Spend")
# ax.set_ylabel("Density")
plt.legend()

# %%
# Distribution of basket volume
fig, axs = plt.subplots(figsize=(12, 7))

ax = sns.kdeplot(
    purchases["Volume"], color="teal", label="Basket Volume"
)  # use weightes arg when we get this info

ax.set_xlabel("Basket Volume")
# ax.set_ylabel("Density")
plt.legend()

# %%
subset = purchase_records[purchase_records["Period"] == 202108]

# %%
subset.to_csv(
    PROJECT_DIR / "inputs/data/purchasing_records_subset.csv",
    encoding="utf-8",
    index=False,
)

# %%
# from pandas_profiling import ProfileReport

# %%
# !{sys.executable} -m pip install -U pandas-profiling[notebook]
# # !jupyter nbextension enable --py widgetsnbextension

# %% [markdown]
# ### Quick exploration of the product_master

# %%
s3 = boto3.client("s3")
# list objects
objects = s3.list_objects_v2(Bucket="ahl-private-data", Prefix="kantar/data/")[
    "Contents"
]
keys = [obj["Key"] for obj in objects]

# retrive_single_object
obj = s3.get_object(Bucket="ahl-private-data", Key="kantar/data/product_master.csv")
products = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="ISO-8859-1")

# %%
products.head(3)

# %%
products[products["Product Code"] == 40346]

# %%
products[products["Product Code"] == 139410]

# %%
# Unique products
len(products)

# %% [markdown]
# ### Quick exploration of the panel_household_master and panel_individual_master

# %%
# retrive_single_object
obj = s3.get_object(
    Bucket="ahl-private-data", Key="kantar/data/panel_household_master.csv"
)
panel_hh = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="ISO-8859-1")

obj2 = s3.get_object(
    Bucket="ahl-private-data", Key="kantar/data/panel_individual_master.csv"
)
panel_indiv = pd.read_csv(io.BytesIO(obj2["Body"].read()), encoding="ISO-8859-1")


# %%
len(panel_hh)

# %%
len(panel_indiv)

# %%
panel_hh.head()

# %%
panel_indiv.head()

# %%
# Distribution of BMI
# Note: some data on height, weight and BMI is missing

fig, axs = plt.subplots(figsize=(12, 7))

ax = sns.kdeplot(
    panel_indiv["BMI"], color="skyblue", label="BMI"
)  # use weightes arg when we get this info

ax.set_xlabel("BMI")
# ax.set_ylabel("Density")
plt.legend()


# %%
# Non missing BMI
len(panel_indiv[panel_indiv["BMI"] != 0.0])

# %%
44975 / len(panel_indiv)

# %%
# Household size
# Note: some data on height, weight and BMI is missing

fig, axs = plt.subplots(figsize=(12, 7))

ax = sns.kdeplot(
    panel_hh["Household Size"], color="skyblue", label="Household size"
)  # use weightes arg when we get this info

ax.set_xlabel("Household size")
# ax.set_ylabel("Density")
plt.legend()

# %%
# Number of children
# Note: some data on height, weight and BMI is missing

fig, axs = plt.subplots(figsize=(12, 7))

ax = sns.kdeplot(
    panel_hh["Number of Children"], color="olive", label="Number of Children"
)  # use weightes arg when we get this info

ax.set_xlabel("Number of children")
# ax.set_ylabel("Density")
plt.legend()

# %%
# Age
# Note: some data on height, weight and BMI is missing

fig, axs = plt.subplots(figsize=(12, 7))

ax = sns.kdeplot(
    panel_indiv["Age"], color="red", label="Age"
)  # use weightes arg when we get this info

ax.set_xlabel("Age")
# ax.set_ylabel("Density")
plt.legend()

# %%
# Non missing Age
len(panel_indiv[panel_indiv["Age"] != 0.0])

# %%
77975 / len(panel_indiv)

# %% [markdown]
# ### Check product attribute tables (product_attribute, product_attribute_coding and product_attribute_values)

# %%
# retrive_single_object
obj = s3.get_object(Bucket="ahl-private-data", Key="kantar/data/product_attribute.csv")
prod_attribute = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="ISO-8859-1")

obj = s3.get_object(
    Bucket="ahl-private-data", Key="kantar/data/product_attribute_coding.csv"
)
prod_attribute_code = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="ISO-8859-1")

obj = s3.get_object(
    Bucket="ahl-private-data", Key="kantar/data/product_attribute_values.csv"
)
prod_attribute_val = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="ISO-8859-1")


# %%
# This corresponds to Kanter internal taxonomy of products
# See 210510 - Kantar Standard Category Detail as of May 2021.xlsx
prod_attribute["Attribute Description"].unique()

# %%
att_dict = dict()
for ix, row in prod_attribute.iterrows():
    att_dict[row["Attribute Number"]] = row["Attribute Description"]

# %%
att_dict[2829] = "Unit_category"

# %% [markdown]
# To get categories, we want
# 1. Unique Product Codes (from product_master)
# 2. Merge with product_attribute_coding
# 3. Merge with product_attribute_values (for specific product_attribute_codes)

# %%
# Define select product attribute codes of interest
select_codes = [3, 5, 2829, 200]

# %%
prod_att_subset = prod_attribute_code[
    prod_attribute_code["Attribute Number"].isin(select_codes)
]

# %%
prod_att_subset["Attribute"] = prod_att_subset["Attribute Number"].apply(
    lambda x: att_dict[x]
)

# %%
combined_prod_att = prod_att_subset.merge(
    prod_attribute_val, left_on="Attribute Value", right_on="Attribute Code", how="left"
)

# %%
combined_prod_att.head()

# %%
len(combined_prod_att)

# %%
combined_prod_att = combined_prod_att[
    ["Product Code", "Attribute", "Attribute Code Description"]
]

# %%
combined_prod_att.columns = ["Product Code", "Attribute", "Value"]

# %%
combined_prod_att.to_csv(
    PROJECT_DIR / "lookups/product_attribute_mapping.csv", index=False
)

# %% [markdown]
# ### Check panel_demographic_coding

# %%
obj = s3.get_object(
    Bucket="ahl-private-data", Key="kantar/data/panel_demographic_coding.csv"
)
panel_demog_coding = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="ISO-8859-1")

obj = s3.get_object(Bucket="ahl-private-data", Key="kantar/data/panel_demographics.csv")
panel_demog = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="ISO-8859-1")

obj = s3.get_object(
    Bucket="ahl-private-data", Key="kantar/data/panel_demographic_values.csv"
)
panel_demog_val = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="ISO-8859-1")

# %%
panel_demog

# %%
panel_demog_dict = dict()
panel_demog_dict[2] = "Urban/rural"
panel_demog_dict[3] = "Social_class"
panel_demog_dict[4] = "Council_tax_band"
panel_demog_dict[5] = "Region"
panel_demog_dict[7] = "Newspaper_read"
panel_demog_dict[8] = "Life_stage"
panel_demog_dict[9] = "Household_income"
panel_demog_dict[11] = "Ethnicity"
panel_demog_dict[12] = "Education_level"

# %%
panel_demog_coding["Demographic_va"] = panel_demog_coding["Demog Id"].apply(
    lambda x: panel_demog_dict[x]
)

# %%
combined_demog = panel_demog_coding.merge(
    panel_demog_val, left_on="Demog Value", right_on=' "Demog Value"', how="left"
)

# %%
combined_demog = combined_demog[["Panel Id", "Demographic_va", ' "Demog Description""']]
combined_demog.columns = ["Panel Id", "Demog_char", "Demog_val"]

# %%
combined_demog.to_csv(PROJECT_DIR / "lookups/panel_demog_char.csv", index=False)

# %%
combined_demog.head()

# %%
