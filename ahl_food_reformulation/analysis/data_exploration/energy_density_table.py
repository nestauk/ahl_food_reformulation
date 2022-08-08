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
import pandas as pd
from ahl_food_reformulation import PROJECT_DIR

# %%
pur_recs = pd.read_csv(PROJECT_DIR / "inputs/data/purchase_records.csv")
nut_recs = pd.read_csv(PROJECT_DIR / "inputs/data/nutrition_data.csv")
prod_mast = pd.read_csv(
    PROJECT_DIR / "inputs/data/product_master.csv", encoding="ISO-8859-1"
)
val_fields = pd.read_csv(PROJECT_DIR / "inputs/data/validation_field.csv")

# %%
uom = pd.read_csv(
    PROJECT_DIR / "inputs/data/uom.csv",
    header=0,
    names=["UOM", "Reported Volume"],
)

# %%
val_fields.drop_duplicates(inplace=True)  # Remove duplicates
pur_recs = pur_recs[
    ["PurchaseId", "Period", "Product Code", "Gross Up Weight", "Quantity", "Volume"]
].merge(prod_mast[["Product Code", "Validation Field"]], on="Product Code", how="left")
pur_recs = pur_recs.merge(
    val_fields[["VF", "UOM"]], left_on="Validation Field", right_on="VF", how="left"
)

# %%
pur_recs = pur_recs.merge(uom[["UOM", "Reported Volume"]], on="UOM", how="left")

# %%
nut_recs["pur_id"] = (
    nut_recs["Purchase Number"].astype(str)
    + "_"
    + nut_recs["Purchase Period"].astype(str)
)
pur_recs["pur_id"] = (
    pur_recs["PurchaseId"].astype(str) + "_" + pur_recs["Period"].astype(str)
)

# %%
pur_recs_kilos = pur_recs[pur_recs["Reported Volume"] == "Kilos"].copy()

# %%
pur_recs_kilos = pur_recs_kilos[pur_recs_kilos["Volume"] != 0]
pur_recs_kilos["grams"] = pur_recs_kilos["Volume"] * 1000

# %%
pur_recs_kilos = pur_recs_kilos.merge(
    nut_recs[["pur_id", "Energy KCal"]], on="pur_id", how="left"
)

# %%
pur_recs_kilos["energy_density"] = (
    pur_recs_kilos["Energy KCal"] / pur_recs_kilos["grams"]
)

# %%
import seaborn as sns

# %%
sns.displot(pur_recs_kilos, x="energy_density", stat="density")

# %%
val_fields[val_fields["VF"] == 216]

# %%
prod_mast[prod_mast["Product Code"] == 1113]

# %%
pur_recs_kilos.shape

# %%
pur_recs_kilos[pur_recs_kilos.energy_density > 40].sort_values(
    by="energy_density", ascending=False
)

# %%
pur_recs_kilos[pur_recs_kilos.energy_density > 40].sort_values(
    by="energy_density", ascending=False
)["Reported Volume"].value_counts()

# %%
pur_recs_kilos.head(10)

# %%
prod_mast[prod_mast["Product Code"] == 1113]

# %% [markdown]
# - Very low energy density foods = less than 0.6 kcal/g
# - Low energy density foods = 0.6 to 1.5 kcal/g
# - Medium energy density foods = 1.5 to 4 kcal/g
# - High energy density foods = more than 4 kcal/g

# %%
bins = [0, 0.6, 1.5, 4.01, 40]
labels = ["very low", "low", "medium", "high"]
pur_recs_kilos["bins"] = pd.cut(
    pur_recs_kilos["energy_density"], bins, right=False, labels=labels
)

# %%
pur_recs_kilos = pur_recs_kilos[pur_recs_kilos["bins"].notna()]

# %%
pur_recs_kilos[pur_recs_kilos["energy_density"] == 4]

# %%
pur_recs_kilos["kcal_gross_up"] = (
    pur_recs_kilos["Energy KCal"] * pur_recs_kilos["Gross Up Weight"]
)

# %%
(
    pur_recs_kilos.groupby("bins")["kcal_gross_up"].sum()
    / pur_recs_kilos["kcal_gross_up"].sum()
) * 100

# %%
# sales weighted average
# Quantity x gross up weight per row
# Total quantity per group / total quantity (weighted avg)

# %%
pur_recs_kilos["quant_gross_up"] = (
    pur_recs_kilos["Quantity"] * pur_recs_kilos["Gross Up Weight"]
)

# %%
(
    pur_recs_kilos.groupby("bins")["quant_gross_up"].sum()
    / pur_recs_kilos["quant_gross_up"].sum()
) * 100

# %%
(
    pur_recs_kilos.groupby("bins")["Product Code"].nunique()
    / (pur_recs_kilos["Product Code"].nunique())
) * 100

# %%
# (0.143887 + 0.244483 + 0.468753 + 0.179540)  # Will be over as some products in both

# %%
