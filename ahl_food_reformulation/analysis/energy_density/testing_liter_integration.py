from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import energy_density as energy
import pandas as pd
import matplotlib.pyplot as plt


# read data
pur_recs = kantar.purchase_records()
nut_recs = kantar.nutrition()
prod_mast = kantar.product_master()
val_fields = kantar.val_fields()
uom = kantar.uom()
prod_meta = kantar.product_metadata()
prod_meas = kantar.product_measurement()


# run function with rst_4_extended
extended = energy.cat_energy_100(
    "rst_4_extended",
    val_fields,
    prod_mast,
    uom,
    pur_recs,
    nut_recs,
    prod_meta,
    prod_meas,
)

extended.head()


# extract categorisations

categories = prod_meta[
    ["rst_4_market", "rst_4_sub_market", "rst_4_market_sector", "rst_4_extended"]
].drop_duplicates()


# merge with categories

df = categories.merge(extended, on="rst_4_extended")

df.head()

# check distribution

df["rst_4_market_sector"].value_counts()

# plot example of Packet Breakfast
# the two charts below show the distribution of energy density of granual categories within the packet breakfast category
# chart 1 is the product count
# chart 2 is the product sold

sector = "Packet Breakfast"

sector_df = df[df["rst_4_market_sector"] == sector].copy()
sector_df


sector_df["bin_s"] = pd.cut(sector_df["kcal_100_s"], 15)

cut_counts = sector_df["bin_s"].value_counts().sort_index().reset_index()

x = cut_counts["index"].astype(str)
y = cut_counts["bin_s"]

fig, ax = plt.subplots(figsize=(12, 8))
plt.xlabel("Kcal per 100g")
plt.ylabel("Product Count")
plt.title("Products" + "-" + sector)
plt.bar(x, y, alpha=0.5)

fig.tight_layout()


sector_df["bin_w"] = pd.cut(sector_df["kcal_100_w"], 15)

cut_counts = sector_df["bin_w"].value_counts().sort_index().reset_index()

x = cut_counts["index"].astype(str)
y = cut_counts["bin_w"]

fig, ax = plt.subplots(figsize=(12, 8))
plt.xlabel("Kcal per 100g")
plt.ylabel("Product Count")
plt.title("Products Sold" + "-" + sector)
plt.bar(x, y, alpha=0.5)

fig.tight_layout()
