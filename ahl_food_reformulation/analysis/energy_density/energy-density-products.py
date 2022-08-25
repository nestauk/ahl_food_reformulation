from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.utils import lookups as lps
from ahl_food_reformulation import PROJECT_DIR
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# function to create percentiles
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = "percentile_%s" % n
    return percentile_


# This script produces descriptive statistics for energy density for a unique set of products in the Kantar take home panel

# read data
pur_recs = kantar.purchase_records()
nut_recs = kantar.nutrition()
prod_mast = kantar.product_master()
val_fields = kantar.val_fields()
uom = kantar.uom()
prod_meta = kantar.product_metadata()

# add standardised volume measurement
pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)

# slice by only kilos (later we will add other measurements)
pur_rec_kilos = pur_rec_vol[pur_rec_vol["Reported Volume"] == "Kilos"].copy()

# generate unique list of products
unique_prods_nut = lps.products_per_100g(["Energy KCal"], pur_rec_kilos, nut_recs)

# remove infinite and missing values
unique_prods_nut.replace([np.inf, -np.inf], np.nan, inplace=True)

unique_prods_nut.dropna(inplace=True)

unique_prods_nut.head()

# generate energy density variable
unique_prods_nut["energy_density"] = unique_prods_nut["Energy KCal_100g"] / 100

# generate energy density category variable based on standard thresholds
unique_prods_nut["energy_density_cat"] = pd.cut(
    unique_prods_nut["energy_density"],
    bins=[0, 0.6, 1.5, 4, float("Inf")],
    labels=["very_low", "low", "medium", "high"],
)

# merge in product info
prod_all = prod_meta.merge(
    unique_prods_nut, left_on=["product_code"], right_on=["Product Code"], how="inner"
).dropna(subset=["Energy KCal_100g"])

prod_all.head()

# describe variables in prod_all
prod_all.describe()


# there are some inplausibla values for energy desnity - drop any value above 20
print(len(prod_all[prod_all["energy_density"] >= 20].index))
prod_all_clean = prod_all[prod_all["energy_density"] < 20]

prod_all.head()

# distribution of energy density across all products
prod_all_clean.energy_density.plot.density(color="black")
plt.show()

# distribution across product groups
grouped = prod_all_clean.groupby("rst_4_market_sector")
column = grouped["energy_density"]
sector_tbl = column.agg(
    [
        "count",
        np.mean,
        np.std,
        np.median,
        np.var,
        np.min,
        np.max,
        percentile(25),
        percentile(75),
    ]
)

sector_tbl

# average energy density for product groups with reference for high density threshold (4)
sector_tbl_sort = sector_tbl.sort_values("mean")

x = sector_tbl_sort.index
y = sector_tbl_sort["mean"]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y)
plt.xticks(x_pos, x)
plt.xticks(rotation=90)
plt.axhline(y=4)

plt.show()

# share of products by energy density category

total = prod_all_clean.groupby(["rst_4_market_sector"]).size().reset_index(name="total")
g = (
    prod_all_clean.groupby(["energy_density_cat", "rst_4_market_sector"])
    .size()
    .reset_index(name="counts")
    .merge(total, on="rst_4_market_sector", how="left")
)

g["share"] = g["counts"] / g["total"]

piv = g.pivot(index="energy_density_cat", columns="rst_4_market_sector", values="share")

sns.heatmap(piv, cmap="YlGnBu")
