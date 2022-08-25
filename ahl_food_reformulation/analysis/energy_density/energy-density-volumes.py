from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.utils import lookups as lps
from ahl_food_reformulation import PROJECT_DIR
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from cmath import nan

# read data
pur_recs = kantar.purchase_records()
nut_recs = kantar.nutrition()
prod_mast = kantar.product_master()
val_fields = kantar.val_fields()
uom = kantar.uom()
prod_meta = kantar.product_metadata()


def add_energy_density(pur_recs):
    """
    Adds four columns to the purchase record:  energy_density (kcal per 1g),  energy_density_cat ('very_low', 'low', 'medium', 'high' based on thresholds), Reported Volume, kcal per 100g
    Args:
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
    Returns:
        pd.DateFrame: Dataframe that is a copy of pur_rec with two additional columns
    """
    # Convert to datetime format
    pur_recs["Purchase Date"] = pd.to_datetime(
        pur_recs["Purchase Date"], format="%d/%m/%Y"
    )

    # add standardised volume measurement
    pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)

    # slice by only kilos (later we will add other measurements)
    pur_rec_kilos = pur_rec_vol[pur_rec_vol["Reported Volume"] == "Kilos"].copy()

    # generate unique list of products
    unique_prods_nut = lps.products_per_100g(["Energy KCal"], pur_rec_kilos, nut_recs)

    # generate energy density variable
    unique_prods_nut["energy_density"] = unique_prods_nut["Energy KCal_100g"] / 100

    # generate energy density category variable based on standard thresholds
    unique_prods_nut["energy_density_cat"] = pd.cut(
        unique_prods_nut["energy_density"],
        bins=[0, 0.6, 1.5, 4, float("Inf")],
        labels=["very_low", "low", "medium", "high"],
    )

    # remove implausible values
    unique_prods_nut = unique_prods_nut[unique_prods_nut["energy_density"] < 20]

    # merge with purchase record
    out = pur_rec_kilos.merge(unique_prods_nut, on="Product Code", how="left")

    return out


pur_recs_energy = add_energy_density(pur_recs)

# at the moment I am only looking at products with reported volume of kilos


pur_recs_energy["product_weight"] = np.where(
    pur_recs_energy["Reported Volume"] == "Kilos", pur_recs_energy["Volume"], nan
)

# subset to products with non-missing energy density
pur_recs_energy.dropna(inplace=True, subset=["energy_density_cat"])

# add scaled gross-up weight
pur_recs_energy["scaled_gross_up_factor"] = (
    pur_recs_energy["Gross Up Weight"] * pur_recs_energy["product_weight"]
)

pur_recs_energy.shape
pur_recs_energy.head()

# distribution of energy density of all purchases (population)
grouped_pop = pur_recs_energy.groupby("energy_density_cat")
column_pop = grouped_pop["scaled_gross_up_factor"]
tbl_pop = column_pop.agg(["sum"])
sum_pop = tbl_pop["sum"].sum()

tbl_pop / sum_pop

# distribution of energy density of all purchases (sample)
grouped_sam = pur_recs_energy.groupby("energy_density_cat")
column_sam = grouped_sam["product_weight"]
tbl_sam = column_sam.agg(["sum"])
sum_sam = tbl_sam["sum"].sum()
tbl_sam / sum_sam

# variation by month
# distribution of energy density of all purchases (population)
pur_recs_energy["month"] = pur_recs_energy["Purchase Date"].dt.month
total = (
    pur_recs_energy.groupby(["month"])["scaled_gross_up_factor"]
    .sum()
    .reset_index(name="total")
)
g = (
    pur_recs_energy.groupby(["energy_density_cat", "month"])["scaled_gross_up_factor"]
    .sum()
    .reset_index(name="counts")
    .merge(total, on="month", how="left")
)

g["share"] = g["counts"] / g["total"]
g_high = g[g["energy_density_cat"] == "high"].copy()

plt.plot(g_high["month"], g_high.share)
plt.title("Share of High Density Purchases")
plt.suptitle("Population")
plt.xlabel("Month")
plt.ylabel("Frequency")
plt.show()

piv = g.pivot(index="energy_density_cat", columns="month", values="share")

sns.heatmap(piv, cmap="YlGnBu")

# get product categories
# merge in product info

prod_all = prod_meta.merge(
    pur_recs_energy, left_on=["product_code"], right_on=["Product Code"], how="inner"
)

prod_all.head()

# distribution across product groups - weighted by sale volume and weights

total = (
    prod_all.groupby(["rst_4_market_sector"])["scaled_gross_up_factor"]
    .sum()
    .reset_index(name="total")
)
g = (
    prod_all.groupby(["energy_density_cat", "rst_4_market_sector"])[
        "scaled_gross_up_factor"
    ]
    .sum()
    .reset_index(name="counts")
    .merge(total, on="rst_4_market_sector", how="left")
)

g["share"] = g["counts"] / g["total"]

# graph shows the share of sales that are for high density products across product categories

piv = g.pivot(index="rst_4_market_sector", columns="energy_density_cat", values="share")

sector_tbl_sort = piv.sort_values("high")

x = sector_tbl_sort.index
y = sector_tbl_sort["high"]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y)
plt.xticks(x_pos, x)
plt.xticks(rotation=90)

plt.show()

# graph shows the share of sales that are for medium density products across product categories
#
piv = g.pivot(index="rst_4_market_sector", columns="energy_density_cat", values="share")

sector_tbl_sort = piv.sort_values("medium")

x = sector_tbl_sort.index
y = sector_tbl_sort["medium"]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y)
plt.xticks(x_pos, x)
plt.xticks(rotation=90)

plt.show()

# graph shows the share of sales that are for low density products across product categories
#

piv = g.pivot(index="rst_4_market_sector", columns="energy_density_cat", values="share")

sector_tbl_sort = piv.sort_values("low")

x = sector_tbl_sort.index
y = sector_tbl_sort["low"]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y)
plt.xticks(x_pos, x)
plt.xticks(rotation=90)

plt.show()

# graph shows the share of sales that are for very low density products across product categories
#

piv = g.pivot(index="rst_4_market_sector", columns="energy_density_cat", values="share")

sector_tbl_sort = piv.sort_values("very_low")

x = sector_tbl_sort.index
y = sector_tbl_sort["very_low"]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y)
plt.xticks(x_pos, x)
plt.xticks(rotation=90)

plt.show()

# most bought products categories in the high density category

high_den = prod_all[prod_all.energy_density_cat == "high"]

high_den_pop = (
    pd.pivot_table(
        high_den,
        values=["scaled_gross_up_factor"],
        index=["rst_4_extended"],
        aggfunc="sum",
    )
    .reset_index()
    .sort_values(by=["scaled_gross_up_factor"], ascending=False)
)

high_den_pop.head(10)

# add product names

high_den_prod = high_den.merge(prod_mast, on="Product Code", how="left")

# most bought products in the high density category

high_den_prod_tbl = (
    pd.pivot_table(
        high_den_prod,
        values=["scaled_gross_up_factor"],
        index=["Product Long Description"],
        aggfunc="sum",
    )
    .reset_index()
    .sort_values(by=["scaled_gross_up_factor"], ascending=False)
)

high_den_prod_tbl.head(20)
