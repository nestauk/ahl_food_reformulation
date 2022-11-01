# Import libraries
import pandas as pd
import numpy as np
import altair as alt
from scipy import stats
import statsmodels.stats.multicomp as mc
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from altair_saver import save
import matplotlib.pyplot as plt
import seaborn as sns

from ahl_food_reformulation.getters import kantar as get_k
from ahl_food_reformulation.utils import lookups as lps
from ahl_food_reformulation.pipeline import transform_data as td
from ahl_food_reformulation import PROJECT_DIR

# Read in data
panel_clusters = pd.read_csv(f"{PROJECT_DIR}/outputs/data/panel_clusters.csv")
prod_mast = get_k.product_master()
val_fields = get_k.val_fields()
uom = get_k.uom()
prod_codes = get_k.product_codes()
prod_vals = get_k.product_values()
prod_att = get_k.product_attribute()
purch_recs = get_k.purchase_subsets(202110)
nutrition = get_k.nutrition()


def cluster_category_subset(purch_prods, cluster_subset, food_cat):
    """Get total kcal purchased per select clusters and food category"""
    nut_clust = purch_prods[
        ["clusters", "RST 4 Market Sector", "Energy KCal", "RST 4 Extended"]
    ].copy()  # select cols
    cat_subset = nut_clust[nut_clust.clusters.isin(cluster_subset)].copy()
    cat_subset = cat_subset[cat_subset["RST 4 Market Sector"] == food_cat].copy()
    data = (
        (
            (
                cat_subset.groupby(["clusters", "RST 4 Extended"])["Energy KCal"].sum()
                / cat_subset.groupby(["clusters"])["Energy KCal"].sum()
            )
        )
        * 100
    ).reset_index()
    # Top 30
    top_items = list(
        cat_subset.groupby(["RST 4 Extended"])["Energy KCal"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .index
    )
    data = data[data["RST 4 Extended"].isin(top_items)].copy()
    return data


# Create combined purchase table
product_info = lps.product_table(
    val_fields, prod_mast, uom, prod_codes, prod_vals, prod_att
)
purch_prods = purch_recs.merge(product_info, how="left", on="Product Code").merge(
    panel_clusters, how="left", on="Panel Id"
)

nut_subset = nutrition[nutrition["Purchase Period"] == 202110].copy()

nut_subset["pur_id"] = (
    nut_subset["Purchase Number"].astype(str)
    + "_"
    + nut_subset["Purchase Period"].astype(str)
)

purch_prods["pur_id"] = (
    purch_prods["PurchaseId"].astype(str) + "_" + purch_prods["Period"].astype(str)
)
nut_subset = nut_subset[
    [
        "pur_id",
        "Energy KJ",
        "Energy KCal",
        "Protein KG",
        "Carbohydrate KG",
        "Sugar KG",
        "Fat KG",
        "Saturates KG",
        "Fibre KG Flag",
        "Sodium KG",
    ]
].copy()

purch_prods = purch_prods.merge(nut_subset, on="pur_id", how="left")

## Plot total kcal purchased per 'RST 4 Market Sector'
plt.figure(figsize=(10, 10))
(
    (
        purch_prods.groupby(["RST 4 Market Sector"])["Energy KCal"].sum()
        / purch_prods["Energy KCal"].sum()
    )
    * 100
).sort_values().plot(kind="barh", color="#0000ffff")
plt.xlabel("Percentage", fontsize=14)
plt.ylabel("Food categories", fontsize=14)
plt.title("Percent of total kcal purchased per Food category", fontsize=16)
plt.show()

## Plot breakdown of kcal purchased for diary products
plt.figure(figsize=(2, 10))
dairy = (
    (
        (
            purch_prods[purch_prods["RST 4 Market Sector"] == "Dairy Products"]
            .groupby(["RST 4 Market Sector", "RST 4 Market"])["Energy KCal"]
            .sum()
            / purch_prods[purch_prods["RST 4 Market Sector"] == "Dairy Products"][
                "Energy KCal"
            ].sum()
        )
        * 100
    )
    .sort_values()
    .reset_index()
)
data = pd.DataFrame(dairy.set_index("RST 4 Market")["Energy KCal"]).T
ax = data.plot.bar(stacked=True, figsize=(2, 10))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.legend(bbox_to_anchor=(0.8, 0.7))
plt.title("Percent kcal purchased in 'Dairy Products' product category", size=20)

## Plot breakdown of kcal purchased for fruit/veg/salad category
fruit_veg_salads = cluster_category_subset(purch_prods, [1, 4, 9], "Fruit+Veg+Salads")
sns.set(rc={"figure.figsize": (25, 8)})
sns.set(font_scale=1.8)
sns.set_style("white")

sns.barplot(x="RST 4 Extended", y="Energy KCal", hue="clusters", data=fruit_veg_salads)
plt.xticks(rotation=90)
plt.title("Distribution of Kcal across the 'Fruit+Veg+Salads' category")
plt.ylabel("Percentage of kcal purchased")
plt.xlabel("Food category")
plt.show()

## Plot breakdown of kcal purchased for alcohol category
alcohol = cluster_category_subset(purch_prods, [13, 15], "Alcohol")
sns.set(rc={"figure.figsize": (25, 8)})
sns.set(font_scale=1.8)
sns.set_style("white")

sns.barplot(x="RST 4 Extended", y="Energy KCal", hue="clusters", data=alcohol)
plt.xticks(rotation=90)
plt.title("Distribution of Kcal across the 'Alcohol' category")
plt.ylabel("Percentage of kcal purchased")
plt.xlabel("Food category")
plt.show()
