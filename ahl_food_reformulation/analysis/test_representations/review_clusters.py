# Import libraries
import pandas as pd
from ahl_food_reformulation.pipeline import transform_data as td
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# Import project libraries and directory
from ahl_food_reformulation.getters import kantar as get_k
from ahl_food_reformulation import PROJECT_DIR

# Read in files
purch_recs = get_k.purchase_subsets(202110)  # Oct 2021 subset
prod_mast = get_k.product_master()
val_fields = get_k.val_fields()
uom = get_k.uom()
prod_codes = get_k.product_codes()
prod_vals = get_k.product_values()
panel_clusters = pd.read_csv(f"{PROJECT_DIR}/outputs/data/panel_clusters.csv")
nutrition = get_k.nutrition()
demog_hh = get_k.household_demog()

# Combine and merge files
purch_recs = td.combine_files(
    val_fields, purch_recs, prod_mast, uom, prod_codes, prod_vals, 2827
)
nut_subset = nutrition[nutrition["Purchase Period"] == 202110].copy()
nut_subset["pur_id"] = (
    nut_subset["Purchase Number"].astype(str)
    + "_"
    + nut_subset["Purchase Period"].astype(str)
)

purch_recs["pur_id"] = (
    purch_recs["PurchaseId"].astype(str) + "_" + purch_recs["Period"].astype(str)
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
purch_recs = purch_recs.merge(nut_subset, on="pur_id", how="left")
purch_recs = purch_recs.merge(panel_clusters, on="Panel Id", how="left")

# Create subset dfs per cluster of total Kcal
cluster_subsets = [
    purch_recs[purch_recs["clusters"] == i]
    for i in range(0, purch_recs["clusters"].nunique())
]

# Demographic information
purch_dem = purch_recs.merge(demog_hh, on="Panel Id", how="left")
# Unique households per cluster
hh_clusters = purch_dem.drop_duplicates(subset=["Panel Id", "clusters"]).copy()

### BMI
# Has high BMI in household and percentage of high BMI
hh_clusters["BMI high in hh"] = np.where(hh_clusters["high_bmi"] > 0, 1, 0)
hh_clusters["% high BMI"] = hh_clusters["high_bmi"] / hh_clusters["Household Size"]
hh_clusters["BMI over 60%"] = np.where(hh_clusters["% high BMI"] > 0.6, 1, 0)
# Remove cases where BMI is 0 (missing)
bmi_non_miss = hh_clusters[hh_clusters["bmi_missing"] == 0].copy()

# High BMI (% of household)
perc_bmi_high = td.percent_demog_group(hh_clusters, "BMI high in hh", "clusters")
perc_bmi_high[perc_bmi_high["BMI high in hh"] == 1].set_index("clusters")[
    "Percent"
].sort_values().plot(kind="barh", figsize=(10, 10))
plt.title(
    "Percent of households per cluster with a high BMI member (25 or over)", pad=10
)
plt.xlabel("Percentage", fontsize=10)
plt.ylabel("Clusters", fontsize=10)

# 60% or higher - high BMI in household
perc_bmi_60 = td.percent_demog_group(hh_clusters, "BMI over 60%", "clusters")
perc_bmi_60[perc_bmi_60["BMI over 60%"] == 1].set_index("clusters")[
    "Percent"
].sort_values().plot(kind="barh", figsize=(10, 10))
plt.title(
    "Percent of households per cluster where more than 60% of the household has high BMI",
    pad=10,
)
plt.xlabel("Percentage", fontsize=10)
plt.ylabel("Clusters", fontsize=10)

# Look at highest BMI cluster
sns.displot(data=bmi_non_miss[bmi_non_miss["clusters"] == 18], x="% high BMI", kde=True)

# Compared to the lowest
sns.displot(data=bmi_non_miss[bmi_non_miss["clusters"] == 13], x="% high BMI", kde=True)

### Kcal top categories per cluster

# high BMI = 18, 12, 5, 0, 15
# low BMI = 13, 8, 14, 17, 2

cluster_totals = [td.total_nutrition_intake(i) for i in cluster_subsets]
total_cats = td.total_nutrition_intake(purch_recs)

clusters = [18, 12, 5, 0, 15, 13, 8, 14, 17, 2]

total_cats = pd.concat(
    [
        total_cats["Energy KCal"],
        cluster_totals[18]["Energy KCal"],
        cluster_totals[12]["Energy KCal"],
        cluster_totals[5]["Energy KCal"],
        cluster_totals[0]["Energy KCal"],
        cluster_totals[15]["Energy KCal"],
        cluster_totals[13]["Energy KCal"],
        cluster_totals[8]["Energy KCal"],
        cluster_totals[14]["Energy KCal"],
        cluster_totals[17]["Energy KCal"],
        cluster_totals[2]["Energy KCal"],
    ],
    axis=1,
)

total_cats.columns = [["total KCal"] + [f"c{n} KCal" for n in clusters]]

# Percentage of category contribution per cluster
for col in total_cats.columns:
    total_cats[col] = (total_cats[col] / total_cats[col].sum()) * 100

total_cats.sort_values(by="total KCal", ascending=False).head(25).sort_values(
    by="total KCal", ascending=True
).loc[:, total_cats.columns != "total KCal"].plot(kind="barh", figsize=(10, 27))
plt.title(
    "Percentage kcal purchased in a cluster per category (top 10 kcal purchased categories)",
    fontsize=14,
    pad=15,
)
plt.xlabel("Percentage purchased per cluster", fontsize=12)
plt.ylabel("Category", fontsize=12)
plt.savefig(f"{PROJECT_DIR}/outputs/figures/cluster-kcal-purchased per category")

plt.show(block=False)

# Top food categories of highest BMI clusters
total_cats["c18 KCal"].sort_values(ascending=False).head(20).sort_values(
    ascending=True
).plot(kind="barh", figsize=(10, 15))
plt.title(
    "Percentage kcal purchased in cluster 18 per category (top 20)",
    fontsize=14,
    pad=15,
)
plt.xlabel("Percentage", fontsize=12)
plt.ylabel("Category", fontsize=12)

# Top food categories of highest BMI clusters
total_cats["c12 KCal"].sort_values(ascending=False).head(20).sort_values(
    ascending=True
).plot(kind="barh", figsize=(10, 15))
plt.title(
    "Percentage kcal purchased in cluster 12 per category (top 20)",
    fontsize=14,
    pad=15,
)
plt.xlabel("Percentage", fontsize=12)
plt.ylabel("Category", fontsize=12)

# Top food categories of lowest BMI cluster
total_cats["c8 KCal"].sort_values(ascending=False).head(20).sort_values(
    ascending=True
).plot(kind="barh", figsize=(10, 15))
plt.title(
    "Percentage kcal purchased in cluster 8 per category (top 20)",
    fontsize=14,
    pad=15,
)
plt.xlabel("Percentage", fontsize=12)
plt.ylabel("Category", fontsize=12)

### Main shoppper Age

# Cluster 18
sns.displot(
    data=hh_clusters[hh_clusters["clusters"] == 18],
    x="Main Shopper Age",
    kind="kde",
    palette="tab10",
)

# Cluster 13
sns.displot(
    data=hh_clusters[hh_clusters["clusters"] == 13],
    x="Main Shopper Age",
    kind="kde",
    palette="tab10",
)

print(hh_clusters[hh_clusters["clusters"] == 18]["Main Shopper Age"].mean())
print(hh_clusters[hh_clusters["clusters"] == 13]["Main Shopper Age"].mean())


# Create main shopper age categories
bins = [0, 20, 30, 40, 50, 60, 70, 120]
labels = ["teen", "20s", "30", "40s", "50s", "60s", "70 plus"]
hh_clusters["Main shopper age group"] = pd.cut(
    hh_clusters["Main Shopper Age"], bins=bins, labels=labels, right=False
)

# Main shopper age group
perc_age_groups = td.percent_demog_group(
    hh_clusters, "Main shopper age group", "clusters"
)
sns.catplot(
    data=perc_age_groups,
    col="clusters",
    x="Main shopper age group",
    y="Percent",
    palette="tab10",
    col_wrap=3,
    kind="bar",
)

### Number of Children
hh_clusters["Children"] = np.where(hh_clusters["Number of Children"] > 0, 1, 0)
hh_clusters["Children"].value_counts()

# Has Children
child = td.percent_demog_group(hh_clusters, "Children", "clusters")

# Number of children
perc_num_child = td.percent_demog_group(hh_clusters, "Number of Children", "clusters")
sns.catplot(
    data=perc_num_child,
    col="clusters",
    x="Number of Children",
    y="Percent",
    palette="tab10",
    col_wrap=3,
    kind="bar",
)

### Region
perc_region = td.percent_demog_group(hh_clusters, "Region", "clusters")
sns.catplot(
    data=perc_region,
    col="clusters",
    x="Region",
    y="Percent",
    palette="tab10",
    col_wrap=3,
    kind="bar",
)

### Household income
perc_income = td.percent_demog_group(hh_clusters, "Household Income", "clusters")

g = sns.relplot(
    data=perc_income,
    col="clusters",
    x="Household Income",
    y="Percent",
    palette="tab10",
    col_wrap=3,
    kind="line",
)
g.set_xticklabels(rotation=90)

high = [18, 12, 5, 0, 15]
low = [13, 8, 14, 17, 2]
hh_high = hh_clusters[hh_clusters["clusters"].isin(high)].copy()
hh_low = hh_clusters[hh_clusters["clusters"].isin(low)].copy()

### Life stage

# Life stage - low BMI
perc_life_stage = td.percent_demog_group(hh_low, "Life Stage", "clusters")

g = sns.catplot(
    data=perc_life_stage,
    col="clusters",
    x="Life Stage",
    y="Percent",
    palette="tab10",
    col_wrap=3,
    kind="bar",
)
g.set_xticklabels(rotation=90)

# Life stage - high BMI
perc_life_stage = td.percent_demog_group(hh_high, "Life Stage", "clusters")

g = sns.catplot(
    data=perc_life_stage,
    col="clusters",
    x="Life Stage",
    y="Percent",
    palette="tab10",
    col_wrap=3,
    kind="bar",
)
g.set_xticklabels(rotation=90)

### Alcohol content

# Remove total column
total_cats.drop(["total KCal"], axis=1, inplace=True)

# Beer
alchohol_list = ["Beer+Lager", "Cider"]

total_cats.loc[total_cats.index.isin(alchohol_list)].T.sort_values(
    by="Beer+Lager"
).plot(kind="barh", figsize=(10, 15))
plt.title("Kcal consumed from Beer and Cider purchased per cluster", fontsize=14)
plt.xlabel("Percentage", fontsize=12)
plt.ylabel("Cluster", fontsize=12)

# Wine
alchohol_list = ["Wine", "Spirits"]

total_cats.loc[total_cats.index.isin(alchohol_list)].T.sort_values(by="Wine").plot(
    kind="barh", figsize=(10, 15)
)
plt.title("Kcal consumed from Wine and Spirits purchased per cluster", fontsize=14)
plt.xlabel("Percentage", fontsize=12)
plt.ylabel("Cluster", fontsize=12)
