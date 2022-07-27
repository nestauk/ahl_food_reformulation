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
panel_clusters = pd.read_csv(f"{PROJECT_DIR}/outputs/data/panel_clusters_v2.csv")
panel_clusters_prev = pd.read_csv(f"{PROJECT_DIR}/outputs/data/panel_clusters.csv")
nutrition = get_k.nutrition()
demog_hh = get_k.household_demog()
panel_weights = pd.read_csv(
    f"{PROJECT_DIR}/inputs/data/panel_demographic_weights_period.csv"
)

panel_weights = panel_weights[panel_weights["purchase_period"] == 202110].copy()
cluster_weights = panel_clusters.merge(
    panel_weights, left_on="Panel Id", right_on="panel_id", how="left"
)
cluster_weights.drop(["panel_id"], axis=1, inplace=True)
cluster_weights_size = (
    cluster_weights.groupby(["clusters"])["demographic_weight"].sum().reset_index()
)

# Plot total households per cluster (unweighted)
sns.set(rc={"figure.figsize": (20, 3)}, style="white")
cluster_counts = (
    panel_clusters["clusters"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "Cluster", "clusters": "Households"})
)
cluster_counts_prev = (
    panel_clusters_prev["clusters"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "Cluster", "clusters": "Households"})
)
ax = sns.barplot(data=cluster_counts, x="Cluster", y="Households", color="lightblue")
sns.lineplot(
    data=cluster_counts_prev,
    x="Cluster",
    y="Households",
    color="black",
    lw=3,
    ls="--",
    label="previous clusters",
    ax=ax,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.tick_params(axis="both", which="major", labelsize=20)
plt.ylabel("Households", fontsize=25)
plt.xlabel("Cluster", fontsize=25)
plt.title("Total households in each cluster (unweighted)", fontsize=25)
plt.show()

# Plot total households per cluster (weighted)
sns.set(rc={"figure.figsize": (20, 3)}, style="white")
ax = sns.barplot(
    data=cluster_weights_size, x="clusters", y="demographic_weight", color="lightblue"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.tick_params(axis="both", which="major", labelsize=20)
plt.ylabel("Households", fontsize=25)
plt.xlabel("Cluster", fontsize=25)
plt.title("Total households in each cluster (weighted)", fontsize=25)
plt.show()

# Get percentage per cluster
cluster_counts["households_unweighted"] = (
    cluster_counts["Households"] / cluster_counts["Households"].sum()
) * 100
cluster_weights_size["households_weighted"] = (
    cluster_weights_size["demographic_weight"]
    / cluster_weights_size["demographic_weight"].sum()
) * 100

# Plot percentage of households per cluster - weighted vs unweighted
cluster_weights_size.merge(
    cluster_counts, left_on="clusters", right_on="Cluster", how="left"
)[["clusters", "households_weighted", "households_unweighted"]].set_index(
    "clusters"
).plot(
    kind="barh", figsize=(10, 10)
)
plt.title(
    "Proportion of households in each cluster - weighted compared to un-weighted",
    fontsize=16,
    pad=15,
)
plt.xlabel("Percentage", fontsize=14)
plt.ylabel("Cluster", fontsize=14)
