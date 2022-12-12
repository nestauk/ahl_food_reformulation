# -*- coding: utf-8 -*-
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
# # Clustering households based on purchasing activity

# %% [markdown]
# - Add intro to analysis here
# - Link back to article

# %%
# Import libraries and directory
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.pipeline import cluster_methods as cluster
from ahl_food_reformulation.analysis.medium_article import (
    functions_medium_article as medium,
)
from ahl_food_reformulation.utils.plotting import configure_plots
import logging
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import time
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import altair as alt

import shap
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
from shap import Explainer, waterfall_plot

# %% [markdown]
# ### Making household representations

# %% [markdown]
# - Brief description of:
#     - Data loaded
#     - Transformations happening and why
#         - Kcal share and adjusted share of kcal
#         - Dimensionality reduction: PCA and UMAP

# %%
# Get data
logging.info("loading data")
purch_recs_subset = kantar.purchase_subsets(202111)
nut_subset = kantar.nutrition_subsets(202111)
pan_ind = kantar.household_ind()
prod_mast = kantar.product_master()
val_fields = kantar.val_fields()
uom = kantar.uom()
prod_codes = kantar.product_codes()
prod_vals = kantar.product_values()

# %%
logging.info("Create representations")
# Scaler
scaler = MinMaxScaler()
# Converted household size
pan_conv = transform.hh_size_conv(pan_ind)
# Add volume measurement
pur_vol = transform.vol_for_purch(purch_recs_subset, val_fields, prod_mast, uom)
# Purchase and product info combined
comb_files = transform.combine_files(
    val_fields,
    pur_vol,
    prod_codes,
    prod_vals,
    2907,  # 2828 (market sector) # granular == 2907, 2829 less granular
)
# Household kcal per category adjusted for size - Make representation
kcal_adj_subset = transform.hh_kcal_volume_converted(
    nut_subset, pan_conv, scaler, comb_files
)
# Share of household kcal per category - Make representation
kcal_share_subset = transform.hh_kcal_per_category(nut_subset, scaler, comb_files)

# %%
comb_files.head(1)

# %%
nut_subset.head(1)

# %%
kcal_share_subset.drop(
    list(kcal_share_subset.filter(regex="Oil")), axis=1, inplace=True
)
kcal_share_subset.drop(
    list(kcal_share_subset.filter(regex="Rice")), axis=1, inplace=True
)
kcal_adj_subset.drop(list(kcal_adj_subset.filter(regex="Oil")), axis=1, inplace=True)
kcal_adj_subset.drop(list(kcal_adj_subset.filter(regex="Rice")), axis=1, inplace=True)

# %%
# kcal_share_subset.drop(['Cooking Oils Sunflower', 'Cooking Oils Vegetable','Cooking Oils Vegetable','Ambient Rice Bulk Plain Rice', 'Cooking Oils Other Cooking'], inplace=True, axis=1)
# kcal_adj_subset.drop(['Cooking Oils Sunflower', 'Cooking Oils Vegetable','Cooking Oils Vegetable','Ambient Rice Bulk Plain Rice', 'Cooking Oils Other Cooking'], inplace=True, axis=1)

# %%
logging.info("Dimensionality reduction")
# Using PCA and UMAP to reduce dimensions
umap_adj_sub = cluster.dimension_reduction(kcal_adj_subset, 0.97)
umap_share_sub = cluster.dimension_reduction(kcal_share_subset, 0.97)

# %% [markdown]
# ### Clustering households

# %% [markdown]
# - Describe method used and what is being tested:
#     - Kmeans
#     - Number of k and representations
# - Evaluation metric: silhoutte score

# %%
logging.info("Apply kmeans to representations")

# Get silhoutte scores for different numbers of k
no_clusters = [10, 20, 30, 40, 50, 60, 70, 80]
scores_share = cluster.kmeans_score_list(no_clusters, umap_share_sub)
scores_adj = cluster.kmeans_score_list(no_clusters, umap_adj_sub)

# %%
# Plot results
fig = plt.figure(figsize=(7, 5))

sns.scatterplot(x=no_clusters, y=scores_share, color="#0000FF")
# sns.scatterplot(x=no_clusters, y=scores_adj)
sns.lineplot(x=no_clusters, y=scores_share, color="#0000FF")
# sns.lineplot(x=no_clusters, y=scores_adj)

# fig.legend(
#    labels=["Kcal share", "Kcal adjusted for household size"],
#    bbox_to_anchor=(0.86, 0.35, 0.5, 0.5),
#    fontsize=11,
# )

plt.scatter(
    x=no_clusters[np.argmax(scores_share)],
    y=max(scores_share),
    color="#0000FF",
    alpha=1,
)
# plt.scatter(
#    x=no_clusters[np.argmax(scores_adj)], y=max(scores_adj), color="orange", alpha=1
# )

plt.text(
    (no_clusters[np.argmax(scores_share)] - 1),
    (max(scores_share) - 0.008),
    max(scores_share).round(3),
)
# plt.text(
#    (no_clusters[np.argmax(scores_adj)] - 4),
#    (max(scores_adj) - 0.012),
#    max(scores_adj).round(3),
# )


plt.xlabel("Number of clusters", fontsize=12)
plt.ylabel("Silhoutte score", fontsize=12)
plt.title(
    "Silhoutte scores for different numbers of k",
    fontsize=14,
    pad=10,
)

sns.despine()
sns.despine(top=True, right=True, left=False, bottom=False)

plt.show()

# %% [markdown]
# Results show share of k-cal at k = 60 is the best performing.

# %% [markdown]
# ### Lets look at the clusters on a 2d map

# %% [markdown]
# Plotting results. Intuitive description of what you can see.

# %%
### Testing different representations using kmeans
logging.info("Plot kmeans representations")

# %%
# Picking best performing number of clusters from previous test
share_clust_num = no_clusters[np.argmax(scores_share)]

# %%
logging.info("Share of kcal")
# Get kmeans labels
_, kmeans_labels_share, clusterer = cluster.kmeans_score(
    share_clust_num, umap_share_sub
)

# %%
# Plot clusters
medium.plot_clusters(
    "Share of kcal:", share_clust_num, umap_share_sub, kmeans_labels_share, clusterer
)

# %% [markdown]
# ### Looking at the best seperated clusters
#
# - Pick out 2-3 very well seperated clusters and compare their share of kcal of categories to the mean

# %%
sample_silhouette_values = silhouette_samples(umap_share_sub, kmeans_labels_share)

means_lst_share = []
for label in range(share_clust_num):
    means_lst_share.append(
        sample_silhouette_values[kmeans_labels_share == label].mean()
    )

# %%
clust_s_scores = pd.DataFrame(
    {"clusters": list(range(share_clust_num)), "scores": means_lst_share}
)

# %%
# Difference in household share compared to avg
purch_recs_comb = transform.make_purch_records(nut_subset, comb_files, ["att_vol"])
kcal_total = transform.hh_kcal_per_prod(purch_recs_comb, "Energy KCal")

# kcal_total.drop(['Cooking Oils Sunflower', 'Cooking Oils Vegetable','Cooking Oils Vegetable','Ambient Rice Bulk Plain Rice', 'Cooking Oils Other Cooking'], inplace=True, axis=1)
kcal_total.drop(list(kcal_total.filter(regex="Oil")), axis=1, inplace=True)
kcal_total.drop(list(kcal_total.filter(regex="Rice")), axis=1, inplace=True)


kcal_total["label"] = list(kmeans_labels_share)
kcal_total_cl = kcal_total.groupby(["label"]).sum()
kcal_perc_cat = (kcal_total_cl.div(kcal_total_cl.T.sum(), axis=0)) * 100
total_per_cat = (kcal_total_cl.sum() / kcal_total_cl.sum().sum()) * 100

# %%
food_cats_avg_cl = (kcal_perc_cat - total_per_cat).T

# %%
clust_counts = kcal_total.label.value_counts().reset_index()
clust_counts.columns = ["clusters", "households"]

# %%
# clust_counts.sort_values(by='households', ascending=True).head(10)

# %%
clust_scores_counts = clust_s_scores.merge(clust_counts, on="clusters")

# %%
clust_scores_counts.head(20)

# %%
source = clust_scores_counts.sort_values(by="scores", ascending=False).head(30)
fig = (
    alt.Chart(source)
    .mark_bar(color="#0000FF")
    .encode(
        alt.Y(
            "clusters:N", sort=alt.EncodingSortField(field="scores", order="descending")
        ),
        x="scores",
    )
)

fig2 = (
    alt.Chart(source)
    .mark_bar(color="#FDB633")
    .encode(
        alt.Y(
            "clusters:N", sort=alt.EncodingSortField(field="scores", order="descending")
        ),
        x="households",
    )
)

figures = fig | fig2

configure_plots(
    figures,
    "Top 30 clusters by silhoutte score",
    "",
    16,
    20,
    16,
)

# %%

# %%
top_clust = list(source.clusters)

# %%
kcal_top = kcal_total[kcal_total.label.isin(top_clust)]

# %%
kcal_share_subset.head(1)

# %%
t_test_kcal_share = kcal_share_subset.copy()

# %%
t_test_kcal_share["label"] = list(kmeans_labels_share)

# %%
from scipy.stats import ttest_ind

t_tests = []
for feature in t_test_kcal_share.columns:
    f = []
    for i in range(0, 60):
        data1 = t_test_kcal_share[t_test_kcal_share["label"] == i][feature]
        data2 = t_test_kcal_share[t_test_kcal_share["label"] != i][feature]
        # compare samples
        stat, p = ttest_ind(data1, data2, random_state=1)
        # interpret
        f.append(p)
    t_tests.append(f)

t_tests_df = pd.DataFrame(t_tests).transpose().round(3)
t_tests_df.columns = t_test_kcal_share.columns

# %%
# t_tests_df_final = t_tests_df.drop(columns=["label"])

# %%
# t_tests_df_final.to_csv("t_tests_final.csv")

# %%
t_tests_df_final = pd.read_csv("t_tests_final.csv")

# %%
t_tests_df_final.drop(columns=["Unnamed: 0"], inplace=True)

# %%
t_lists = []
for col in t_tests_df_final.T.columns:
    col_list = [t_tests_df_final.T[col][t_tests_df_final.T[col] < 0.05]]
    t_lists.append(col_list)

# %%
clust_sig = []
for clust in t_lists:
    clust_sig.append(len(clust[0]))

# %%
min(clust_sig)

# %%
max(clust_sig)

# %%
clust_scores_counts["significant features"] = clust_sig

# %%
clust_scores_counts.sort_values(by="scores", ascending=False).tail(20)

# %%
source = clust_scores_counts.sort_values(by="scores", ascending=False).head(30)
fig = (
    alt.Chart(source)
    .mark_bar(color="#0000FF")
    .encode(
        alt.Y(
            "clusters:N", sort=alt.EncodingSortField(field="scores", order="descending")
        ),
        x="scores",
    )
)

fig2 = (
    alt.Chart(source)
    .mark_bar(color="#FDB633")
    .encode(
        alt.Y(
            "clusters:N", sort=alt.EncodingSortField(field="scores", order="descending")
        ),
        x="households",
    )
)

fig3 = (
    alt.Chart(source)
    .mark_bar(color="#EB003B")
    .encode(
        alt.Y(
            "clusters:N", sort=alt.EncodingSortField(field="scores", order="descending")
        ),
        x="significant features",
    )
)

figures = fig | fig2 | fig3

configure_plots(
    figures,
    "Top 30 clusters by silhoutte score",
    "",
    16,
    20,
    16,
)

# %%
clust_list = []

for col in food_cats_avg_cl.columns:
    clust_list.append(
        list(food_cats_avg_cl[col].sort_values(ascending=False).head(1).index)[0]
    )

# %%
# Import libraries and directory
from array import array
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_cluster_cats(
    title: str, n_clusters: int, df: array, cluster_labels: list, clusterer, clust_list
):
    """
    Plot of clustering based on number of k and labels
    Args:
        title (str): Description of representation to add to start of plot title
        no_clusters (int): k
        df (array): Umap representation of household representations
        cluster_labels (list): List of assigned labels
    """
    fig, ax2 = plt.subplots()
    fig.set_size_inches(15, 15)

    silhouette_avg = silhouette_score(df, cluster_labels)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    y_lower = 10

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        df[:, 0],
        df[:, 1],
        marker=".",
        s=30,
        lw=0,
        alpha=0.1,
        c=colors,
        edgecolor="k",
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_

    for i, c in enumerate(centers):
        # ax2.scatter(c[0], c[1], marker="$%d$" % clust_list[i], alpha=1, s=50, edgecolor="k")
        ax2.annotate(clust_list[i], (c[0], c[1]), fontsize=12)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        title + " Silhouette analysis for clustering with n_clusters = %d" % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    plt.show()


# %%
plot_cluster_cats(
    "Share of kcal:",
    share_clust_num,
    umap_share_sub,
    kmeans_labels_share,
    clusterer,
    clust_list,
)

# %%
centers = clusterer.cluster_centers_
centers_df = pd.DataFrame(centers)
centers_df["labels"] = clust_list
centers_df.columns = ["x", "y", "label"]

# %%
centers_df["size"] = clust_scores_counts["households"]
centers_df["score"] = clust_scores_counts["scores"]

# %%
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler(feature_range=(10, 500))

centers_df[["size_adj"]] = mms.fit_transform(centers_df[["size"]])

# %%
centers_df.head(5)

# %%
import altair as alt
import pandas as pd

source = centers_df

points = (
    alt.Chart(source)
    .mark_point()
    .encode(
        x=alt.X("x:Q", axis=alt.Axis(grid=False)),
        y=alt.Y("y:Q", axis=alt.Axis(grid=False)),
        size=alt.Size("size_adj:N", scale=alt.Scale(range=[10, 20])),
        color=alt.Color("score:Q", scale=alt.Scale(scheme="reds")),
    )
)

text = points.mark_text(align="left", baseline="middle", dx=7, angle=5).encode(
    text="label"
)

fig = (points + text).properties(width=900, height=900)

# %%
configure_plots(
    fig,
    "",
    "",
    16,
    20,
    16,
)


# %%
def difference_shares_plot(df, clust_num):
    source = df[[clust_num]].copy()
    source["Absolute_diff"] = source[clust_num].abs()
    source = source.sort_values(by="Absolute_diff", ascending=False).head(10)
    source.reset_index(inplace=True)
    source.columns = ["Category", "Percent_difference", "Absolute_diff"]

    fig = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            alt.X(
                "Category:N",
                sort=alt.EncodingSortField(field="Absolute_diff", order="descending"),
            ),
            y="Percent_difference:Q",
            color=alt.condition(
                alt.datum.Percent_difference > 0,
                alt.value("#0000FF"),  # The positive color
                alt.value("orange"),  # The negative color
            ),
        )
        .properties(width=600)
    )
    return fig, source


# %%
df = pd.DataFrame()
for i in range(0, 60):
    clust = pd.concat([t_lists[i][0], food_cats_avg_cl[i], kcal_perc_cat.T[i]], axis=1)
    clust.reset_index(inplace=True)
    clust["cluster"] = i
    clust.columns = [
        "category",
        "pvalue",
        "difference in share",
        "avg kcal share",
        "cluster",
    ]
    clust.dropna(inplace=True)
    df = pd.concat([df, clust])

# %%
source["avg kcal share"].max()

# %%
source

# %%
source = df[df.cluster == 21].copy()
source["Absolute_diff"] = source["difference in share"].abs()
source = source.sort_values(by="Absolute_diff", ascending=False).head(20)

fig = (
    alt.Chart(source)
    .mark_circle()
    .encode(
        x="difference in share:Q",
        y=alt.Y(
            "category:N",
            sort=alt.EncodingSortField(field="difference in share", order="descending"),
        ),
        size=alt.Size(
            "pvalue:Q",
            scale=alt.Scale(
                reverse=True,
                domain=[source["pvalue"].min(), source["pvalue"].max() + 0.003],
            ),
        ),
        color=alt.Color(
            "avg kcal share:Q",
            scale=alt.Scale(
                range=["#e0afbc", "#EB003B"], domain=[0, source["avg kcal share"].max()]
            ),
        ),
    )
)

configure_plots(
    fig,
    "Cluster 21: Biggest differences in category shares compared to the average",
    "",
    16,
    14,
    14,
)

# %%
source = df[df.cluster == 34].copy()
source["Absolute_diff"] = source["difference in share"].abs()
source = source.sort_values(by="Absolute_diff", ascending=False).head(20)

fig = (
    alt.Chart(source)
    .mark_circle()
    .encode(
        x="difference in share:Q",
        y=alt.Y(
            "category:N",
            sort=alt.EncodingSortField(field="difference in share", order="descending"),
        ),
        size=alt.Size(
            "pvalue:Q",
            scale=alt.Scale(
                reverse=True,
                domain=[source["pvalue"].min(), source["pvalue"].max() + 0.003],
            ),
        ),
        color=alt.Color(
            "avg kcal share:Q",
            scale=alt.Scale(
                range=["#dadaf2", "#0000FF"], domain=[0, source["avg kcal share"].max()]
            ),
        ),
    )
)

configure_plots(
    fig,
    "Cluster 34: Biggest differences in category shares compared to the average",
    "",
    16,
    14,
    14,
)

# %%
source.sort_values(by="pvalue").head(40)

# %%
source

# %%
source = df[df.cluster == 16].copy()
source = source.sort_values(by="pvalue").head(40)

alt.Chart(source).mark_circle().encode(
    x="difference in share:Q",
    y="category:N",
    size=alt.Size("pvalue:Q", scale=alt.Scale(reverse=True)),
    color=alt.Color("avg kcal share:Q", scale=alt.Scale(scheme="reds")),
)

# %%
clust_12.head(1)

# %%
import altair as alt
import pandas as pd

source = clust_12

points = (
    alt.Chart(source)
    .mark_point()
    .encode(
        x=alt.X("avg kcal share:Q", axis=alt.Axis(grid=False)),
        y=alt.Y("pvalue:Q", axis=alt.Axis(grid=False)),
    )
)


fig = (points).properties(width=900, height=900)

configure_plots(
    fig,
    "",
    "",
    16,
    20,
    16,
)

# %%
fig, source = difference_shares_plot(food_cats_avg_cl, 12)

configure_plots(
    fig,
    "Cluster 12: Biggest differences in category shares compared to the average",
    "",
    16,
    20,
    16,
)

# %%
source

# %%
fig, source = difference_shares_plot(food_cats_avg_cl, 21)

configure_plots(
    fig,
    "Cluster 21: Biggest differences in category shares compared to the average",
    "",
    16,
    20,
    16,
)

# %%
fig, _ = difference_shares_plot(food_cats_avg_cl, 16)

configure_plots(
    fig,
    "Cluster 16: Biggest differences in category shares compared to the average",
    "",
    16,
    20,
    16,
)

# %%
fig, _ = difference_shares_plot(food_cats_avg_cl, 3)

configure_plots(
    fig,
    "Cluster 3: Biggest differences in category shares compared to the average",
    "",
    16,
    20,
    16,
)

# %%

# %% [markdown]
# ### Using SHAP to understand the clusters

# %%
# K-means labels and kcal share df
y = kmeans_labels_share
x = kcal_share_subset.copy()

# %%
# Train and fit a classifier
clf = RandomForestClassifier()
# To speed up / test code run with ".head(100)"
clf.fit(x, y)

# %%
# Sample size to create SHAP values (for speed)
sample_size = 100

# %%
# Creating SHAP values (takes a while to run)
explainer = Explainer(clf)
sv_red = explainer.shap_values(x.head(sample_size), check_additivity=False)

# %%
shap.summary_plot(sv_red, x.head(sample_size), plot_type="bar")

# %%
# Subset SHAP values by only selection highlighted in previous section
# label_sub = [10, 4, 33, 27, 52, 49]
label_sub = [12]
sv_cls = [sv_red[i] for i in label_sub]
expl = Explanation(
    sv_red,
    feature_names=x.head(sample_size).columns,
)

# %%
x.head(sample_size).columns

# %%
fig, ax = plt.subplots(1, 1)
fig.legend(loc="upper right")

# Summary plot based on select clusters
shap.summary_plot(
    expl,
    x.head(sample_size).values,
    max_display=10,
    feature_names=x.head(sample_size).columns,
    # class_names=["12"],
)

fig.legend(loc="upper right")

# %%
# Beeswarm plot of cluster 4
# shap.plots.beeswarm(expl[0])

# %%
# Slicing by single cluster
cls = 3  # class to explain
sv_cls = sv_red[cls]

expl = Explanation(
    sv_cls,
    explainer.expected_value[cls],
    feature_names=x.head(sample_size).columns,
)

# %%
# Summary plot of cluster 4
shap.summary_plot(
    expl,
    x.head(sample_size).values,
    max_display=20,
    feature_names=x.head(sample_size).columns,
    plot_type="bar",
)

# %%
# Summary plot of cluster 4
shap.summary_plot(
    expl,
    x.head(sample_size).values,
    max_display=20,
    feature_names=x.head(sample_size).columns,
)

# %%
# Looking at single cluster AND single observation
cls = 12  # class to explain
sv_cls = sv_red[cls]
idx = 0  # Household / observation

waterfall_plot(
    Explanation(
        sv_cls[idx],
        explainer.expected_value[cls],
        feature_names=x.head(sample_size).columns,
    )
)

# %%

# %%
