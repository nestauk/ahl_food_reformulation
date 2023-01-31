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

# %%
# # Clustering households based on purchasing activity

# Import libraries and directory
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.analysis.medium_article import (
    functions_medium_article as medium,
    get_data,
)
from ahl_food_reformulation.analysis.medium_article.plotting_style import (
    configure_plots,
)
import logging
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import time
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import altair as alt

# Get data
logging.info("loading data")
purch_recs_subset = get_data.purchase_subsets(202111)
nut_subset = get_data.nutrition_subsets(202111)
prod_mast = get_data.product_master()
val_fields = get_data.val_fields()
uom = get_data.uom()
prod_codes = get_data.product_codes()
prod_vals = get_data.product_values()

# %% [markdown]
# ### Creating representations

# %%
logging.info("Create representations")
# Scaler
scaler = MinMaxScaler()
# Add volume measurement
pur_vol = medium.vol_for_purch(purch_recs_subset, val_fields, prod_mast, uom)
# Purchase and product info combined
comb_files = medium.combine_files(
    val_fields,
    pur_vol,
    prod_codes,
    prod_vals,
    2907,  # 2828 (market sector) # granular == 2907, 2829 less granular
)
# Share of household kcal per category - Make representation
kcal_share_subset = medium.hh_kcal_per_category(nut_subset, scaler, comb_files)

# %%
logging.info("Dimensionality reduction")
# Using PCA and UMAP to reduce dimensions
umap_share_sub = medium.dimension_reduction(kcal_share_subset, 0.97)

# %% [markdown]
# ### K-means clustering

# %%
logging.info("Apply kmeans to representations")
# Get silhoutte scores for different numbers of k
no_clusters = [10, 20, 30, 40, 50, 60, 70, 80]
scores_share = medium.kmeans_score_list(no_clusters, umap_share_sub)

# %%
# Plot results
fig = plt.figure(figsize=(7, 5))

sns.scatterplot(x=no_clusters, y=scores_share, color="#0000FF")
sns.lineplot(x=no_clusters, y=scores_share, color="#0000FF")


plt.scatter(
    x=no_clusters[np.argmax(scores_share)],
    y=max(scores_share),
    color="#0000FF",
    alpha=1,
)

plt.text(
    (no_clusters[np.argmax(scores_share)] - 1),
    (max(scores_share) - 0.008),
    max(scores_share).round(3),
)


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
# ### Lets look at the clusters on a 2d map

# %%
### Testing different representations using kmeans
logging.info("Plot kmeans representations")

# %%
# Picking best performing number of clusters from previous test
share_clust_num = no_clusters[np.argmax(scores_share)]

# %%
logging.info("Share of kcal")
# Get kmeans labels
_, kmeans_labels_share, clusterer = medium.kmeans_score(share_clust_num, umap_share_sub)

# %%
# Plot clusters
medium.plot_clusters(
    "Share of kcal:", share_clust_num, umap_share_sub, kmeans_labels_share, clusterer
)

# %% [markdown]
# ### Looking at the best seperated clusters

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
purch_recs_comb = medium.make_purch_records(nut_subset, comb_files, ["att_vol"])
kcal_total = medium.hh_kcal_per_prod(purch_recs_comb, "Energy KCal")

kcal_total["label"] = list(kmeans_labels_share)
kcal_total_cl = kcal_total.groupby(["label"]).sum()
kcal_perc_cat = (kcal_total_cl.div(kcal_total_cl.T.sum(), axis=0)) * 100
total_per_cat = (kcal_total_cl.sum() / kcal_total_cl.sum().sum()) * 100

food_cats_avg_cl = (kcal_perc_cat - total_per_cat).T

# %%
# Cluster score and size
clust_counts = kcal_total.label.value_counts().reset_index()
clust_counts.columns = ["clusters", "households"]
clust_scores_counts = clust_s_scores.merge(clust_counts, on="clusters")

# %%
# Plotting cluster score and size
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
top_clust = list(source.clusters)

# %%
kcal_top = kcal_total[kcal_total.label.isin(top_clust)]

# %%
# PREVIOUS VERSION - V1

# %%
# Test for significant features in clusters - skip and read in file in next cell if short of time
t_test_kcal_share = kcal_share_subset.copy()
t_test_kcal_share["label"] = list(kmeans_labels_share)

t_tests = []
for feature in t_test_kcal_share.columns:
    f = []
    for i in range(0, 70):
        data1 = t_test_kcal_share[t_test_kcal_share["label"] == i][feature]
        data2 = t_test_kcal_share[t_test_kcal_share["label"] != i][feature]
        # compare samples
        stat, p = ttest_ind(data1, data2, random_state=1)
        # interpret
        f.append(p)
    t_tests.append(f)

t_tests_df = pd.DataFrame(t_tests).transpose()  # .round(3)
t_tests_df.columns = t_test_kcal_share.columns
t_tests_df_final = t_tests_df.drop(columns=["label"])
t_tests_df_final.to_csv(PROJECT_DIR / "outputs/data/t_tests_features2.csv")

# %%
# Read in t-test results
# t_tests_df_final = get_data.t_tests()
t_tests_df_final = pd.read_csv(PROJECT_DIR / "outputs/data/t_tests_features2.csv")
t_tests_df_final.drop(columns=["Unnamed: 0"], inplace=True)

# %% [markdown]
# ### Significance test update - Bonferroni correction

# %%
# Create one list of all t-test values
ttest_list = [item for sublist in t_tests_df_final.values.tolist() for item in sublist]

# %%
# Get adjusted p-values using the bonferonni correction
import statsmodels.stats.multitest

(
    decision,
    adj_pvals,
    sidak_aplha,
    bonf_alpha,
) = statsmodels.stats.multitest.multipletests(
    pvals=ttest_list, alpha=0.05, method="bonferroni"
)

# %%
# Look at results
adj_pvals

# %%
# Re-assign adjusted t-test list back to dataframe
chunks = [adj_pvals[x : x + 1563] for x in range(0, len(adj_pvals), 1563)]
t_tests_df_final = pd.DataFrame(chunks, columns=t_tests_df_final.columns)

# %%
# Create lists of significant features per cluster
t_lists = []
for col in t_tests_df_final.T.columns:
    col_list = [t_tests_df_final.T[col][t_tests_df_final.T[col] < 0.05]]
    t_lists.append(col_list)
clust_sig = []
for clust in t_lists:
    clust_sig.append(len(clust[0]))

# %%
# Min and max adjusted p-values (very similar to the original counts)
print(min(clust_sig), max(clust_sig))

# %%
# Avg number of significant features
np.mean(clust_sig)

# %%
# New col in cluster info df
clust_scores_counts["significant features"] = clust_sig

# %%
# Histogram of count of significant features
fig = (
    alt.Chart(clust_scores_counts)
    .mark_bar(color="#0000FF")
    .encode(
        alt.X("significant features:Q", bin=True, title="Significant features"),
        y=alt.Y("count()", title="Count of clusters"),
    )
    .properties(width=300, height=200)
)
configure_plots(
    fig,
    "Histogram of significant features",
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
df["avg kcal share"].max()

# %%
c_num = 50
source = df[df.cluster == c_num].copy()
source["Absolute_diff"] = source["difference in share"].abs()
source = source.sort_values(by="Absolute_diff", ascending=False).head(20)

fig = (
    alt.Chart(source)
    .mark_circle()
    .encode(
        x=alt.X(
            "difference in share:Q",
            scale=alt.Scale(domain=[-4, 13]),
            axis=alt.Axis(tickCount=12),
        ),
        y=alt.Y(
            "category:N",
            sort=alt.EncodingSortField(field="difference in share", order="descending"),
        ),
        size=alt.Size(
            "pvalue:Q",
            scale=alt.Scale(
                reverse=True,
                domain=[0, 0.05],
            ),
        ),
        color=alt.Color(
            "avg kcal share:Q",
            scale=alt.Scale(range=["#e0afbc", "#EB003B"], domain=[0, 12]),
        ),
    )
)

configure_plots(
    fig,
    "Cluster "
    + str(c_num)
    + ": Biggest differences in category shares compared to the average",
    "",
    16,
    14,
    14,
)

# %%
# Cluster 24 table
source.sort_values(by="difference in share", ascending=False)

# %%
c_num = 49
source = df[df.cluster == c_num].copy()
source["Absolute_diff"] = source["difference in share"].abs()
source = source.sort_values(by="Absolute_diff", ascending=False).head(20)

fig = (
    alt.Chart(source)
    .mark_circle()
    .encode(
        x=alt.X(
            "difference in share:Q",
            scale=alt.Scale(domain=[-4, 13]),
            axis=alt.Axis(tickCount=12),
        ),
        y=alt.Y(
            "category:N",
            sort=alt.EncodingSortField(field="difference in share", order="descending"),
        ),
        size=alt.Size(
            "pvalue:Q",
            scale=alt.Scale(
                reverse=True,
                domain=[0, 0.05],
            ),
        ),
        color=alt.Color(
            "avg kcal share:Q",
            scale=alt.Scale(range=["#dadaf2", "#0000FF"], domain=[0, 12]),
        ),
    )
)

configure_plots(
    fig,
    "Cluster "
    + str(c_num)
    + ": Biggest differences in category shares compared to the average",
    "",
    16,
    14,
    14,
)
