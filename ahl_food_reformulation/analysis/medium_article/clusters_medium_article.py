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

# %%
# Import libraries and directory
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.pipeline import cluster_methods as cluster
from ahl_food_reformulation.analysis.medium_article import (
    functions_medium_article as medium,
)
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
import shap

# %% [markdown]
# ### Making household representations

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
logging.info("Dimensionality reduction")
# Using PCA and UMAP to reduce dimensions
umap_adj_sub = cluster.dimension_reduction(kcal_adj_subset, 0.97)
umap_share_sub = cluster.dimension_reduction(kcal_share_subset, 0.97)

# %% [markdown]
# ### Clustering households

# %%
logging.info("Apply kmeans to representations")

# Get silhoutte scores for different numbers of k
no_clusters = [10, 20, 30, 40, 50, 60, 70, 80]
scores_share = cluster.kmeans_score_list(no_clusters, umap_share_sub)
scores_adj = cluster.kmeans_score_list(no_clusters, umap_adj_sub)

# %%
# Plot results
fig = plt.figure(figsize=(7, 5))
sns.scatterplot(x=no_clusters, y=scores_share)
sns.scatterplot(x=no_clusters, y=scores_adj)
sns.lineplot(x=no_clusters, y=scores_share)
sns.lineplot(x=no_clusters, y=scores_adj)

fig.legend(
    labels=["Kcal share", "Kcal adjusted for household size"],
    bbox_to_anchor=(0.86, 0.35, 0.5, 0.5),
    fontsize=11,
)

plt.scatter(
    x=no_clusters[np.argmax(scores_share)], y=max(scores_share), color="b", alpha=1
)
plt.scatter(
    x=no_clusters[np.argmax(scores_adj)], y=max(scores_adj), color="orange", alpha=1
)

plt.text(
    (no_clusters[np.argmax(scores_share)] - 1),
    (max(scores_share) - 0.012),
    max(scores_share).round(3),
)
plt.text(
    (no_clusters[np.argmax(scores_adj)] - 4),
    (max(scores_adj) - 0.012),
    max(scores_adj).round(3),
)


plt.xlabel("Number of clusters", fontsize=12)
plt.ylabel("Silhoutte score", fontsize=12)
plt.title(
    "Silhoutte scores for adjusted and share of kcal representations",
    fontsize=14,
    pad=10,
)
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
# - Do the above, more simply for all clusters

# %%
from sklearn.metrics import silhouette_samples, silhouette_score

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
clust_s_scores.sort_values(by="scores", ascending=False).head(10)

# %%
# Plot results
fig = plt.figure(figsize=(15, 5))
sns.barplot(x=list(range(share_clust_num)), y=means_lst_share)
plt.show()

# %%
food_cats_labels = kcal_share_subset.copy()

# %%
food_cats_labels["label"] = list(kmeans_labels_share)

# %%
food_cats_avg_cl = (
    (food_cats_labels.groupby(["label"]).mean() - food_cats_labels.mean()).T
)[:-1]

# %% [markdown]
# #### What categories is c27 different to the mean?

# %%
# Buying more calories of...
food_cats_avg_cl[27].sort_values(ascending=False).head(10)

# %%
# Buying less calories of...
food_cats_avg_cl[27].sort_values(ascending=True).head(10)

# %% [markdown]
# #### What categories is c33 different to the mean?

# %%
# Buying more calories of...
food_cats_avg_cl[33].sort_values(ascending=False).head(10)

# %%
# Buying less calories of...
food_cats_avg_cl[33].sort_values(ascending=True).head(10)

# %% [markdown]
# #### What categories is c52 different to the mean?

# %%
# Buying more calories of...
food_cats_avg_cl[52].sort_values(ascending=False).head(10)

# %%
# Buying less calories of...
food_cats_avg_cl[52].sort_values(ascending=True).head(10)

# %% [markdown]
# #### What categories is c49 different to the mean?

# %%
# Buying more calories of...
food_cats_avg_cl[49].sort_values(ascending=False).head(10)

# %%
# Buying less calories of...
food_cats_avg_cl[49].sort_values(ascending=True).head(10)

# %% [markdown]
# 10 and 4 are two large clusters far from the center (based on visual)...

# %%
# Buying more calories of...
food_cats_avg_cl[10].sort_values(ascending=False).head(10)

# %%
# Buying less calories of...
food_cats_avg_cl[10].sort_values(ascending=True).head(10)

# %%
# Buying more calories of...
food_cats_avg_cl[4].sort_values(ascending=False).head(10)

# %%
# Buying less calories of...
food_cats_avg_cl[4].sort_values(ascending=True).head(10)

# %% [markdown]
# #### Plot....

# %%
food_cats_labels.mean().sort_values(ascending=False)

# %%
food_cats_labels.groupby(["label"]).mean().T.head(1)

# %%

# %%

# %% [markdown]
# ### Deep dive into best performing clusters

# %% [markdown]
# #### Testing SHAP values on a sample

# %%
from shap import TreeExplainer, Explanation
from shap.plots import waterfall

# %%
# Using share of kcal clusters / transformations
# kmeanModel = KMeans(n_clusters=3)
# y = kmeanModel.fit(umap_share_sub).labels_
# y = label_binarize(y, classes=list(range(3)))

# %%
# K-means labels
y = kmeans_labels_share

# %%
# import xgboost
# clf = xgb.XGBClassifier()

# %%
clf = RandomForestClassifier()
clf.fit(kcal_share_subset.head(100), y[:100])

logging.info("Creating SHAP values")
shap_values = shap.TreeExplainer(clf).shap_values(kcal_share_subset.head(100))

# %%
from shap import Explainer, waterfall_plot

explainer = Explainer(clf)
sv = explainer.shap_values(kcal_share_subset.head(100))

cls = 9  # class to explain
sv_cls = sv[cls]
idx = 99

waterfall_plot(
    Explanation(
        sv_cls[idx],
        explainer.expected_value[cls],
        feature_names=kcal_share_subset.head(100).columns,
    )
)

# %%
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(kcal_share_subset.head(100))

# %%
# shap.summary_plot(shap_values, kcal_share_subset.head(100).values)

# %%
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)
# model = xgboost.XGBClassifier().fit(kcal_share_subset.head(100), y[:100])

# %%
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values, max_display=12, order=shap.Explanation.abs.mean(0))

# %%

# %%
shap.initjs()
explainer = shap.TreeExplainer(clf)

# %%
sv = explainer(kcal_share_subset.head(100))
exp = Explanation(
    sv.values[:, :, 1],
    sv.base_values[:, 1],
    data=kcal_share_subset.head(100).values,
    feature_names=kcal_share_subset.head(100).columns,
)
idx = 99
waterfall(exp[idx])

# %%
shap_values = explainer.shap_values(kcal_share_subset.head(100))

# %%
# More classes than clusters?
shap.summary_plot(
    shap_values,
    kcal_share_subset.head(100).values,
    max_display=30,
    feature_names=kcal_share_subset.head(100).columns,
)

# %%
