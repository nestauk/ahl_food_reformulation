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
adj_clust_num = no_clusters[np.argmax(scores_adj)]

# %%
logging.info("Share of kcal")
# Get kmeans labels
_, kmeans_labels = cluster.kmeans_score(share_clust_num, umap_share_sub)
# Plot clusters
medium.plot_clusters("Share of kcal:", share_clust_num, umap_share_sub, kmeans_labels)

# %%
logging.info("Adjusted share of kcal")
# Get kmeans labels
_, kmeans_labels = cluster.kmeans_score(adj_clust_num, umap_adj_sub)
# Plot clusters
medium.plot_clusters(
    "Adjusted share of kcal:", adj_clust_num, umap_adj_sub, kmeans_labels
)

# %% [markdown]
# ### Temp - trialing other methods

# %%
from sklearn import mixture

gmm = mixture.GaussianMixture(n_components=50).fit(umap_share_sub)
gmm_labels = gmm.predict(umap_share_sub)

# Plot clusters
medium.plot_clusters("Share of kcal:", 50, umap_share_sub, gmm_labels)

# %%
import hdbscan

hdb = hdbscan.HDBSCAN(min_cluster_size=100)
hdb_clust = hdb.fit(umap_share_sub)
hdb_clust.labels_

# %%
from sklearn.metrics import silhouette_samples, silhouette_score

silhouette_score(umap_share_sub, hdb_labels)

# %%
from sklearn.cluster import SpectralClustering

spc = SpectralClustering(n_clusters=50)
spc_labels = spc.fit_predict(umap_share_sub)

# Plot clusters
medium.plot_clusters("Share of kcal:", 50, umap_share_sub, spc_labels)

# %% [markdown]
# ### Deep dive into best performing clusters

# %% [markdown]
# #### Testing SHAP

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
import shap

# %%
kmeanModel = KMeans(n_clusters=no_clusters[np.argmax(scores_share)])
y = kmeanModel.fit(umap_share_sub).labels_
y = label_binarize(y, classes=list(range(no_clusters[np.argmax(scores_share)])))
clf = RandomForestClassifier()
clf.fit(kcal_share_subset, y)

# %%
shap.initjs()

# %%
explainer = shap.TreeExplainer(clf)

# %%
# shap_values = explainer(kcal_share_subset).values # Very slow / does not complete

# %%
# shap.summary_plot(shap_values, X, plot_type='bar')

# %%
# visualize the first prediction’s explanation
# shap.force_plot(explainer.expected_value[0], shap_values[0])

# %% [markdown]
# ##### NOTES
#
# - Currently cannot get shap values using explainer on whole dataset (very slow). Need to explore alternative option.
#     - Tried reducing to broader category but still very slow / not completing
#     - Trying with market sector category (stopped after 10 mins or so...)
# - One option could be to use the PCA components and altair plots (suggested in article) to explain the PCA components?

# %%
