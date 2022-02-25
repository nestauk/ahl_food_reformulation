# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prototype EDA and dimensionality reduction

# %% [markdown]
# ## Datasets
# - Open Food Facts Kaggle dataset (356K rows)
# - Open Food Facts up to date dataset (2mln rows, after keeping only US, UK, Ireland and Australia and some other combinaions with UK down to 495741)
# - USDA dataset (7.8K rows)
#

# %% [markdown]
# Tasks:
#
# * [x] Read in a sample Open Food Facts data
# * [x] Determine the columns that are useful
# * [x] Determine which variables to use for subsetting (e.g. country, language)
# * [ ] Explore product categories (is there a hierarchy?)

# %%
import pandas as pd
import os
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
# %matplotlib inline

# %%
pd.set_option("display.max_columns", 50)

# %%
from ahl_food_reformulation import PROJECT_DIR

# %% [markdown]
# ## Read in curated dataset

# %% [markdown]
# The curated dataset was produced from Open Food Facts data (see Data_prep.py).
# It contains data on 5 food categories. These were selected to align with some categories of interest (as listed in ITT).
#
# Duplicates and missing values for core nutrients were excluded during preprocessing.

# %%
select_cats = pd.read_csv(PROJECT_DIR / "select_cats.csv", index_col=0)

# %%
len(select_cats)

# %%
select_cats.groupby("major_cat").size()

# %%
# Rank variables by the number of missing values
select_cats.isnull().sum(axis=0).sort_values()

# %%
nutrient_info = [
    "serving_size",
    "energy_100g",
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "trans-fat_100g",
    "cholesterol_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
    "vitamin-a_100g",
    "vitamin-d_100g",
    "vitamin-e_100g",
    "vitamin-c_100g",
    "vitamin-b6_100g",
    "vitamin-b12_100g",
    "folates_100g",
    "potassium_100g",
    "calcium_100g",
    "iron_100g",
    "magnesium_100g",
    "pantothenic-acid_100g",
    "fruits-vegetables-nuts_100g",
    "fruits-vegetables-nuts-estimate-from-ingredients_100g",
    "nutrition-score-uk_100g",
]

# %%
core_nutrient_info = [
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    #                    'trans-fat_100g',
    #                    'cholesterol_100g',
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    #                    'sodium_100g',
    #                    'nutrition-score-uk_100g'
]

# %% [markdown]
# ## First glance

# %% [markdown]
# ### Distributions of core nutrient content

# %%
# Set background color / chart style
sns.set_style(style="white")

# Set up  matplotlib figure
fig, axs = plt.subplots(3, 3, figsize=(20, 12))

sns.boxplot(data=select_cats, x="major_cat", y="energy-kcal_100g", ax=axs[0, 0])
sns.boxplot(data=select_cats, x="major_cat", y="fat_100g", ax=axs[0, 1])
sns.boxplot(data=select_cats, x="major_cat", y="saturated-fat_100g", ax=axs[0, 2])
sns.boxplot(data=select_cats, x="major_cat", y="sugars_100g", ax=axs[1, 0])
sns.boxplot(data=select_cats, x="major_cat", y="carbohydrates_100g", ax=axs[1, 1])
sns.boxplot(data=select_cats, x="major_cat", y="salt_100g", ax=axs[1, 2])
sns.boxplot(data=select_cats, x="major_cat", y="proteins_100g", ax=axs[2, 0])
sns.boxplot(data=select_cats, x="major_cat", y="fiber_100g", ax=axs[2, 1])
sns.boxplot(
    data=select_cats,
    x="major_cat",
    y="fruits-vegetables-nuts-estimate-from-ingredients_100g",
    ax=axs[2, 2],
)

# %%
# Set up  matplotlib figure
fig, axs = plt.subplots(3, 3, figsize=(20, 12))

sns.violinplot(data=select_cats, x="major_cat", y="energy-kcal_100g", ax=axs[0, 0])
sns.violinplot(data=select_cats, x="major_cat", y="fat_100g", ax=axs[0, 1])
sns.violinplot(data=select_cats, x="major_cat", y="saturated-fat_100g", ax=axs[0, 2])
sns.violinplot(data=select_cats, x="major_cat", y="sugars_100g", ax=axs[1, 0])
sns.violinplot(data=select_cats, x="major_cat", y="carbohydrates_100g", ax=axs[1, 1])
sns.violinplot(data=select_cats, x="major_cat", y="salt_100g", ax=axs[1, 2])
sns.violinplot(data=select_cats, x="major_cat", y="proteins_100g", ax=axs[2, 0])
sns.violinplot(data=select_cats, x="major_cat", y="fiber_100g", ax=axs[2, 1])
sns.violinplot(
    data=select_cats,
    x="major_cat",
    y="fruits-vegetables-nuts-estimate-from-ingredients_100g",
    ax=axs[2, 2],
)


# %% [markdown]
# ### Define outliers

# %%
def count_outliers(column):
    """
    Calculate number of outliers in a given column.

    Tukey method is used to identify outliers (1.5 * IQR)
    """
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1

    limit = 1.5 * IQR

    list_outliers = column[(column < Q1 - limit) | (column > Q3 + limit)]
    return len(list_outliers)


# %%
for name, group in select_cats.groupby("major_cat"):
    print(name)
    for col in core_nutrient_info:
        print(f"There are {count_outliers(group[col])} outliers in {col} variable")
    print("-------")


# %%
# Retrieve indeces for outliers
def drop_outliers(column_list, df, n=1):
    """
    Retrieve rows that contain more than 1 outlier from a given df.

    Tukey method is used to identify outliers (1.5 * IQR)
    """
    # In this case, we considered outliers as rows that have at least two outlied numerical values.
    indexes = []

    for col in column_list:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1

        limit = 1.5 * IQR

        list_outliers = df[(df[col] < Q1 - limit) | (df[col] > Q3 + limit)].index
        indexes.extend(
            list_outliers
        )  # append the found outlier indices for col to the list of outlier indices

    index_counts = collections.Counter(indexes)
    multiple_outliers = list(k for k, v in index_counts.items() if v > n)
    df = df.drop(multiple_outliers, axis=0)  # reset_index(drop=True)
    return (df, len(multiple_outliers))


# %%
filtered = []
for name, group in select_cats.groupby("major_cat"):
    filtered_df, num_outliers = drop_outliers(core_nutrient_info, group)
    filtered.append(filtered_df)
    print(
        f"For {name} category, {num_outliers} were removed. This represents {round((num_outliers/len(group))*100, 2)} % of observations in this category"
    )
    print("-------")

# %%
select_cats_filt = pd.concat(filtered)

# %% [markdown]
# ### Updated distribution plots

# %%
# Manually remove outlier for salt content
select_cats_filt.drop(444926, inplace=True)
# select_cats_filt.sort_values(by = 'salt_100g', ascending = False).head()

# %%
# Set up  matplotlib figure
fig, axs = plt.subplots(3, 3, figsize=(20, 12))

sns.boxplot(data=select_cats_filt, x="major_cat", y="energy-kcal_100g", ax=axs[0, 0])
sns.boxplot(data=select_cats_filt, x="major_cat", y="fat_100g", ax=axs[0, 1])
sns.boxplot(data=select_cats_filt, x="major_cat", y="saturated-fat_100g", ax=axs[0, 2])
sns.boxplot(data=select_cats_filt, x="major_cat", y="sugars_100g", ax=axs[1, 0])
sns.boxplot(data=select_cats_filt, x="major_cat", y="carbohydrates_100g", ax=axs[1, 1])
sns.boxplot(data=select_cats_filt, x="major_cat", y="salt_100g", ax=axs[1, 2])
sns.boxplot(data=select_cats_filt, x="major_cat", y="proteins_100g", ax=axs[2, 0])
sns.boxplot(data=select_cats_filt, x="major_cat", y="fiber_100g", ax=axs[2, 1])
sns.boxplot(
    data=select_cats_filt,
    x="major_cat",
    y="fruits-vegetables-nuts-estimate-from-ingredients_100g",
    ax=axs[2, 2],
)

# %%
import pandas_profiling
import sys

# %%
from pandas_profiling import ProfileReport


# %%
# !{sys.executable} -m pip install -U pandas-profiling[notebook]
# !jupyter nbextension enable --py widgetsnbextension

# %%
subset = select_cats_filt[core_nutrient_info].reset_index()

# %%
subset.info(verbose=True, null_counts=True)

# %%
report = subset.profile_report()

# %%
report

# %%
report.to_file(PROJECT_DIR / "eda.html")


# %%
subset.describe()

# %% [markdown]
# ## Exploring individual categories

# %%
breads = select_cats_filt[select_cats_filt["major_cat"] == "Breads"]

# %%
fig, axs = plt.subplots(2, 3, figsize=(20, 12))

sns.histplot(data=breads, x="energy-kcal_100g", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=breads, x="fat_100g", kde=True, color="gold", ax=axs[0, 2])
sns.histplot(data=breads, x="sugars_100g", kde=True, color="teal", ax=axs[1, 0])
sns.histplot(data=breads, x="salt_100g", kde=True, color="teal", ax=axs[1, 1])
sns.histplot(data=breads, x="proteins_100g", kde=True, color="teal", ax=axs[1, 2])
sns.histplot(data=breads, x="fiber_100g", kde=True, color="skyblue", ax=axs[0, 0])

plt.show()

# %%
fig, axs = plt.subplots(2, 3, figsize=(20, 12))

sns.boxplot(data=breads, x="energy-kcal_100g", color="olive", ax=axs[0, 1])
sns.boxplot(data=breads, x="fat_100g", color="gold", ax=axs[0, 2])
sns.boxplot(data=breads, x="sugars_100g", color="teal", ax=axs[1, 0])
sns.boxplot(data=breads, x="salt_100g", color="teal", ax=axs[1, 1])
sns.boxplot(data=breads, x="proteins_100g", color="teal", ax=axs[1, 2])
sns.boxplot(data=breads, x="fiber_100g", color="skyblue", ax=axs[0, 0])

plt.show()

# %%
nutrient_corr = breads[core_nutrient_info].corr()

# %%
nutrient_corr

# %%
# https://stackoverflow.com/questions/39409866/correlation-heatmap
# Set background color / chart style
# sns.set_style(style = 'white')

# Set up  matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Add diverging colormap from red to blue
cmap = sns.diverging_palette(250, 10, as_cmap=True)

sns.heatmap(
    nutrient_corr,
    #             mask=mask,
    cmap=cmap,
    square=True,
    linewidth=0.5,
    cbar_kws={"shrink": 0.5},
    ax=ax,
)


# %%
f, ax = plt.subplots(figsize=(10, 6))
ax = sns.scatterplot(data=breads, x="energy-kcal_100g", y="carbohydrates_100g")
# ax = sns.scatterplot(data=select_cats_filt, x="energy-kcal_100g", y="fiber_100g", hue="major_cat")


# %%
breads[breads["energy-kcal_100g"] < 200].sort_values(
    by="energy-kcal_100g", ascending=True
)

# %% [markdown]
# ## Dimensionality reduction and plotting

# %%
from ahl_food_reformulation.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import umap
import hdbscan
import altair as alt

# %%
from sklearn import preprocessing

# %%
norm_breads = preprocessing.normalize(breads[core_nutrient_info])

# %%
# Check the shape of the sentence embeddings array
print(norm_breads.shape)

# %%
# Create a 2D embedding
reducer = umap.UMAP(n_components=2, random_state=1)
# embedding = reducer.fit_transform(core_soups)
embedding = reducer.fit_transform(norm_breads)

# %%
# Check the shape of the reduced embedding array
embedding.shape

# %%
# Create another low-dim embedding for clustering
reducer_clustering = umap.UMAP(n_components=8, random_state=1)
# embedding_clustering = reducer_clustering.fit_transform(core_soups)
embedding_clustering = reducer_clustering.fit_transform(norm_breads)

# %%
embedding_clustering.shape

# %%
# Clustering with hdbscan
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100, min_samples=3, cluster_selection_method="leaf"
)
# clusterer.fit(core_soups)
# clusterer.fit(norm_soups)
clusterer.fit(embedding_clustering)

# %%
collections.Counter(clusterer.labels_)

# %%
# Prepare dataframe for visualisation
df = breads.copy()
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]
df["cluster"] = [str(x) for x in clusterer.labels_]


# %%
df_sample = df.sample(n=5000, random_state=123)

# %%
df_sample[df_sample["cluster"] == "19"].head()

# %%
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
fig = (
    alt.Chart(df_sample, width=500, height=500)
    .mark_circle(size=60)
    .encode(
        x="x", y="y", tooltip=["cluster", "product_name", "brands"], color="cluster"
    )
).interactive()

fig

# %%

# %% [markdown]
# Appendix

# %%
broths = soups[
    soups.product_name.str.contains("broth")
    | soups.product_name.str.contains("stock")
    | soups.product_name.str.contains("mix")
    | soups.product_name.str.contains("boullion")
]

# %%
# len(broths)

# %%
# broths.head(10)

# %%
# collections.Counter(soups.categories)

# %%
# df.loc[149316]

# %%
# ax = sns.boxplot(x="cluster", y="energy_100g", data=df)

# %%
df_filt = df[df["cluster"] != "-1"]

# %%
fig, ax = plt.subplots(figsize=(15, 6))
ax = sns.boxplot(
    x="cluster",
    y="fiber_100g",
    data=df_filt,
    #                  data=_df[_df.status == "approved"],
    #                  orient="h",
    #                  order=order,
    palette="Paired",
    ax=ax,
)

ax.set_xlabel("HDBSCAN cluster")
# ax.set_xlabel(f"'{route.replace(' ','-').replace(',','').lower()}ness' of apprenticeship standard description")
ax.set_ylabel("Fiber content")
# ax.set_xlim(-0.003, 0.01)
# ax.set_title("Energy density")
#         plt.savefig(f"figs/language_specialisation/{route}-{int(stop_pc*100)}.png", bbox_inches = "tight")

# %%
fig, ax = plt.subplots(figsize=(15, 6))
ax = sns.boxplot(
    x="cluster",
    y="salt_100g",
    data=df_filt,
    #                  data=_df[_df.status == "approved"],
    #                  orient="h",
    #                  order=order,
    palette="Paired",
    ax=ax,
)

ax.set_xlabel("HDBSCAN cluster")
# ax.set_xlabel(f"'{route.replace(' ','-').replace(',','').lower()}ness' of apprenticeship standard description")
ax.set_ylabel("Salt content")
# ax.set_xlim(-0.003, 0.01)
# ax.set_title("Energy density")
#         plt.savefig(f"figs/language_specialisation/{route}-{int(stop_pc*100)}.png", bbox_inches = "tight")

# %%
fig, ax = plt.subplots(figsize=(15, 6))
ax = sns.boxplot(
    x="cluster",
    y="sugars_100g",
    data=df_filt,
    #                  data=_df[_df.status == "approved"],
    #                  orient="h",
    #                  order=order,
    palette="Paired",
    ax=ax,
)

ax.set_xlabel("HDBSCAN cluster")
# ax.set_xlabel(f"'{route.replace(' ','-').replace(',','').lower()}ness' of apprenticeship standard description")
ax.set_ylabel("Sugar content")
# ax.set_xlim(-0.003, 0.01)
# ax.set_title("Energy density")
#         plt.savefig(f"figs/language_specialisation/{route}-{int(stop_pc*100)}.png", bbox_inches = "tight")

# %%
