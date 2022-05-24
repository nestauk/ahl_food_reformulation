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
driver = google_chrome_driver_setup()
save_altair(fig, "cluster_descriptions_", driver)


# %% [markdown]
# ### Utilities

# %% [markdown]
# #### Generic aggregation

# %%
def get_segments(df, column_name, n=4):
    """Group the values from the specified column in a df into n-segments.

    Args:
        df: A dataframe.
        column_name: A string that specifies the name of the variable in the dataframe.
        n: An int indicating number of segments. Default value is 4 (equivalent to quartiles).

    Returns:
        An array with segment labels.

    Raises:
        TypeError: If incorrect data type was passed in args.
    """
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    labelled_series = pd.qcut(df[column_name], n, labels=False)
    return labelled_series.values


# %% [markdown]
# #### Nutrient profiling

# %%
def assign_nps_energy_score(df, column_name):
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350, 5000]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(df[column_name], bins=thresholds, labels=scores, right=True)
    return binned_cats


# %%
def assign_nps_satf_score(df, column_name):
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


# %%
def assign_nps_sugars_score(df, column_name):
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45, 100]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


# %%
def assign_nps_sodium_score(df, column_name):
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 90, 180, 270, 360, 450, 540, 630, 720, 810, 900, 2400]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


# %%
energy_scores = assign_nps_energy_score(df, "energy-kcal_100g")

# %%
dummy_df = pd.DataFrame(
    {
        "category": "Bread",
        "energy": [220, 250, 335, 336, 670, 675, 1005, 1030, 1340, 3350],
        "sat_fat": [0, 0, 1, 1, 2, 3, 4, 4, 5, 11],
        "sugar": [0, 4.5, 5, 10, 15, 13.5, 50, 23, 22.5, 28],
        "sodium": [5, 90, 95, 180, 190, 250, 270, 272, 900, 901],
    }
)

# %%
dummy_df["energy_scores"] = assign_nps_energy_score(dummy_df, "energy")
dummy_df["satf_scores"] = assign_nps_satf_score(dummy_df, "sat_fat")
dummy_df["sugar_scores"] = assign_nps_sugars_score(dummy_df, "sugar")
dummy_df["sodium_scores"] = assign_nps_sodium_score(dummy_df, "sodium")


# %%
dummy_df


# %%
def assign_nps_protein_score(df, column_name):
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 1.6, 3.2, 4.8, 6.4, 8, 42]
    scores = [0, 1, 2, 3, 4, 5]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


# %%
def assign_nps_fiber_score(df, column_name):
    """Use 2004-2005 NPS to assign C points for fiber (AOAC)."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [
        0,
        0.9,
        1.9,
        2.8,
        3.7,
        4.7,
        24,
    ]  # NSP thresholds [0, 0.7, 1.4, 2.1, 2.8, 3.5]
    scores = [0, 1, 2, 3, 4, 5]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


# %%
dummy_df["protein"] = [0.5, 1.5, 1.6, 1.7, 1.88, 1.32, 6.3, 6.5, 9, 4.88]
dummy_df["fiber"] = [0.5, 0.6, 0.9, 1.0, 1.1, 3.1, 2.8, 3.5, 3.7, 3.8]

# %%
dummy_df["protein_scores"] = assign_nps_protein_score(dummy_df, "protein")
dummy_df["fiber_scores"] = assign_nps_fiber_score(dummy_df, "fiber")


# %%
def calculate_nps_fvn_score(value):
    score = None
    if np.isnan(value):
        score = np.NaN
    elif value <= 0.4:
        score = 0
    elif value > 0.4 and value <= 0.6:
        score = 1
    elif value > 0.6 and value <= 0.8:
        score = 2
    else:
        score = 5
    return score


# %%
def assign_nps_fvn_score(df, column_name):
    """Use 2004-2005 NPS to assign C points for fruit, vegetable and nut content."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    fvn_scores = df[column_name].apply(lambda x: calculate_nps_fvn_score(x))
    return fvn_scores


# %%
dummy_df["fvn"] = [np.NaN, 0, 0.1, 0.41, 0.5, 0.65, 0.7, 0, 0, 0.81]

# %%
np.isnan(np.NaN)

# %%
dummy_df["fvn_scores"] = assign_nps_fvn_score(dummy_df, "fvn")

# %%
dummy_df


# %% [markdown]
# #### Entropy

# %%
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    if type(array) is list:
        array = np.array(array)
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()  # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array += 0.0000001  # values cannot be 0
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return (np.sum((2 * index - n - 1) * array)) / (
        n * np.sum(array)
    )  # Gini coefficient


# %%
# from entropy_estimators import continuous

# %%
collections.Counter(df["subcategory"])

# %%
bagels = df[df["subcategory"] == "Bagel breads"]

# %%
fig, axs = plt.subplots(2, 3, figsize=(20, 12))

sns.histplot(data=df, x="energy-kcal_100g", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=df, x="fat_100g", kde=True, color="gold", ax=axs[0, 2])
sns.histplot(data=df, x="sugars_100g", kde=True, color="teal", ax=axs[1, 0])
sns.histplot(data=df, x="salt_100g", kde=True, color="teal", ax=axs[1, 1])
sns.histplot(data=df, x="proteins_100g", kde=True, color="teal", ax=axs[1, 2])
sns.histplot(data=df, x="fiber_100g", kde=True, color="skyblue", ax=axs[0, 0])

plt.show()

# %%
gini(df["sodium_100g"].values)

# %%
from scipy.stats import entropy


# %%
def entropy_discrete(df, column_name):
    value_counts = df[column_name].value_counts()
    return entropy(value_counts)


# %%
test_df = pd.DataFrame(
    {
        "product": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "brand": ["A", "B", "C", "D", "D", "E", "F", "G"],
    }
)
# ["A", "A", "A", "A", "B", "B", "B", "B"] # 0.6931471805599453
# ["A", "A", "B", "B", "C", "C", "D", "D"] # 1.3862943611198906
# ["A", "B", "B", "C", "D", "E", "A", "F"] # 1.7328679513998633
# ["A", "B", "C", "D", "D", "E", "F", "G"] # 1.9061547465398496

# %%
value_counts = test_df["brand"].value_counts(normalize=True)

# %%
value_counts

# %%
entropy(test_df["brand"].value_counts())

# %%
entropy_discrete(test_df, "brand")


# %%
def gini_discrete(df, column_name):
    value_counts = df[column_name].value_counts(normalize=True)
    value_counts = sorted(value_counts)
    # see this post for calculations https://stackoverflow.com/questions/31416664/python-gini-coefficient-calculation-using-numpy
    value_counts.insert(0, 0)
    shares_cumsum = np.cumsum(a=value_counts, axis=None)
    # perfect equality area
    pe_line = np.linspace(start=0.0, stop=1.0, num=len(shares_cumsum))
    area_under_pe = np.trapz(y=pe_line, dx=1 / len(shares_cumsum))
    # lorenz area
    area_under_lorenz = np.trapz(y=shares_cumsum, dx=1 / len(shares_cumsum))
    gini = (area_under_pe - area_under_lorenz) / area_under_pe
    return gini


# %%
gini_discrete(test_df, "brand")

# %%
# from matplotlib import pyplot as plt

# plt.plot(pe_line, shares_cumsum, label='lorenz_curve')
# plt.plot(pe_line, pe_line, label='perfect_equality')
# plt.fill_between(pe_line, shares_cumsum)
# plt.title('Gini: {}'.format(gini), fontsize=20)
# plt.ylabel('Cummulative brand share', fontsize=15)
# plt.xlabel('Brand share (Lowest to Highest)', fontsize=15)
# plt.legend()
# plt.tight_layout()
# plt.show()

# %% [markdown]
# Prepare data for disaggregations
# 1. Group spend by product code and household (e.g. get purchasing for the whole year by a given household)
# 2. Sum spend and volume (number of packs)
# 3. Merge product info with purchasing info: category, subcategory, brand (manufacturer), pack weight, nutrient info
# 4. Merge household info with purchasing info: household weight, life stage, social grade, region (how do we geocode region?)
# 5. Calculate avg price (spend/num packs) per 100g (divide by pack weight) # need to do more thinking on this
# 6. Calculate total calories (volume x energy density)
# 7. Calculate weighted spend, volume, total calories using household weights

# %% [markdown]
# Produce aggregate estimates
# For each granular subcategory:
# 1. Group by brand (manufacturer), sum weighted spend, volume, total calories
# 2. Group by product (this is going to be a useful aggregated object):
#     - assign price quantiles using avg price per 100g
#     - group by price quantiles (sum weighted spend, volume, calories).
# 3. Group by household characteristics:
#     - Region
#     - Life stage
#     - Social grade
#
#     sum weighted spend, volume, total calories

# %% [markdown]
# Assign NPS scores to products
# 1. Use product info on macronutrients to assign individual nutrient NPS scores.
# 2. Produce aggregated score (sum of A points, sum of known C points)

# %% [markdown]
# Measure dispersion of purchasing across various variables.
# 1. Calculate Gini index (for continuous variables) for energy density
# 2. Calculate Gini index (traditional definition) for brand (manufacturer), region, household life stage, household social grade, product price quantiles
# 3. Calculate LQs for calorie consumption (for a given product category) across regions

# %% [markdown]
# ### Appendix

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
