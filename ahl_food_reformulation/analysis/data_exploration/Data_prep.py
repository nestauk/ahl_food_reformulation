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
# # Prepare test dataset for EDA and subsequent analysis

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
# ## Read in data and check missing values

# %%
openfoods = pd.read_csv(
    PROJECT_DIR / "inputs/data/openfoodfacts.csv"
)  # for original file use sep='\t'

# %%
openfoods.info(verbose=True, null_counts=True)

# %%
# keep_fields = ['code',
#                'url',
#                'product_name',
#                'brands',
#                'brands_tags',
#                'categories',
#                'countries_tags',
#                'countries_en',
#                'ingredients_text',
#                'serving_size',
#                'energy_100g',
#                'energy-kcal_100g',
#                'fat_100g',
#                'saturated-fat_100g',
#                'trans-fat_100g',
#                'cholesterol_100g',
#                'carbohydrates_100g',
#                'sugars_100g',
#                'fiber_100g',
#                'proteins_100g',
#                'salt_100g',
#                'sodium_100g',
#                'vitamin-a_100g',
#                'vitamin-d_100g',
#                'vitamin-e_100g',
#                'vitamin-c_100g',
#                'vitamin-b6_100g',
#                'vitamin-b12_100g',
#                'folates_100g',
#                'potassium_100g',
#                'calcium_100g',
#                'iron_100g',
#                'magnesium_100g',
#                'pantothenic-acid_100g',
#                'fruits-vegetables-nuts_100g',
#                'fruits-vegetables-nuts-estimate-from-ingredients_100g',
#                'nutrition-score-uk_100g'
#               ]

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
    "cholesterol_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    #                    'salt_100g',
    "sodium_100g",
    #                    'nutrition-score-uk_100g'
]

# %%
duplicate_cols = [
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "cholesterol_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "sodium_100g",
    "product_name",
    "brands",
]

# %%
# subset_columns = ['csvcut -t -c',
#                   ','.join(keep_fields),
#                   'inputs/data/en.openfoodfacts.org.products.csv > inputs/data/openfoodfacts.csv']

# %%
# print(' '.join(subset_columns))

# %% [markdown]
# ### Keep products sold in the UK, US, Ireland and Australia

# %%
c_tags = [
    "en:united-states",
    "en:united-kingdom",
    "en:france,en:united-kingdom",
    "en:united-kingdom,en:united-states",
    "en:belgium,en:france,en:netherlands,en:united-kingdom",
    "en:france,en:united-kingdom,en:united-states",
    "en:ireland",
    "en:australia",
]

# %%
openfoods = openfoods[openfoods["countries_tags"].isin(c_tags)]

# %%
len(openfoods)

# %%
openfoods.info(verbose=True, null_counts=True)

# %%
collections.Counter(openfoods.categories)

# %%
openfoods.head(5)

# %%
catch_all = openfoods.dropna(subset="categories")

# %%
catch_all["top_category"] = catch_all["categories"].apply(lambda x: x.split(",")[0])

# %%
catch_all["subcategory"] = catch_all["categories"].apply(
    lambda x: x.split(",")[-1].strip()
)

# %% [markdown]
# ## Curate data for select categories

# %% [markdown]
# ### Bread

# %%
bread = catch_all[catch_all.categories.str.contains("Bread")]

# %% [markdown]
# Explore unique product categories to filter out those that are not relevant

# %%
len(bread.top_category.unique())

# %%
top_cat_list = ["Plant-based foods and beverages", "Naan bread"]

# %%
bread = bread[bread["top_category"].isin(top_cat_list)]

# %%
sub_cat_list = [
    "Breads",
    "Wheat breads",
    "White breads",
    "Hamburger buns",
    "Hot dog buns",
    "Bagel breads",
    "Pitas",
    "English muffins",
    "Sliced breads",
    "Baguettes",
    "Flatbreads",
    "Wheat flatbreads",
    "Naans",
    "Rye breads",
    "Buns",
    "Gluten-free breads",
]

# %%
bread = bread[bread["subcategory"].isin(sub_cat_list)]

# %%
len(bread)

# %% [markdown]
# Remove duplicates and entries with no info on core nutrients

# %%
bread.drop_duplicates(subset=duplicate_cols, inplace=True)

# %%
len(bread)  # 98.7% remain

# %%
bread.dropna(subset=core_nutrient_info, inplace=True)

# %%
len(bread)  # 91.6% remain

# %%
bread["major_cat"] = "Breads"

# %%
bread.head(3)

# %%
bread.to_csv(PROJECT_DIR / "breads.csv")

# %% [markdown]
# ### Yogurt

# %%
yogurts = catch_all[catch_all.categories.str.contains("Yogurt")]

# %%
sorted(
    collections.Counter(yogurts.top_category).items(), key=lambda x: x[1], reverse=True
)

# %%
yogurts = yogurts[yogurts.top_category == "Dairies"]

# %%
sorted(
    collections.Counter(yogurts.subcategory).items(), key=lambda x: x[1], reverse=True
)

# %% [markdown]
# Remove duplicates and entries with no info on core nutrients

# %%
len(yogurts)

# %%
yogurts.drop_duplicates(subset=duplicate_cols, inplace=True)

# %%
len(yogurts)  # 94.9% remain

# %%
yogurts.dropna(subset=core_nutrient_info, inplace=True)

# %%
len(yogurts)  # 85% remain

# %%
yogurts["major_cat"] = "Yogurts"

# %%
yogurts.to_csv(PROJECT_DIR / "yogurts.csv")

# %% [markdown]
# ### Sauces

# %%
sauces = catch_all[catch_all.categories.str.contains("Sauces")]

# %%
sorted(
    collections.Counter(sauces.subcategory).items(), key=lambda x: x[1], reverse=True
)

# %%
sub_cat_list = [
    #                 'Sauces',
    "Dips",
    "Hot sauces",
    "Ketchup",
    "Mayonnaises",
    "Barbecue sauces",
    "Pasta sauces",
    "Salad dressings",
    "Yellow mustards",
    "Dijon mustards",
    "Tartare sauces",
]

# %%
sauces = sauces[sauces["subcategory"].isin(sub_cat_list)]

# %%
len(sauces)

# %% [markdown]
# Remove duplicates and entries with no info on core nutrients

# %%
sauces.drop_duplicates(subset=duplicate_cols, inplace=True)

# %%
len(sauces)  # 95.76%

# %%
sauces.dropna(subset=core_nutrient_info, inplace=True)

# %%
len(sauces)  # 76.6%

# %%
sauces.head(3)

# %%
sauces["major_cat"] = "Sauces"

# %%
sauces.to_csv(PROJECT_DIR / "sauces.csv")

# %% [markdown]
# ### Pizza

# %%
pizza = catch_all[catch_all.categories.str.contains("Pizza")]

# %%
sorted(collections.Counter(pizza.categories).items(), key=lambda x: x[0], reverse=True)

# %%
sorted(
    collections.Counter(catch_all.subcategory).items(), key=lambda x: x[1], reverse=True
)

# %%
pizza = pizza[pizza.categories == "Meals, Pizzas pies and quiches, Pizzas"]

# %%
pizza.head(3)

# %% [markdown]
# Remove duplicates and entries with no info on core nutrients

# %%
len(pizza)

# %%
pizza.drop_duplicates(subset=duplicate_cols, inplace=True)

# %%
len(pizza)  # 99% remain

# %%
pizza.dropna(subset=core_nutrient_info, inplace=True)

# %%
len(pizza)  # 97.7% remain

# %%
pizza["major_cat"] = "Pizza"

# %%
pizza.to_csv(PROJECT_DIR / "pizza.csv")

# %% [markdown]
# ### Sausages

# %%
sausages = catch_all[catch_all.categories.str.contains("Sausages")]

# %%
sub_cat_list = [
    "Sausages",
    "Smoked sausages",
    "Turkey sausages",
    "Chicken sausages",
    "Pork sausages",
    "Vegetarian sausages",
    "Chipolatas",
    "Frankfurter sausages",
]

# %%
sausages = sausages[sausages["subcategory"].isin(sub_cat_list)]

# %%
# sorted(collections.Counter(sausages.categories).items(), key = lambda x: x[1], reverse = True)

# %% [markdown]
# Remove duplicates and entries with no info on core nutrients

# %%
len(sausages)

# %%
sausages.drop_duplicates(subset=duplicate_cols, inplace=True)

# %%
len(sausages)  # 97.3% remain

# %%
sausages.dropna(subset=core_nutrient_info, inplace=True)

# %%
len(sausages)  # 83.28% remain

# %%
sausages["major_cat"] = "Sausages"

# %%
sausages.to_csv(PROJECT_DIR / "sausages.csv")

# %% [markdown]
# ### Combine categories into one dataframe

# %%
select_cats = pd.concat([bread, yogurts, sauces, pizza, sausages])

# %%
select_cats.to_csv(PROJECT_DIR / "select_cats.csv")

# %%
