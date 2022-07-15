# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Preamble

# %%
import ahl_food_reformulation.getters.kantar as kantar

# %%
import re
from typing import Dict, Any, List, Union


import altair as alt
import pandas as pd
import numpy as np



# %%
from ahl_food_reformulation import PROJECT_DIR
import ahl_food_reformulation.analysis.clustering_interpretation as cluster_interp
from ahl_food_reformulation.utils.altair_save_utils import save_altair
from ahl_food_reformulation.getters.miscelaneous import postcode_region_lookup

# %% [markdown]
# ## Read data

# %%
# Cluster assignments
clust = kantar.panel_clusters()

clust["clusters"].value_counts().plot(kind="bar")

# %%
# Household BMI

demog = kantar.demog_clean()

demog.head()

# %%
### QUESTION: we have 5641 households without a cluster. Why?
len(demog) - len(clust)

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Predictors of clustering

# %% [markdown]
# #### Descriptive analysis: categorical variables

# %%
category_charts = {}

CATEGORY_VARS = ["council_tax_band","education_level","ethnicity","household_income","life_stage","region","social_class"]

for var in cluster_interp.CATEGORY_VARS:
    
    cat_share = cluster_interp.calculate_cluster_shares(demog_clust,var)
    plot = cluster_interp.plot_cluster_comparison_cat(cat_share,var,drop=["Unknown"],pos_text=3)
    category_charts[var] = plot
    

# %%
category_charts["education_level"]

# %%
category_charts["ethnicity"]

# %%
category_charts["life_stage"]

# %%
category_charts["region"]

# %%
category_charts["social_class"]

# %%
category_charts["household_income"]

# %% [markdown]
# #### Descriptive analysis: Other variables

# %%
cluster_interp.plot_cluster_comparison_non_cat(demog_clust,"high_bmi")

# %%
cluster_interp.plot_cluster_comparison_non_cat(demog_clust,"main_shopper_age")

# %%
cluster_interp.plot_cluster_comparison_non_cat(demog_clust,"household_size")

# %% [markdown]
# ### Predictors of clustering

# %%
X_train, X_test, y_train, y_test, all_X, all_y = cluster_interp.make_modelling_dataset(demog_clust)

# %%
cluster_interp.simple_grid_search(X_train,X_test,y_train,y_test,[0.005,0.01,0.05,0.1,0.5,1,10,100])

# %%
regression_coefficients = cluster_interp.get_regression_coefficients(all_X,all_y,0.005,top_keep=10)

# %%
cluster_interp.plot_regression_coeffs(regression_coefficients)

# %%
