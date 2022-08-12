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
from typing import Dict, Any, List, Union, Tuple
from functools import partial
from toolz import pipe

import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
from ahl_food_reformulation import PROJECT_DIR
import ahl_food_reformulation.analysis.clustering_interpretation as cluster_interp
from ahl_food_reformulation.utils.altair_save_utils import (
    save_altair,
    google_chrome_driver_setup,
    altair_text_resize,
)
from ahl_food_reformulation.getters.miscelaneous import postcode_region_lookup


# %%
driver = google_chrome_driver_setup()


# %%
# Functions


def plot_pipeline(
    demog,
    comp_variable,
    order_vars,
    drops=["Unknown", "Did not want to answer", "Other"],
):
    """Pipeline to generate demographic plots"""
    return pipe(
        demog.query("cluster!=18"),
        partial(cluster_interp.calculate_cluster_shares, variable=comp_variable),
        partial(
            cluster_interp.plot_cluster_comparison_cat,
            drop=drops,
            pos_text=3,
            var_order=order_vars,
            var_name=comp_variable,
        ),
    )


def make_salience_table(
    coefficients_df: pd.DataFrame,
    coeff_name: str = "coefficient",
    comp_var: str = "variable",
    top_n: int = 10,
):
    """Create table with top scores"""
    return (
        coefficients_df.groupby("cluster")
        .apply(
            lambda df: (
                df.assign(
                    high_predictive=lambda df: ", ".join(
                        df.sort_values(coeff_name, ascending=False).head(n=top_n)[
                            comp_var
                        ]
                    )
                )
                .assign(
                    low_predictive=lambda df: ", ".join(
                        df.sort_values(coeff_name, ascending=True).head(n=top_n)[
                            comp_var
                        ]
                    )
                )[["cluster", "high_predictive", "low_predictive"]]
                .drop_duplicates("cluster")
            )
        )
        .reset_index(drop=True)
    )


def item_distribution(purchase_table, category):

    return (
        purchase_table.groupby(category).apply(lambda df: df["volume"].sum())
        / purchase_table["volume"].sum()
    )


def plot_distro(distribution, category):

    return (
        alt.Chart(distribution.reset_index(name="share"))
        .mark_line(point=True, stroke="red")
        .encode(
            y=alt.Y(category, sort=alt.EncodingSortField("share", order="descending")),
            x=alt.X("share", title="Share of volume", axis=alt.Axis(format="%")),
        )
    )


def make_purchase_shares_normalised(
    purchase_records: pd.DataFrame, category: str, top_n: int = 5, num_clust: int = 18
) -> pd.DataFrame:
    """Creates normalised purchase shares"""

    purchase_shares_cluster = (
        purchase_records.query("clust!=18")
        .groupby(["clust"])
        .apply(
            lambda df: (
                df.groupby(category)["volume"].sum() / df["volume"].sum()
            ).sort_values(ascending=False)
        )
    )

    return (
        purchase_shares_cluster.unstack(level=1)
        .fillna(0)
        .apply(lambda col: zscore(col))
        .stack()
        .reset_index(name="share_normalised")
        .groupby(category)  # This is to extract labels for outlierish clusters
        .apply(
            lambda df: df.assign(
                add_label=[
                    text if (rank < top_n) | (rank > num_clust - top_n) else ""
                    for text, rank in zip(df["clust"], df["share_normalised"].rank())
                ]
            )
        )
        .reset_index(drop=False)
    )


def plot_shares_normalised(
    purchase_normalised: pd.DataFrame, category: str, width: int = 700
) -> alt.Chart:
    """ """

    ch = (
        alt.Chart(purchase_normalised)
        .mark_point(filled=True, size=50, stroke="black", strokeWidth=0.5)
        .encode(
            y=category,
            x=alt.X("share_normalised", title="Share of volume, normalised"),
            tooltip=["clust"],
            color=alt.Color("clust:O", scale=alt.Scale(scheme="tableau20")),
        )
    )

    txt = (
        alt.Chart(purchase_normalised)
        .mark_text(dx=10, dy=-4)
        .encode(
            y=category,
            x=alt.X("share_normalised", title="Share of volume, normalised"),
            tooltip=["clust"],
            text="add_label",
            color=alt.Color("clust:O", scale=alt.Scale(scheme="tableau20")),
        )
    )

    rul = (
        alt.Chart(pd.DataFrame({"x": [1]}))
        .mark_rule(strokeDash=[2, 2], stroke="red")
        .encode(x="x")
    )

    return (ch + rul + txt).properties(width=width)


def make_regression_dataset(
    purchase_records: pd.DataFrame,
    clust_lu: Dict,
    demog_table: pd.DataFrame,
    category: str,
    sample_size: int = 5000,
) -> Tuple:
    """Creates a regression dataset"""

    house_sample = random.sample(
        purchase_records.dropna(axis=0, subset=["clust"])
        .query("clust!=18")["panel_id"]
        .unique()
        .tolist(),
        sample_size,
    )

    purchase_records_sample = purchase_recs_may.loc[
        purchase_recs_may["panel_id"].isin(house_sample)
    ].reset_index(drop=True)

    print(len(purchase_records_sample))

    househ_shares_target = (
        purchase_records_sample.groupby(["panel_id"])
        .apply(lambda df: df.groupby(category)["volume"].sum() / df["volume"].sum())
        .unstack(level=1)
        .fillna(0)
        .apply(lambda x: zscore(x))
    )

    househ_features = (
        househ_shares_target.reset_index(drop=False)["panel_id"]
        .to_frame()
        .assign(cluster=lambda df: df["panel_id"].map(clust_lu))
        .assign(
            hous_size=lambda df: df["panel_id"].map(
                demog.set_index("panel_id")["household_size"].to_dict()
            )
        )
    )

    return househ_shares_target, househ_features


def fit_purchase_regression(
    target: pd.DataFrame, features: pd.DataFrame, category
) -> pd.DataFrame:
    """Regresses share of purchases on household basket on cluster + household size"""

    coeffs = []
    print("Regressing...")
    for cat in target.columns:
        print("\t" + cat)

        for clust in features["cluster"].unique():

            endog = target[cat].to_numpy()
            exog = sm.add_constant(
                features.drop(axis=1, labels=["panel_id"]).assign(
                    cluster=lambda df: (df["cluster"] == clust).astype(int)
                )
            ).to_numpy()

            mod = sm.OLS(endog, exog)
            out = mod.fit()
            coeffs.append([cat, clust, out.params[1], out.pvalues[1]])

    return pd.DataFrame(coeffs, columns=[category, "cluster", "coefficient", "p_value"])


def plot_regression_result(
    regression_table, pvalue_thres=0.05, category="rst_4_market_sector"
):
    """Plots regression results"""

    return (
        alt.Chart(regression_table.query(f"p_value<{pvalue_thres}"))
        .mark_rect(stroke="darkgrey", strokeWidth=0.5)
        .encode(
            y=category,
            x="cluster:O",
            tooltip=[category, "cluster", alt.Tooltip("coefficient", format=".3f")],
            color=alt.Color(
                "coefficient",
                scale=alt.Scale(scheme="redblue", domainMid=0),
                sort="descending",
                title="Regression coefficient",
            ),
        )
    )


# %% [markdown]
# ## Read data

# %%
# Cluster assignments
clust = kantar.panel_clusters()

sizeplot = clust["clusters"].value_counts().plot(kind="bar")

plt.savefig(f"{PROJECT_DIR}/outputs/figures/sizeplot.png")


# %%
clust_lu = clust.set_index("Panel Id")["clusters"].to_dict()

# %%
# Household BMI

demog = kantar.demog_clean()

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

CATEGORY_VARS = [
    "council_tax_band",
    "education_level",
    "ethnicity",
    "household_income",
    "life_stage",
    "region",
    "social_class",
    "urban-rural",
]

for var in cluster_interp.CATEGORY_VARS:

    cat_share = cluster_interp.calculate_cluster_shares(demog.query("cluster!=18"), var)
    plot = cluster_interp.plot_cluster_comparison_cat(
        cat_share, var, drop=["Unknown"], pos_text=3
    )
    category_charts[var] = plot


# %%
house_plot = plot_pipeline(
    demog,
    "household_income",
    [
        "£0 - £9,999 pa",
        "£10,000 - £19,999 pa",
        "£20,000 - £29,999 pa",
        "£30,000 - £39,999 pa",
        "£40,000 - £49,999 pa",
        "£50,000 - £59,999 pa",
        "£60,000 - £69,999 pa",
        "£70,000 +",
    ],
)

save_altair(
    altair_text_resize(house_plot).properties(width=600), "income_comp", driver=driver
)

house_plot


# %%
edu_plot = plot_pipeline(
    demog,
    "education_level",
    ["None", "GCSE", "A Level", "Higher education", "Degree or higher"],
)

save_altair(
    altair_text_resize(edu_plot).properties(width=600), "edu_plot", driver=driver
)

edu_plot

# %% [markdown]
# #### Descriptive analysis: Other variables

# %%
high_bmi = altair_text_resize(
    cluster_interp.plot_cluster_comparison_non_cat(
        demog.query("cluster!=18").query("bmi_missing==0"), "high_bmi", n_cols=7
    )
)

save_altair(high_bmi, "high_bmi_comp", driver=driver)

high_bmi

# %%
ageplot = altair_text_resize(
    cluster_interp.plot_cluster_comparison_non_cat(
        demog.query("cluster!=18"), "main_shopper_age"
    )
)

save_altair(ageplot, "age_cluster_comp", driver=driver)

ageplot

# %%
cluster_interp.plot_cluster_comparison_non_cat(
    demog.query("cluster!=18"), "household_size"
)

# %% [markdown]
# ### Predictors of clustering

# %%
X_train, X_test, y_train, y_test, all_X, all_y = cluster_interp.make_modelling_dataset(
    demog.query("cluster!=18")
)

# %%
cluster_interp.simple_grid_search(
    X_train, X_test, y_train, y_test, [0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]
)

# %%
regression_coefficients = cluster_interp.get_regression_coefficients(
    all_X, all_y, 0.005, top_keep=10
).assign(cluster=lambda df: df["cluster"].astype(str).replace("18", "19").astype(int))

# %%
demog_regression = cluster_interp.plot_regression_coeffs(
    regression_coefficients
).properties(width=1000, height=300)

save_altair(altair_text_resize(demog_regression), "demog_regression", driver=driver)

demog_regression

# %%
make_salience_table(regression_coefficients, top_n=10).to_csv(
    f"{PROJECT_DIR}/outputs/demog_salience.csv", index=False
)

make_salience_table(regression_coefficients, top_n=5)

# %% [markdown]
# ### Differences in consumption patterns

# %%
from scipy.stats import zscore

# %%
prod_info = kantar.product_metadata()

# %%
prod_info.head()

# %%
prod_info.shape

# %%
purchase_recs = kantar.purchase_records()

# %%
purchase_recs_may = (
    purchase_recs.rename(
        columns={c: re.sub(" ", "_", c.lower()) for c in purchase_recs.columns}
    )
    .query("period==202106")
    .reset_index(drop=True)
    .merge(
        prod_info[
            [
                "product_code",
                "rst_4_extended",
                "rst_4_market",
                "rst_4_market_sector",
                "rst_4_trading_area",
            ]
        ],
        left_on="product_code",
        right_on="product_code",
        how="left",
    )
    .assign(clust=lambda df: df["panel_id"].map(clust_lu))
)

# %%
share_distro = plot_distro(
    item_distribution(purchase_recs_may, "rst_4_market_sector"), "rst_4_market_sector"
)

save_altair(altair_text_resize(share_distro), "purchase_volume", driver=driver)

share_distro

# %%
food_volumes = plot_shares_normalised(
    make_purchase_shares_normalised(purchase_recs_may, "rst_4_market_sector", top_n=2),
    "rst_4_market_sector",
)

save_altair(
    altair_text_resize(food_volumes).properties(width=1200),
    "food_vol_comp",
    driver=driver,
)

food_volumes

# %% [markdown]
# ### Multivariate analysis

# %%
import statsmodels.api as sm
import random

# %%

# %%
househ_shares_target, househ_features = make_regression_dataset(
    purchase_recs_may, clust_lu, demog, "rst_4_market_sector"
)

# %%
purchase_reg_coeffs = fit_purchase_regression(
    househ_shares_target, househ_features, "rst_4_market_sector"
)

# %%
share_reg = plot_regression_result(purchase_reg_coeffs)

save_altair(share_reg, "regression_purchases", driver=driver)

share_reg

# %%
reg_salience = make_salience_table(
    purchase_reg_coeffs, "coefficient", "rst_4_market_sector", top_n=5
)

reg_salience.to_csv(f"{PROJECT_DIR}/outputs/purchase_salience.csv", index=False)

reg_salience
