# Code to generate analysis of Kantar food purchasing clusters

import logging
import re
from typing import Dict, Any, List, Union, Tuple
from functools import partial

import altair as alt
import pandas as pd
import numpy as np


# %%
from ahl_food_reformulation import PROJECT_DIR
import ahl_food_reformulation.getters.kantar as kantar
import ahl_food_reformulation.analysis.clustering_interpretation as cluster_interp
from ahl_food_reformulation.utils.altair_save_utils import (
    save_altair,
    google_chrome_driver_setup,
    altair_text_resize,
)


HOUSEHOLD_INCOME = [
    "£0 - £9,999 pa",
    "£10,000 - £19,999 pa",
    "£20,000 - £29,999 pa",
    "£30,000 - £39,999 pa",
    "£40,000 - £49,999 pa",
    "£50,000 - £59,999 pa",
    "£60,000 - £69,999 pa",
    "£70,000 +",
]

EDUCATION_LEVEL = ["None", "GCSE", "A Level", "Higher education", "Degree or higher"]

if __name__ == "__main__":
    driver = google_chrome_driver_setup()

    logging.info("Reading data")

    # Cluster assignments
    clust = kantar.panel_clusters()

    clust_lu = clust.set_index("Panel Id")["clusters"].to_dict()
    num_clust = len(set(clust_lu.values()))
    print(num_clust)

    # NB we are dropping the weird cluster 18
    demog = kantar.demog_clean()

    logging.info("Descriptive analysis: categorical")
    # Descriptive analysis: categorical variables

    category_charts = {}

    for var in cluster_interp.CATEGORY_VARS:
        # One plot for each categorical variable of interest. We don't do much with them

        cat_share = cluster_interp.calculate_cluster_shares(demog, var)
        category_charts[var] = cluster_interp.plot_cluster_comparison_cat(
            cat_share, var, drop=["Unknown"], pos_text=3
        )
    # Various charts
    house_plot = cluster_interp.plot_demog_pipeline(
        demog,
        "household_income",
        HOUSEHOLD_INCOME,
    )

    save_altair(
        altair_text_resize(house_plot).properties(width=600),
        "income_comp",
        driver=driver,
    )

    edu_plot = cluster_interp.plot_demog_pipeline(
        demog,
        "education_level",
        EDUCATION_LEVEL,
    )

    save_altair(
        altair_text_resize(edu_plot).properties(width=600), "edu_plot", driver=driver
    )

    logging.info("Descriptive analysis: continuous variables")

    # BMI
    high_bmi = altair_text_resize(
        cluster_interp.plot_cluster_comparison_non_cat(
            demog.query("bmi_missing==0"), "high_bmi", n_cols=7
        )
    )

    save_altair(high_bmi, "high_bmi_comp", driver=driver)

    # Age
    ageplot = altair_text_resize(
        cluster_interp.plot_cluster_comparison_non_cat(demog, "main_shopper_age")
    )

    save_altair(ageplot, "age_cluster_comp", driver=driver)

    # Household size
    cluster_interp.plot_cluster_comparison_non_cat(demog, "household_size")

    logging.info("Cluster predictive analysis")
    # What determines cluster membership?

    (
        X_train,
        X_test,
        y_train,
        y_test,
        all_X,
        all_y,
    ) = cluster_interp.make_modelling_dataset(demog)

    cluster_interp.simple_grid_search(
        X_train, X_test, y_train, y_test, [0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]
    )

    # NB we are choosing C=0.005 (a lot of regularization) as the best model
    regression_coefficients = cluster_interp.get_regression_coefficients(
        all_X, all_y, 0.005, top_keep=10
    ).assign(cluster=lambda df: df["cluster"].astype(int))

    # Plot regression coefficients
    demog_regression = cluster_interp.plot_regression_coeffs(
        regression_coefficients
    ).properties(width=1000, height=300)

    save_altair(altair_text_resize(demog_regression), "demog_regression", driver=driver)

    cluster_interp.make_salience_table(regression_coefficients, top_n=10).to_csv(
        f"{PROJECT_DIR}/outputs/demog_salience.csv", index=False
    )

    logging.info(cluster_interp.make_salience_table(regression_coefficients, top_n=5))

    logging.info("Differences in purchasing patterns across clusters")

    logging.info("Reading data")
    prod_info = kantar.product_metadata()
    purchase_recs = kantar.purchase_records()

    # Focus on May, a boring month
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

    print(purchase_recs_may.head())

    share_distro = cluster_interp.plot_item_share(
        cluster_interp.item_share(purchase_recs_may, "rst_4_market_sector"),
        "rst_4_market_sector",
    )

    save_altair(altair_text_resize(share_distro), "purchase_volume", driver=driver)

    # Shares normalised
    food_volumes = cluster_interp.plot_shares_normalised(
        cluster_interp.make_purchase_shares_normalised(
            purchase_recs_may, "rst_4_market_sector", top_n=2, num_clust=num_clust
        ),
        "rst_4_market_sector",
    )

    save_altair(
        altair_text_resize(food_volumes).properties(width=1200),
        "food_vol_comp",
        driver=driver,
    )

    logging.info("Multivariate analysis of purchases")
    # What clusters purchase what items?

    househ_shares_target, househ_features = cluster_interp.make_regression_dataset(
        purchase_recs_may, clust_lu, demog, "rst_4_market_sector"
    )

    purchase_reg_coeffs = cluster_interp.fit_purchase_regression(
        househ_shares_target, househ_features, "rst_4_market_sector"
    )

    share_reg = cluster_interp.plot_regression_result(purchase_reg_coeffs)

    save_altair(share_reg, "regression_purchases", driver=driver)

    reg_salience = cluster_interp.make_salience_table(
        purchase_reg_coeffs, "coefficient", "rst_4_market_sector", top_n=5
    )

    reg_salience.to_csv(f"{PROJECT_DIR}/outputs/purchase_salience.csv", index=False)

    logging.info(reg_salience.head())
