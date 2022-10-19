import logging
import re
from scipy.stats import zscore
from toolz import pipe
from functools import partial
import numpy as np
import pandas as pd
import altair as alt
from typing import Union

from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.utils.plotting import configure_plots
from ahl_food_reformulation.utils.io import load_s3_data
from ahl_food_reformulation.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)


clean_variable_names = {
    "number_products": "number_products",
    "kcal_100_s": "high_density_share",
    "kcal_100_w": "high_density_share_sales",
    "percent_high_ed": "high_density_products_share",
    "percent_high_ed_sales_weighted": "high_density_products_share_sales",
    "mean_kcal_size_adj": "mean_kcal_sales_adjusted",
    "median_kcal_size_adj": "median_kcal_sales_adjusted",
    "IQR_kcal_size_adj": "interquantile_range_kcal",
    "percent_kcal_contrib": "kcal_contribution_share",
    "percent_kcal_contrib_size_adj": "kcal_contribution_share_adjusted",
    "variance": "kcal_density_variance",
    "variance_size_adj": "kcal_density_variance_normalised",
    "variance_adj_scaled": "kcal_density_variance_adjusted_normalised",
    "entropy": "nutrient_entropy",
    "entropy_size_adj": "nutrient_entropy_normalised",
    "share_sh": "share_population_impacted",
    "share_adj": "share_population_impacted_adjusted_clusters",
    "cluster_no_sh": "clusters_impacted_share_clusters",
    "cluster_low_sh": "clusters_low_income_impacted_share_clusters",
    "cluster_low_adj": "clusters_impacted_adjusted_clusers",
}

var_order = list(clean_variable_names.values())

plotting_names = {name: re.sub("_", " ", name.capitalize()) for name in var_order}

plotting_order = list(plotting_names.values())

# We will focus on these variables after the correlation analysis
selected_vars = [
    "high_density_share_sales",
    "kcal_contribution_share_adjusted",
    "kcal_density_variance_normalised",
    "nutrient_entropy_normalised",
    "clusters_low_income_impacted_share_clusters",
]

# Lookup between clean variable names and categories
var_category_lookup = {
    "high_density_share_sales": "Desirability",
    "kcal_contribution_share_adjusted": "Desirability",
    "kcal_density_variance_normalised": "Feasibility",
    "nutrient_entropy_normalised": "Feasibility",
    "clusters_impacted_share_clusters": "Inclusion",
    "clusters_low_income_impacted_share_clusters": "Inclusion",
}

# Functions


def drop_diag(table: pd.DataFrame) -> pd.DataFrame:
    """Quick and dirty way to remove diagonal from correlation matrix"""

    t_copy = table.copy()

    for ind in table.index:
        for col in table.columns:
            if ind == col:
                t_copy.loc[ind, col] = None

    return t_copy


def relabel_names(table: pd.DataFrame, columns: list, clean_dict: dict) -> pd.DataFrame:
    """Relabel columns in table with clean_dict"""

    t2 = table.copy()

    for c in columns:

        t2[c] = t2[c].map(clean_dict)

    return t2


def make_correlation_heatmap(
    report_table: pd.DataFrame,
    axis_order: list,
    var_names: list = ["clean_label", "clean_label_1", "correlation"],
    col_thres: float = 0.5,
):
    """Make heatmap of correlation matrix"""

    # Creates the heatmap
    heatmap = (
        alt.Chart(report_table)
        .mark_rect(stroke="lightgrey")
        .encode(
            x=alt.X(var_names[0], sort=axis_order),
            y=alt.Y(var_names[1], sort=axis_order),
            color=alt.Color(var_names[2], title=var_names[2].capitalize()),
        )
    )

    # Text labels
    text = (
        alt.Chart(report_table)
        .mark_text(color="white", fontSize=9)
        .encode(
            x=alt.X(var_names[0], sort=axis_order, title=None),
            y=alt.Y(var_names[1], sort=axis_order, title=None),
            text=alt.Text(var_names[2], format=".1f"),
            color=alt.condition(
                f"datum.{var_names[2]} > {col_thres}",
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    return heatmap + text


def make_indicator_heatmap(
    report_table: pd.DataFrame,
    axis_order: list,
    var_names: list,
    col_thres=0.5,
) -> pd.DataFrame:
    """Adaptation of the function above to make heatmaps of other indicators"""

    heatmap = (
        alt.Chart(report_table)
        .mark_rect(stroke="lightgrey")
        .encode(
            x=alt.X(var_names[0], sort=axis_order),
            y=alt.Y(
                var_names[1],
                sort=alt.EncodingSortField(var_names[2], op="mean", order="descending"),
            ),
            color=alt.Color(var_names[2], title=var_names[2].capitalize()),
        )
    )

    text = (
        alt.Chart(report_table)
        .mark_text(color="white", fontSize=9)
        .encode(
            x=alt.X(var_names[0], sort=axis_order, title=None),
            y=alt.Y(
                var_names[1],
                sort=alt.EncodingSortField(var_names[2], op="mean", order="descending"),
                title=None,
            ),
            text=alt.Text(var_names[2], format=".1f"),
            color=alt.condition(
                f"datum.{var_names[2]} > {col_thres}",
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    return heatmap + text


def make_recommendations(
    report_table: pd.DataFrame, top_reccs: int = 5
) -> pd.DataFrame:
    """Creates a table with reformulation recommendations based on different indicators
    and the mean of different indicators"""

    report_table_aggregated = (
        report_table.groupby(["category", "rst_4_market_sector"])["z_score"]
        .mean()
        .reset_index(drop=False)
    )

    top_candidates = {}

    for cat in report_table_aggregated["category"].unique():

        top_candidates[cat] = ", ".join(
            report_table_aggregated.query(f"category=='{cat}'").sort_values(
                "z_score", ascending=False
            )["rst_4_market_sector"][:top_reccs]
        )

    return pd.DataFrame(top_candidates, index=["Top candidates"]).assign(
        Average=", ".join(
            report_table_aggregated.groupby("rst_4_market_sector")["z_score"]
            .mean()
            .sort_values(ascending=False)
            .index[:top_reccs]
            .values
        )
    )


def make_detailed_recommendations(
    high_level_cats: list,
    detailed_table: pd.DataFrame,
    variables: list = [
        "nutrient_entropy_normalised",
        "kcal_contribution_share_adjusted",
    ],
    thresholds: list = [100, 10],
) -> pd.DataFrame:
    """Function to make detailed recommendations about products to reformulate"""

    detailed_products = {}

    for t in top_cats:

        detailed_products[t] = ", ".join(
            (
                detailed_table.query(f"rst_4_market_sector=='{t}'")
                .dropna(axis=0, subset=variables[0])
                # This beging by sorting by out "top criterion" and then the second
                .sort_values(variables[0], ascending=False)[: thresholds[0]]
                .sort_values(variables[1], ascending=False)["product"][: thresholds[1]]
                .values
            )
        )

    return detailed_products


if __name__ == "__main__":

    webdr = google_chrome_driver_setup()

    logging.info("Reading data")
    report_table_clean = (
        load_s3_data(
            "ahl-private-data",
            "kantar/data_outputs/decision_table/decision_table_rst_4_market_sector.csv",
        )
        .drop(axis=1, labels=["chosen_unit"])
        .melt(id_vars="rst_4_market_sector")
        .assign(clean_label=lambda df: df["variable"].map(clean_variable_names))
        .reset_index(drop=True)
        .pivot_table(
            index="rst_4_market_sector", columns="clean_label", values="value"
        )[var_order]
    )

    logging.info("Calculating correlation matrix")

    corr_table_long = (
        drop_diag(report_table_clean.corr())
        .rename_axis("clean_label_1")
        .stack()
        .reset_index(name="correlation")
    )

    corr_plot = (
        pipe(
            corr_table_long,
            partial(
                relabel_names,
                columns=["clean_label", "clean_label_1"],
                clean_dict=plotting_names,
            ),
            partial(make_correlation_heatmap, axis_order=plotting_order),
            configure_plots,
        )
        .configure_axis(labelLimit=1000, labelFontSize=12)
        .properties(width=500, height=500)
    )

    save_altair(corr_plot, "indicator_correlation_matrix", driver=webdr)

    logging.info("Calculating indicator heatmaps")

    report_table_clean_long = (
        report_table_clean[selected_vars]
        .apply(lambda x: zscore(x, nan_policy="omit"))
        .stack()
        .reset_index(name="z_score")
        .assign(category=lambda df: df["clean_label"].map(var_category_lookup))
    )

    indicator_heatmap = (
        pipe(
            report_table_clean_long,
            partial(relabel_names, columns=["clean_label"], clean_dict=plotting_names),
            partial(
                make_indicator_heatmap,
                axis_order=plotting_order,
                var_names=["clean_label", "rst_4_market_sector", "z_score"],
            ),
            configure_plots,
        )
        .configure_axis(labelLimit=1000, labelFontSize=12)
        .properties(width=200, height=800)
    )

    save_altair(indicator_heatmap, "indicator_heatmap", driver=webdr)

    logging.info("Making high level recommendations")
    recc_table = make_recommendations(report_table_clean_long, 5).T

    logging.info(recc_table.head())

    recc_table.to_csv(f"{PROJECT_DIR}/outputs/reports/recommendation.csv")

    logging.info("Making detailed recommendations")

    report_table_detailed = pipe(
        load_s3_data(
            "ahl-private-data",
            "kantar/data_outputs/decision_table/decision_table_rst_4_extended.csv",
        ).rename(columns={"Unnamed: 0": "product"}),
        lambda df: df.rename(
            columns={
                v: clean_variable_names[v] if v in clean_variable_names.keys() else v
                for v in df.columns
            }
        ),
    )[["product", "rst_4_market_sector"] + selected_vars]

    top_cats = recc_table.loc["Average", "Top candidates"].split(", ")

    detailed_reccs = pd.DataFrame(
        make_detailed_recommendations(top_cats, report_table_detailed),
        index=["Detailed recommendations"],
    ).T

    logging.info(detailed_reccs.head())

    detailed_reccs.to_csv(f"{PROJECT_DIR}/outputs/reports/detailed_recommendation.csv")