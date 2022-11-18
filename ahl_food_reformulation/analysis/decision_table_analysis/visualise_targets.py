# Import libraries
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.utils.plotting import configure_plots
from ahl_food_reformulation.utils.altair_save_utils import (
    save_altair,
    google_chrome_driver_setup,
    altair_text_resize,
)
from ahl_food_reformulation.utils.io import load_s3_data
import json
import numpy as np
import altair as alt
import logging
import pandas as pd

# define cats
broader_category = "rst_4_market"
granular_category = "rst_4_extended"

# Define colours
NESTA_COLOURS = ["#0000FF", "#FDB633", "#18A48C", "#9A1BBE", "#EB003B", "#646363"]

# Plotting functions
def decision_table_scatter(
    source: pd.DataFrame,
    col_x: str,
    col_y: str,
    x_title: str,
    y_title: str,
    NESTA_COLOURS: list,
    title: str,
    alignment: list,
):
    """
    Plots altair scatter based on metrics in decision table for market sector
    Args:
        source (pd.DataFrame): Pandas dataframe market sector decision table
        col_x (str): x-axis column
        col_y (str): y-axis column
        x_title (str): Title for x-axis
        y_title (str): Title for y-axis
        NESTA_COLOURS (list): List of hex nesta colours
        title (st): Plot title
        alignment (list): Specs for alignment of annotations
    Returns:
        Scatter plot
    """
    opacity = alt.condition(
        alt.datum.Categories == "Other categories", alt.value(0.2), alt.value(1)
    )
    points = (
        alt.Chart(source)
        .mark_circle(size=50)
        .encode(
            x=alt.X(col_x, axis=alt.Axis(title=x_title, grid=False)),
            y=alt.Y(col_y, axis=alt.Axis(title=y_title, grid=False)),
            color=alt.Color("Categories", scale=alt.Scale(range=NESTA_COLOURS)),
            opacity=opacity,
        )
        .properties(width=400, height=400)
    )
    text = (
        points.transform_filter(alt.datum.Categories != "Other categories")
        .mark_text(
            align=alignment[0],
            baseline=alignment[1],
            dx=alignment[2],
            dy=alignment[3],
            fontSize=14,
        )
        .encode(text="Categories")
    )
    fig = points + text
    return configure_plots(
        fig,
        title,
        "",
        16,
        20,
        16,
    )


def facet_bar_perc_ed(
    gran_decision_subset: pd.DataFrame,
    avg: int,
    avg_desc: str,
    col_x: str,
    x_label: str,
    hex_colour: list,
    title: str,
    max_scale: int,
    y_type: str,
):
    """
    Plots altair scatter based on metrics in decision table for market sector
    Args:
        gran_decision_subset (pd.DataFrame): Pandas dataframe rst extended decision table
        avg (int): Average percent high energy density
        avg_desc (str): Avg description for legend
        col_x (str): x-axis column
        x_label (str): Title for x-axis
        NESTA_COLOURS (list): List of hex nesta colours
        title (st): Plot title
        max_scale (int): Maximum axis scale
        y_type (str): Y format
    Returns:
        Facet bar plot
    """
    source = gran_decision_subset.copy()
    source["mean"] = avg
    source["kind"] = avg_desc
    # Create granular cats plot
    fig_gran = (
        alt.Chart()
        .mark_bar()
        .encode(
            y=alt.Y(
                "Categories",
                title="Categories",
                axis=alt.Axis(titlePadding=20),
                sort=alt.EncodingSortField(col_x, order="descending"),
            ),
            x=alt.X(
                col_x,
                title=x_label,
                axis=alt.Axis(format=y_type, grid=False),
                scale=alt.Scale(domain=[0, max_scale]),
            ),
        )
    )
    chart_two = (
        alt.Chart()
        .mark_rule(strokeDash=[5, 5], size=2)
        .encode(
            x="mean",
            color=alt.Color(
                "kind", legend=alt.Legend(title=""), scale=alt.Scale(range=["#646363"])
            ),
        )
    )
    final_plot = (
        (fig_gran + chart_two)
        .properties(width=150, height=120)
        .facet(
            facet=alt.Facet(
                "Market sector", header=alt.Header(labelFontSize=16), title=""
            ),
            columns=2,
            data=source,
        )
    )
    return configure_plots(
        final_plot.resolve_scale(x="independent", y="independent").configure_mark(
            color=hex_colour, opacity=0.8
        ),
        title,
        "",
        18,
        14,
        14,
    )


if __name__ == "__main__":

    logging.info("Reading data")

    broad_decision = load_s3_data(
        "ahl-private-data",
        "kantar/data_outputs/decision_table/decision_table_"
        + broader_category
        + ".csv",
    )
    gran_decision = (
        load_s3_data(
            "ahl-private-data",
            "kantar/data_outputs/decision_table/decision_table_"
            + granular_category
            + ".csv",
        )
    ).rename({"Unnamed: 0": "rst_4_extended"}, axis=1)

    # Get averages
    avg_kcal_cont = (gran_decision["percent_kcal_contrib_weighted"].mean()) / 100
    avg_ed_sales = gran_decision["kcal_100_s"].mean()

    # Unique list of chosen cat groups
    chosen_cats_list = ["_10", "_3"]

    # Set driver for altair
    driver = google_chrome_driver_setup()

    for chosen_method in chosen_cats_list:
        # Get chosen categories as dataframe
        with open(
            f"{PROJECT_DIR}/outputs/reports/detailed_products" + chosen_method + ".json"
        ) as f:
            chosen_cats = (
                pd.DataFrame.from_dict(json.load(f), orient="index")
                .transpose()
                .melt(var_name=broader_category, value_name=granular_category)
                .dropna()
            )

        logging.info("Create tables for: " + chosen_method)
        # df for broad_cats
        broad_decision_subset = broad_decision.copy()
        gran_decision_sub = gran_decision.copy()

        # Unique list of broad cats
        broad_cats = list(chosen_cats[broader_category].drop_duplicates())

        # Broad plot table
        broad_decision_subset["Categories"] = np.where(
            broad_decision_subset[broader_category].isin(broad_cats),
            broad_decision_subset[broader_category],
            "Other categories",
        )
        # Granular plot table
        gran_decision_subset = chosen_cats.merge(
            gran_decision_sub[
                [
                    granular_category,
                    "percent_kcal_contrib_weighted",
                    "percent_kcal_contrib_size_adj_weighted",
                    "kcal_100_s",
                ]
            ],
            on=granular_category,
            how="left",
        )
        gran_decision_subset.columns = [
            "Market sector",
            "Categories",
            "Percent kcal contr",
            "Percent kcal contr - size adj",
            "Sales weight ED",
        ]

        gran_decision_subset["Percent kcal contr"] = (
            gran_decision_subset["Percent kcal contr"] / 100
        )
        gran_decision_subset["Percent kcal contr - size adj"] = (
            gran_decision_subset["Percent kcal contr - size adj"] / 100
        )

        # # Group cats for plot with same values
        broad_decision_clusters = broad_decision_subset.copy()

        logging.info("Create plots")
        # Create plots
        figure_density_prod_sales = decision_table_scatter(
            broad_decision_subset,
            "kcal_100_s",
            "kcal_100_w",
            "Kcal per 100g/l",
            "Kcal per 100g/l sales weighted",
            NESTA_COLOURS,
            "Kcal density products and sales weighted",
            ["right", "line-bottom", 15, -5],
        )
        figure_cont_perc_ed = decision_table_scatter(
            broad_decision_subset,
            "percent_high_ed",
            "percent_kcal_contrib_weighted",
            "Percent high energy density",
            "Percent kcal contribution",
            NESTA_COLOURS,
            "Kcal contribution vs rate of high energy density products",
            ["left", "line-top", 6, 1],
        )
        figure_cluster_share = decision_table_scatter(
            broad_decision_clusters,
            "cluster_no_sh",
            "cluster_low_sh",
            "clusters impacted share",
            "clusters low income impacted share",
            NESTA_COLOURS,
            "Share of clusters and low income clusters impacted",
            ["right", "line-bottom", 15, -5],
        )
        figure_facet_kcal_cont = facet_bar_perc_ed(
            gran_decision_subset,
            avg_kcal_cont,
            "Average kcal contribution",
            "Percent kcal contr",
            "Percent kcal contribution",
            "#18A48C",
            "Kcal contribution per chosen product category",
            0.05,
            "%",
        )
        figure_facet_sales_ed = facet_bar_perc_ed(
            gran_decision_subset,
            avg_ed_sales,
            "Average energy density",
            "Sales weight ED",
            "Sales weight Energy Density",
            "#18A48C",
            "Sales weighted Energy density per 100g/l",
            700,
            ".0f",
        )

        logging.info("Save plots")
        # Save plots
        save_altair(
            altair_text_resize(figure_density_prod_sales),
            "scatter_density_prod_sales" + chosen_method,
            driver=driver,
        )
        save_altair(
            altair_text_resize(figure_cont_perc_ed),
            "scatter_cont_perc_ed" + chosen_method,
            driver=driver,
        )
        save_altair(
            altair_text_resize(figure_cluster_share),
            "scatter_clust_share" + chosen_method,
            driver=driver,
        )
        save_altair(
            altair_text_resize(figure_facet_kcal_cont),
            "facet_kcal_cont" + chosen_method,
            driver=driver,
        )
        save_altair(
            altair_text_resize(figure_facet_sales_ed),
            "facet_sales_weight_ed" + chosen_method,
            driver=driver,
        )
