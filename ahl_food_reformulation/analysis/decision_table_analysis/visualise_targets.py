# Import libraries
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import energy_density as energy
from ahl_food_reformulation.utils.plotting import configure_plots
from ahl_food_reformulation.utils.altair_save_utils import (
    save_altair,
    google_chrome_driver_setup,
    altair_text_resize,
)
import altair as alt
import logging
import json
import pandas as pd
import numpy as np

# define cats
broader_category = "rst_4_market_sector"
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
                axis=alt.Axis(format="%", grid=False),
                scale=alt.Scale(domain=[0, 1]),
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
    # read data
    pur_recs = kantar.purchase_records_updated()
    nut_recs = kantar.nutrition()
    prod_meta = kantar.prod_meta_update()
    prod_meas = kantar.product_measurement()
    pan_ind = kantar.household_ind()
    prod_mast = kantar.product_master()
    val_fields = kantar.val_fields()
    uom = kantar.uom()
    prod_codes = kantar.product_codes()
    prod_vals = kantar.product_values()
    panel_weight = kantar.panel_weights_year()
    gran_decision = pd.read_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/decision_table_"
        + granular_category
        + ".csv"
    )
    broad_decision = pd.read_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/decision_table_"
        + broader_category
        + ".csv"
    )

    # Set driver for altair
    driver = google_chrome_driver_setup()

    # Get chosen categories as dataframe
    with open(f"{PROJECT_DIR}/outputs/reports/detailed_products.json") as f:
        chosen_cats = pd.DataFrame(json.load(f)).melt(
            var_name=broader_category, value_name=granular_category
        )

    logging.info("Create tables")
    # Unique list of broad cats
    broad_cats = list(chosen_cats.rst_4_market_sector.drop_duplicates())
    # Broad plot table
    broad_decision["Categories"] = np.where(
        broad_decision["rst_4_market_sector"].isin(broad_cats),
        broad_decision["rst_4_market_sector"],
        "Other categories",
    )
    # Granular plot table
    gran_decision.rename({"Unnamed: 0": "rst_4_extended"}, inplace=True, axis=1)
    gran_decision_subset = chosen_cats.merge(
        gran_decision[
            ["rst_4_extended", "percent_high_ed_sales_weighted", "percent_high_ed"]
        ],
        on="rst_4_extended",
        how="left",
    )
    gran_decision_subset.columns = [
        "Market sector",
        "Categories",
        "Percent sales high ED",
        "Percent products high ED",
    ]
    gran_decision_subset["Percent sales high ED"] = (
        gran_decision_subset["Percent sales high ED"] / 100
    )
    gran_decision_subset["Percent products high ED"] = (
        gran_decision_subset["Percent products high ED"] / 100
    )

    # Get avergaes for % high energy density
    df_prod_ed = energy.prod_energy_100(
        broader_category,
        pur_recs,
        nut_recs,
        prod_meta,
        prod_meas,
    )
    df_prod_ed["energy_density_cat"] = energy.energy_density_score(
        df_prod_ed["kcal_100g_ml"]
    )
    # Energy density sales
    ed_cats_sales = (
        df_prod_ed.groupby(["energy_density_cat"])["total_sale"].sum()
        / df_prod_ed["total_sale"].sum()
    )
    # Energy density products
    ed_cats_num = (
        df_prod_ed.groupby(["energy_density_cat"]).size()
        / df_prod_ed.groupby(["energy_density_cat"]).size().sum()
    )
    # Avg variables
    mean_ed_sales = ed_cats_sales.high
    mean_ed_prods = ed_cats_num.high

    # Group cats for plot with same values
    broad_decision_clusters = broad_decision.copy()
    broad_decision_clusters["Categories"] = np.where(
        broad_decision_clusters["Categories"].isin(
            ["Ambient Bakery Products", "Chilled Convenience"]
        ),
        "Ambient Bakery Products + Chilled Convenience",
        broad_decision_clusters["Categories"],
    )

    logging.info("Create plots")
    # Create plots
    figure_density_prod_sales = decision_table_scatter(
        broad_decision,
        "kcal_100_s",
        "kcal_100_w",
        "Kcal per 100g/l",
        "Kcal per 100g/l sales weighted",
        NESTA_COLOURS,
        "Kcal density products and sales weighted",
        ["right", "line-bottom", 15, -5],
    )
    figure_cont_perc_ed = decision_table_scatter(
        broad_decision,
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
    figure_facet_sales_ed = facet_bar_perc_ed(
        gran_decision_subset,
        mean_ed_sales,
        "Average % of sales",
        "Percent sales high ED",
        "Percent of sales",
        "#18A48C",
        "Percent of Sales as High Energy Density",
    )
    figure_facet_prods_ed = facet_bar_perc_ed(
        gran_decision_subset,
        mean_ed_prods,
        "Average % of products",
        "Percent products high ED",
        "Percent of products",
        "#EB003B",
        "Percent of Products as High Energy Density",
    )

    logging.info("Save plots")
    # Save plots
    save_altair(
        altair_text_resize(figure_density_prod_sales),
        "scatter_density_prod_sales",
        driver=driver,
    )
    save_altair(
        altair_text_resize(figure_cont_perc_ed),
        "scatter_cont_perc_ed",
        driver=driver,
    )
    save_altair(
        altair_text_resize(figure_cluster_share),
        "scatter_clust_share",
        driver=driver,
    )
    save_altair(
        altair_text_resize(figure_facet_sales_ed),
        "facet_sales_ed",
        driver=driver,
    )
    save_altair(
        altair_text_resize(figure_facet_prods_ed),
        "facet_products_ed",
        driver=driver,
    )
