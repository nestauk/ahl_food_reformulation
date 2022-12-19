# Importing libraries
import logging
import re
import json
from scipy.stats import zscore
from toolz import pipe
from functools import partial
import numpy as np
import pandas as pd
import altair as alt
from typing import List

from ahl_food_reformulation.utils import plotting as pu
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.utils.plotting import configure_plots
from ahl_food_reformulation.utils.io import load_s3_data
from ahl_food_reformulation.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)

clean_variable_names = {
    "kcal_100_s": "energy_density_average",
    "kcal_100_w": "energy_density_average_weighted_by_sales",
    "percent_high_ed": "high_energy_density_products_share",
    "percent_high_ed_sales_weighted": "high_energy_density_products_share_weighted_by_sales",
    "mean_kcal_size_adj_weighted": "mean_kcal_sales_adjusted_by_size",
    "median_kcal_size_adj_weighted": "median_kcal_sales_adjusted_by_size",
    "IQR_kcal_size_adj_weighted": "interquartile_range_kcal",
    "percent_kcal_contrib_weighted": "kcal_contribution_share",
    "percent_kcal_contrib_size_adj_weighted": "kcal_contribution_share_adjusted_by_size",
    "variance": "kcal_density_variance",
    "variance_size_adj": "kcal_density_variance_normalised",
    "entropy": "kcal_entropy",
    "entropy_size_adj": "kcal_entropy_normalised",
    "share_sh": "share_population_impacted_clusters_share",
    "share_adj": "share_population_impacted_clusters_volume",
    "cluster_no_sh": "clusters_impacted_clusters_share",
    "cluster_no_adj": "clusters_impacted_clusters_volume",
    "cluster_low_sh": "clusters_low_income_impacted_clusters_share",
    "cluster_low_adj": "clusters_low_income_impacted_clusters_volume",
}

var_order = list(clean_variable_names.values())

plotting_names = {name: re.sub("_", " ", name.capitalize()) for name in var_order}

plotting_order = list(plotting_names.values())

# We will focus on these variables after the correlation analysis
selected_vars = [
    "energy_density_average_weighted_by_sales",
    "kcal_contribution_share",
    "kcal_density_variance_normalised",
    "kcal_entropy_normalised",
    "clusters_low_income_impacted_clusters_share",
]

# Lookup between clean variable names and categories
var_category_lookup = {
    "energy_density_average_weighted_by_sales": "Impact on Diets",
    "kcal_contribution_share": "Impact on Diets",
    "kcal_density_variance_normalised": "Feasibility",
    "kcal_entropy_normalised": "Feasibility",
    "clusters_impacted_clusters_share": "Inclusion",
    "clusters_low_income_impacted_clusters_share": "Inclusion",
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


def make_indicator_bubblechart(
    report_table: pd.DataFrame,
    axis_order: list,
    var_names: list,
    col_thres=2,
) -> pd.DataFrame:
    """Create a bubblechart to visualise the distribution of indicators"""

    base = alt.Chart(
        report_table.assign(abs_value=lambda df: [np.abs(x) for x in df[var_names[2]]])
    ).encode(
        x=alt.X(var_names[0], sort=axis_order, title=None),
        y=alt.Y(
            var_names[1],
            title=None,
            sort=alt.EncodingSortField(var_names[2], op="mean", order="descending"),
        ),
    )

    heatmap = base.mark_point(stroke="black", filled=True, strokeWidth=0.5).encode(
        color=alt.Color(
            "z_score",
            title="Z-score",
            scale=alt.Scale(scheme="redblue", domainMid=0),
            sort="descending",
        ),
        size=alt.Size("abs_value", title="Absolute score"),
    )

    return heatmap


def make_recommendations(
    report_table: pd.DataFrame, report_table_clean: pd.DataFrame, top_reccs: int = 37
) -> pd.DataFrame:
    """Creates a table with reformulation recommendations based on different indicators
    and the mean of different indicators"""

    report_table_aggregated = (
        report_table.groupby(["category", broad_cat_str])["z_score"]
        .mean()
        .reset_index(drop=False)
    ).merge(report_table_clean["kcal_contribution_share"], on=broad_cat_str)

    top_candidates = {}

    for cat in ["Impact on Diets", "Feasibility", "Inclusion"]:

        top_candidates[cat] = ", ".join(
            report_table_aggregated.query(f"category=='{cat}'").sort_values(
                "z_score", ascending=False
            )[broad_cat_str]
        )
    report_table_aggregated["cross"] = np.where(
        report_table_aggregated["category"] == "Impact on Diets",
        report_table_aggregated["z_score"] * 0.5,
        report_table_aggregated["z_score"] * 0.25,
    )

    return pd.DataFrame(top_candidates, index=["Top candidates"]).assign(
        Average=", ".join(
            report_table_aggregated.groupby(broad_cat_str)["cross"]
            .sum()
            .reset_index()
            .merge(
                report_table_aggregated[[broad_cat_str, "kcal_contribution_share"]],
                on=broad_cat_str,
            )
            .drop_duplicates()
            .sort_values("cross", ascending=False)
            .assign(cum=lambda x: np.cumsum(x["kcal_contribution_share"]))
            .query(f"cum <= @top_reccs")
            .sort_values("cross", ascending=False)[broad_cat_str]
            .values
        )
    )


def make_sequential_table(
    table: pd.DataFrame,
    selection_sequence: List = ["Impact on Diets", "Feasibility", "Inclusion"],
    selection_thresholds: List = [15, 10, 5],
) -> pd.DataFrame:
    """Creates a table with reformulation recommendations based on different indicators
    and selection thresholds

    Args:
        table: Table with reformulation recommendations
        selection_sequence: Sequence of selection criteria
        selection_thresholds: Thresholds for each selection criteria

    Returns:
        A table with reformulation recommendations at different steps
    """

    table_copy = table.copy()

    table_container = []

    selected = [table_copy[broad_cat_str].unique()]

    for n, crits in enumerate(zip(selection_sequence, selection_thresholds)):

        t_filt = (
            # Subsets a table on a selection criterion
            table.query(f"category=='{crits[0]}'")
            .reset_index(drop=True)
            # Only keeps products that hadn't been filtered before
            .assign(
                z_score_filtered=lambda df: [
                    z if prod in selected[n] else np.nan
                    for prod, z in zip(df[broad_cat_str], df["z_score"])
                ]
            )
            .sort_values("z_score_filtered", ascending=False)
            # Only keep colour if the z-score is above the threshold
            .assign(
                color=lambda df: [
                    z if n < crits[1] else np.nan
                    for n, z in enumerate(df["z_score_filtered"])
                ]
            )
            # Absolute value to make the size of the bubble proportional to the z-score
            .assign(z_score_abs=lambda df: [np.abs(z) for z in df["z_score"]])
            # Dummy to assign shape depending on whether the zscore is + or -
            .assign(
                shape=lambda df: [
                    "Positive score" if z > 0 else "Negative score"
                    for z in df["z_score"]
                ]
            )
        )

        # Selected criteria will be used for filtering the next table
        selected.append(t_filt[broad_cat_str][: crits[1]].unique())

        table_container.append(t_filt)

    return pd.concat(table_container)


def make_selection_chart(
    decision_table: pd.DataFrame, sort_by: str = "Inclusion"
) -> alt.Chart:
    """Creates a chart to visualise the selection process

    decision_table: Table with scores according to decision criteria
    sort_by: Sort the table by a specific decision criteria
    """

    # How to sort the categories
    sort_prod = (
        seq_table.query(f"category=='{sort_by}'")
        .sort_values("z_score_filtered", ascending=False)[broad_cat_str]
        .tolist()
    )

    return (
        alt.Chart(decision_table)
        .mark_point(filled=True, strokeWidth=1)
        .encode(
            y=alt.Y(broad_cat_str, sort=sort_prod, title=None),
            x=alt.X(
                "category",
                sort=["Impact on Diets", "Feasibility", "Inclusion"],
                title=None,
            ),
            size=alt.Size("z_score_abs", title="Absolute score"),
            shape=alt.Shape(
                "shape",
                scale=alt.Scale(range=["triangle-up", "triangle-down"]),
                sort=["Positive score", "Negative Score"],
            ),
            color=alt.condition(
                "datum.color !== null", alt.value("orange"), alt.value("white")
            ),
            stroke=alt.condition(
                "datum.color !== null", alt.value("black"), alt.value("grey")
            ),
        )
    ).properties(width=100, height=600)


def make_detailed_recommendations(
    high_level_cats: list,
    detailed_table: pd.DataFrame,
    broad_cat_str: str,
    variables: list = [
        "kcal_contribution_share",
    ],
    threshold: int = 5,
) -> pd.DataFrame:
    """Function to make detailed recommendations about products to reformulate"""

    detailed_products = {}

    for t in high_level_cats:

        detailed_products[t] = ", ".join(
            (
                detailed_table[detailed_table[broad_cat_str] == t]
                .dropna()
                .sort_values(variables[0], ascending=False)["product"][:threshold]
                # .assign(cum=lambda x: np.cumsum(x[variables[0]]))
                # .query(f"cum <= @threshold")
                # .sort_values("cum", ascending=False)["product"]
                .values
            )
        )

    return detailed_products


def make_pareto_chart(
    report_table: pd.DataFrame, report_table_clean: pd.DataFrame, top_reccs: int = 0.37
):
    """Creates a chart to visualise the cumulative contirbution of selected categories to shopping baskets"""
    report_table_aggregated = (
        report_table.groupby(["category", broad_cat_str])["z_score"]
        .mean()
        .reset_index(drop=False)
    ).merge(report_table_clean["kcal_contribution_share"], on=broad_cat_str)

    report_table_aggregated["cross"] = np.where(
        report_table_aggregated["category"] == "Impact on Diets",
        report_table_aggregated["z_score"] * 0.5,
        report_table_aggregated["z_score"] * 0.25,
    )

    chart_df = (
        report_table_aggregated.groupby(broad_cat_str)["cross"]
        .sum()
        .reset_index()
        .merge(
            report_table_aggregated[[broad_cat_str, "kcal_contribution_share"]],
            on=broad_cat_str,
        )
        .drop_duplicates()
        .sort_values("cross", ascending=False)
        .assign(cum=lambda x: np.cumsum(x["kcal_contribution_share"] / 100))
        .query(f"cum <= @top_reccs")
        .sort_values("cross", ascending=False)
    )

    chart_df["share"] = chart_df["kcal_contribution_share"] / 100

    sort_order = chart_df["rst_4_market"].tolist()

    base = alt.Chart(chart_df).encode(
        x=alt.X(
            "rst_4_market:O",
            sort=sort_order,
            title="Category",
            axis=alt.Axis(labelAngle=-45),
        ),
    )

    bars = base.mark_bar(size=20).encode(
        y=alt.Y("cross:Q", title="Average Score"),
        color=alt.Color(
            "rst_4_market",
            legend=None,
            scale=alt.Scale(
                domain=sort_order,
                range=[
                    "lightgrey",
                    pu.NESTA_COLOURS[1],
                    pu.NESTA_COLOURS[1],
                    "lightgrey",
                    pu.NESTA_COLOURS[1],
                    pu.NESTA_COLOURS[1],
                    pu.NESTA_COLOURS[1],
                    pu.NESTA_COLOURS[1],
                    pu.NESTA_COLOURS[1],
                    pu.NESTA_COLOURS[1],
                    "lightgrey",
                    pu.NESTA_COLOURS[1],
                    "lightgrey",
                    pu.NESTA_COLOURS[1],
                ],
            ),
        ),
    )

    # Create the line chart with length encoded along the Y axis
    line = base.mark_line(strokeWidth=1.5, color="#000000").encode(
        y=alt.Y("cum:Q", title="Cumulative kcal Share", axis=alt.Axis(format=".0%")),
        text=alt.Text("cum:Q"),
    )

    # Mark the percentage values on the line with Circle marks
    points = base.mark_circle(strokeWidth=3, color="#000000").encode(
        y=alt.Y("cum:Q", axis=None),
    )

    # Mark the bar marks with the value text
    bar_text = bars.mark_text(
        align="left",
        baseline="middle",
        dx=-10,  # the dx and dy can be manipulated to position text
        dy=-10,  # relative to the bar
    ).encode(
        y=alt.Y("cross:Q", title="Average Score", axis=None),
        # we'll use the percentage as the text
        text=alt.Text("cross:Q", format=".1f"),
        color=alt.value("#000000"),
    )
    # Mark the Circle marks with the value text
    point_text = points.mark_text(
        align="left",
        baseline="middle",
        dx=-10,
        dy=-10,
    ).encode(
        y=alt.Y("cum:Q", axis=None),
        # we'll use the percentage as the text
        text=alt.Text("cum:Q", format="0.0%"),
        color=alt.value("#000000"),
    )
    # Layer all the elements together

    return (
        (bars + bar_text + line + points + point_text)
        .resolve_scale(y="independent")
        .properties(width=800, height=400)
    )


def make_bar_chart(detailed_reccs: pd.DataFrame, report_table_detailed: pd.DataFrame):
    """
    Returns a chart to visualise the distribution of kcal contribution for the detailed recommendations

    """
    keep = (
        detailed_reccs["Detailed recommendations"]
        .str.split(
            ",",
            expand=True,
        )
        .T
    )

    keep[keep.columns] = keep.apply(lambda x: x.str.strip())

    cat = keep.columns

    selected_df = []

    for i in cat:
        selected_df.append(
            report_table_detailed[report_table_detailed["product"].isin(keep[i])]
        )

    print(selected_df)

    select = pd.concat(selected_df)

    sort_order = select.sort_values(by=["kcal_contribution_share"], ascending=False)[
        "product"
    ].to_list()

    return (
        alt.Chart(select)
        .mark_bar()
        .encode(x=alt.X("product", sort=sort_order), y="kcal_contribution_share")
        .facet(facet="rst_4_market", columns=3)
        .resolve_scale(x="independent", y="independent")
    )


if __name__ == "__main__":

    # Defining categories
    broad_cat_str = "rst_4_market"
    granular_cat_str = "rst_4_extended"

    webdr = google_chrome_driver_setup()

    logging.info("Reading data")
    report_table_clean = (
        load_s3_data(
            "ahl-private-data",
            "kantar/data_outputs/decision_table/decision_table_"
            + broad_cat_str
            + ".csv",
            # + "_reduced.csv",
        )
        .drop(axis=1, labels=["chosen_unit"])
        .melt(id_vars=broad_cat_str)
        .assign(clean_label=lambda df: df["variable"].map(clean_variable_names))
        .reset_index(drop=True)
        .pivot_table(index=broad_cat_str, columns="clean_label", values="value")[
            var_order
        ]
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
        # We focus on the variables that are not supercorrelated with each other
        report_table_clean[selected_vars]
        .apply(lambda x: zscore(x, nan_policy="omit"))
        .stack()
        .reset_index(name="z_score")
        .assign(category=lambda df: df["clean_label"].map(var_category_lookup))
    )

    report_table_clean_long["w"] = np.where(
        report_table_clean_long["category"] == "Feasibility", 0.125, 0.25
    )
    report_table_clean_long["cross"] = (
        report_table_clean_long["z_score"] * report_table_clean_long["w"]
    )

    # Reduce number to top 30 for plot
    reduced_cats = list(
        report_table_clean_long.groupby(["rst_4_market"])["cross"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .index
    )
    avg_table_plots = report_table_clean_long[
        report_table_clean_long["rst_4_market"].isin(reduced_cats)
    ]

    indicator_heatmap = (
        pipe(
            avg_table_plots,
            partial(relabel_names, columns=["clean_label"], clean_dict=plotting_names),
            partial(
                make_indicator_heatmap,
                axis_order=plotting_order,
                var_names=["clean_label", broad_cat_str, "cross"],
            ),
            configure_plots,
        )
        .configure_axis(labelLimit=1000, labelFontSize=13)
        .properties(width=200, height=700)
        .properties(width=200, height=800)
    )

    save_altair(indicator_heatmap, "indicator_heatmap", driver=webdr)

    logging.info("bubblechart with averaged indicators")
    aggr_bubble_chart = pipe(
        avg_table_plots.groupby(["category", broad_cat_str])[["cross", "z_score"]]
        .sum()
        .reset_index(drop=False),
        partial(
            make_indicator_bubblechart,
            axis_order=["Impact on Diets", "Feasibility", "Inclusion"],
            var_names=["category", broad_cat_str, "cross"],
        ),
    ).properties(width=200)

    save_altair(
        configure_plots(aggr_bubble_chart), "indicator_bubblechart_aggr", driver=webdr
    )

    indicator_bubble_chart = (
        pipe(
            avg_table_plots,
            partial(relabel_names, columns=["clean_label"], clean_dict=plotting_names),
            partial(
                make_indicator_bubblechart,
                axis_order=plotting_order,
                var_names=["clean_label", broad_cat_str, "cross"],
            ),
            configure_plots,
        )
        .configure_axis(labelLimit=1000, labelFontSize=13)
        .properties(width=300, height=700)
    )

    save_altair(
        configure_plots(indicator_bubble_chart), "indicator_bubblechart", driver=webdr
    )

    logging.info("Making high level recommendations")
    recc_table = make_recommendations(report_table_clean_long, report_table_clean, 37).T

    logging.info(recc_table.head())

    recc_table.to_csv(f"{PROJECT_DIR}/outputs/reports/recommendation.csv")

    logging.info("Implementing alternative decision-making criterion")

    # Creates table for visualisation
    seq_table = pipe(
        report_table_clean_long.groupby([broad_cat_str, "category"])["z_score"]
        .mean()
        .reset_index(name="z_score"),
        make_sequential_table,
    )

    # Take top 30 products
    seq_plot_table = seq_table[
        seq_table[broad_cat_str].isin(
            (
                seq_table.query(f"category=='Inclusion'")
                .sort_values("z_score_filtered", ascending=False)[broad_cat_str]
                .head(30)
                .tolist()
            )
        )
    ].copy()

    save_altair(
        configure_plots(make_selection_chart(seq_plot_table)),
        "indicator_sequential_decision",
        driver=webdr,
    )

    # The top criteria would be...
    top_sequential_cats = (
        seq_table.query("category=='Inclusion'")
        .dropna(axis=0, subset=["color"])[broad_cat_str]
        .tolist()
    )
    logging.info(f"Top sequential market sectors: {top_sequential_cats}")

    logging.info("pareto chart")

    pareto_chart = make_pareto_chart(report_table_clean_long, report_table_clean)

    save_altair(configure_plots(pareto_chart), "pareto_chart", driver=webdr)

    logging.info("Making detailed recommendations")

    report_table_detailed = pipe(
        load_s3_data(
            "ahl-private-data",
            "kantar/data_outputs/decision_table/decision_table_"
            + granular_cat_str
            + ".csv",
            # + "_reduced.csv",
        ).rename(columns={"Unnamed: 0": "product"}),
        lambda df: df.rename(
            columns={
                v: clean_variable_names[v] if v in clean_variable_names.keys() else v
                for v in df.columns
            }
        ),
    )[["product", broad_cat_str] + selected_vars]

    top_cats = recc_table.loc["Average", "Top candidates"].split(", ")

    detailed_reccs = pd.DataFrame(
        make_detailed_recommendations(top_cats, report_table_detailed, broad_cat_str),
        index=["Detailed recommendations"],
    ).T

    logging.info(detailed_reccs.head())

    logging.info("Bar chart for detailed reccs")

    bar_chart = make_bar_chart(detailed_reccs, report_table_detailed)

    save_altair(configure_plots(bar_chart), "bar_chart", driver=webdr)

    detailed_reccs.to_csv(f"{PROJECT_DIR}/outputs/reports/detailed_recommendation.csv")

    detailed_reccs_sec = pd.DataFrame(
        make_detailed_recommendations(
            top_sequential_cats, report_table_detailed, broad_cat_str
        ),
        index=["Detailed recommendations"],
    ).T

    logging.info(detailed_reccs_sec.head())

    # Save an easier to parse version for the macro nutrient chart

    with open(f"{PROJECT_DIR}/outputs/reports/detailed_products.json", "w") as outfile:
        json.dump(
            detailed_reccs["Detailed recommendations"].str.split(", ").to_dict(),
            outfile,
        )

    # select top 10

    drop = ["Total Bread", "Margarine", "Cooking Oils", "Total Milk"]
    top10 = detailed_reccs[~detailed_reccs.index.isin(drop)]

    with open(
        f"{PROJECT_DIR}/outputs/reports/detailed_products_10.json", "w"
    ) as outfile:
        json.dump(
            top10["Detailed recommendations"].str.split(", ").to_dict(),
            outfile,
        )

    # select top 3

    keep = ["Ambient Cakes+Pastries", "Chocolate Confectionery", "Everyday Biscuits"]
    top3 = detailed_reccs[detailed_reccs.index.isin(keep)]

    with open(
        f"{PROJECT_DIR}/outputs/reports/detailed_products_3.json", "w"
    ) as outfile:
        json.dump(
            top3["Detailed recommendations"].str.split(", ").to_dict(),
            outfile,
        )
