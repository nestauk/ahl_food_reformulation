import altair as alt
from ahl_food_reformulation.utils.altair_save_utils import (
    save_altair,
    google_chrome_driver_setup,
    altair_text_resize,
)

from ahl_food_reformulation import PROJECT_DIR
import pandas as pd
from ahl_food_reformulation.utils.plotting import configure_plots
from ahl_food_reformulation.utils import plotting as pu
from ahl_food_reformulation.getters import kantar
import logging
import json
from ahl_food_reformulation.pipeline import kcal_impact as kcal
import numpy as np

alt.data_transformers.disable_max_rows()


if __name__ == "__main__":

    logging.info("Reading data")

    pur_recs = kantar.purchase_records_updated()
    nut_rec = kantar.nutrition()
    pan_ind = kantar.household_ind()
    val_fields = kantar.val_fields()
    uom = kantar.uom()
    prod_codes = kantar.product_codes()
    prod_vals = kantar.product_values()
    panel_weight = kantar.panel_weights_year()
    prod_meta = kantar.prod_meta_update()
    # kcal_est = kantar.kcal_reduction()

    # define cats
    broader_category = "rst_4_market"
    granular_category = "rst_4_extended"

    with open(f"{PROJECT_DIR}/outputs/reports/detailed_products.json") as f:
        chosen_cats_average = (
            pd.DataFrame.from_dict(json.load(f), orient="index")
            .transpose()
            .melt(var_name=broader_category, value_name=granular_category)
            .dropna()
        )

    with open(f"{PROJECT_DIR}/outputs/reports/detailed_products_sequential.json") as f:
        chosen_cats_seq = (
            pd.DataFrame.from_dict(json.load(f), orient="index")
            .transpose()
            .melt(var_name=broader_category, value_name=granular_category)
            .dropna()
        )

    logging.info("Merging data - Averages")

    purch_recs_comb_scenarios_avg = kcal.make_impact(
        chosen_cats_average,
        # kcal_est,
        val_fields,
        pur_recs,
        prod_codes,
        prod_vals,
        prod_meta,
        nut_rec,
        broader_category,
        granular_category,
        2907,
    )

    hh_kcal_filter_avg = kcal.compare_scenarios(
        panel_weight, purch_recs_comb_scenarios_avg, pan_ind
    )

    logging.info("Merging data - Sequential")

    purch_recs_comb_scenarios_seq = kcal.make_impact(
        chosen_cats_seq,
        # kcal_est,
        val_fields,
        pur_recs,
        prod_codes,
        prod_vals,
        prod_meta,
        nut_rec,
        broader_category,
        granular_category,
        2907,
    )

    hh_kcal_filter_seq = kcal.compare_scenarios(
        panel_weight, purch_recs_comb_scenarios_seq, pan_ind
    )

    driver = google_chrome_driver_setup()

    logging.info("Option 1 - negative difference and deciles")

    no_reform_avg = hh_kcal_filter_avg["Gross_up_kcal_daily"].quantile(
        [x / 10 for x in range(11)]
    )
    low_reform_avg = hh_kcal_filter_avg["Gross_up_kcal_min_daily"].quantile(
        [x / 10 for x in range(11)]
    )
    high_reform_avg = hh_kcal_filter_avg["Gross_up_kcal_max_daily"].quantile(
        [x / 10 for x in range(11)]
    )

    diff_high_avg = pd.DataFrame(
        high_reform_avg - no_reform_avg, columns=["value"]
    ).reset_index()
    diff_high_avg["Reformulation Level"] = "10%"
    diff_low_avg = pd.DataFrame(
        low_reform_avg - no_reform_avg, columns=["value"]
    ).reset_index()
    diff_low_avg["Reformulation Level"] = "5%"

    diff_avg = pd.concat([diff_high_avg, diff_low_avg])

    fig1a = (
        alt.Chart(diff_avg)
        .mark_line()
        .encode(
            x=alt.X("index", axis=alt.Axis(format="%", title="Percentile")),
            y=alt.Y(
                "value",
                axis=alt.Axis(title="kcal/day"),
                scale=alt.Scale(domain=[-150, 0]),
            ),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig1a = configure_plots(
        fig1a,
        "",
        "",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig1a).properties(width=250, height=250),
        "cumplot_average_opt1",
        driver=driver,
    )

    no_reform_seq = hh_kcal_filter_seq["Gross_up_kcal_daily"].quantile(
        [x / 10 for x in range(11)]
    )
    low_reform_seq = hh_kcal_filter_seq["Gross_up_kcal_min_daily"].quantile(
        [x / 10 for x in range(11)]
    )
    high_reform_seq = hh_kcal_filter_seq["Gross_up_kcal_max_daily"].quantile(
        [x / 10 for x in range(11)]
    )

    diff_high_seq = pd.DataFrame(
        high_reform_seq - no_reform_seq, columns=["value"]
    ).reset_index()
    diff_high_seq["Reformulation Level"] = "10%"
    diff_low_seq = pd.DataFrame(
        low_reform_seq - no_reform_seq, columns=["value"]
    ).reset_index()
    diff_low_seq["Reformulation Level"] = "5%"

    diff_seq = pd.concat([diff_high_seq, diff_low_seq])

    fig1b = (
        alt.Chart(diff_seq)
        .mark_line()
        .encode(
            x=alt.X("index", axis=alt.Axis(format="%", title="Percentile")),
            y=alt.Y(
                "value",
                axis=alt.Axis(title="kcal/day"),
                scale=alt.Scale(domain=[-150, 0]),
            ),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig1b = configure_plots(
        fig1b,
        "",
        "",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig1b).properties(width=250, height=250),
        "cumplot_sequential_opt1",
        driver=driver,
    )

    logging.info("Option 2 - positive difference and deciles")

    diff_high_avg = pd.DataFrame(
        no_reform_avg - high_reform_avg, columns=["value"]
    ).reset_index()
    diff_high_avg["Reformulation Level"] = "10%"
    diff_low_avg = pd.DataFrame(
        no_reform_avg - low_reform_avg, columns=["value"]
    ).reset_index()
    diff_low_avg["Reformulation Level"] = "5%"

    diff_avg = pd.concat([diff_high_avg, diff_low_avg])

    fig2a = (
        alt.Chart(diff_avg)
        .mark_line()
        .encode(
            x=alt.X("index", axis=alt.Axis(format="%", title="Percentile")),
            y=alt.Y(
                "value",
                axis=alt.Axis(title="kcal/day"),
                scale=alt.Scale(domain=[0, 150]),
            ),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig2a = configure_plots(
        fig2a,
        "",
        "",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig2a).properties(width=250, height=250),
        "cumplot_average_opt2",
        driver=driver,
    )

    diff_high_seq = pd.DataFrame(
        no_reform_seq - high_reform_seq, columns=["value"]
    ).reset_index()
    diff_high_seq["Reformulation Level"] = "10%"
    diff_low_seq = pd.DataFrame(
        no_reform_seq - low_reform_seq, columns=["value"]
    ).reset_index()
    diff_low_avg["Reformulation Level"] = "5%"

    diff_seq = pd.concat([diff_high_seq, diff_low_seq])

    fig2b = (
        alt.Chart(diff_avg)
        .mark_line()
        .encode(
            x=alt.X("index", axis=alt.Axis(format="%", title="Percentile")),
            y=alt.Y(
                "value",
                axis=alt.Axis(title="kcal/day"),
                scale=alt.Scale(domain=[0, 150]),
            ),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig2b = configure_plots(
        fig2b,
        "",
        "",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig2b).properties(width=250, height=250),
        "cumplot_sequential_opt2",
        driver=driver,
    )

    logging.info("Option 3 - cumulative count of hh over positive difference")

    cum_df_avg = pd.melt(
        hh_kcal_filter_avg,
        id_vars=["Panel Id", "demographic_weight"],
        value_vars=["pos_diff_5", "pos_diff_10"],
    )
    cum_df_avg["Reformulation Level"] = np.where(
        cum_df_avg["variable"] == "pos_diff_5", "5%", "10%"
    )

    fig3a = (
        alt.Chart(cum_df_avg)
        .transform_window(
            cumulative_count="sum(demographic_weight)",
            sort=[{"field": "value"}],
            groupby=["Reformulation Level"],
        )
        .mark_line()
        .encode(
            x=alt.X("value:Q", axis=alt.X(title="kcal/day")),
            y=alt.Y(
                "cumulative_count:Q", axis=alt.Axis(title="Cumulative Household Count")
            ),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig3a = configure_plots(
        fig3a,
        "",
        "",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig3a).properties(width=250, height=250),
        "cumulative_positive_avg",
        driver=driver,
    )

    cum_df_seq = pd.melt(
        hh_kcal_filter_seq,
        id_vars=["Panel Id", "demographic_weight"],
        value_vars=["pos_diff_5", "pos_diff_10"],
    )
    cum_df_seq["Reformulation Level"] = np.where(
        cum_df_seq["variable"] == "pos_diff_5", "5%", "10%"
    )

    fig3b = (
        alt.Chart(cum_df_seq)
        .transform_window(
            cumulative_count="sum(demographic_weight)",
            sort=[{"field": "value"}],
            groupby=["Reformulation Level"],
        )
        .mark_line()
        .encode(
            x=alt.X("value:Q", axis=alt.X(title="kcal/day")),
            y=alt.Y(
                "cumulative_count:Q", axis=alt.Axis(title="Cumulative Household Count")
            ),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig3b = configure_plots(
        fig3b,
        "",
        "",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig3b).properties(width=250, height=250),
        "cumulative_positive_seq",
        driver=driver,
    )

    logging.info("Option 4 - cumulative density")

    fig4a = (
        alt.Chart(cum_df_avg)
        .transform_density(
            "value",
            as_=["value", "density"],
            cumulative=True,
            groupby=["Reformulation Level"],
        )
        .mark_line()
        .encode(
            x=alt.X("value:Q", axis=alt.X(title="kcal/day")),
            y=alt.Y("density:Q", axis=alt.Axis(title="Cumulative Density")),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig4a = configure_plots(
        fig4a,
        "",
        "",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig4a).properties(width=250, height=250),
        "cumulative_density_avg",
        driver=driver,
    )

    fig4b = (
        alt.Chart(cum_df_seq)
        .transform_density(
            "value",
            as_=["value", "density"],
            cumulative=True,
            groupby=["Reformulation Level"],
        )
        .mark_line()
        .encode(
            x=alt.X("value:Q", axis=alt.X(title="kcal/day")),
            y=alt.Y("density:Q", axis=alt.Axis(title="Cumulative Density")),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig4b = configure_plots(
        fig4b,
        "",
        "",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig4b).properties(width=250, height=250),
        "cumulative_density_seq",
        driver=driver,
    )

    logging.info("Option 5 - Stochastic Dominance")
    # stochastic dominance

    cum_df_avg["Method"] = "Average"
    cum_df_seq["Method"] = "Sequential Filtering"

    df = pd.concat([cum_df_avg, cum_df_seq])

    fig5 = (
        alt.Chart(df)
        .transform_density(
            "value",
            as_=["value", "density"],
            cumulative=True,
            groupby=["Reformulation Level", "Method"],
        )
        .mark_line()
        .encode(
            x=alt.X("value:Q", axis=alt.X(title="kcal/day")),
            y=alt.Y("density:Q", axis=alt.Axis(title="Cumulative Density")),
            color=alt.Color(
                "Method",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in range(0, 2))},
            ),
        )
        .facet(facet="Reformulation Level", columns=2)
    )

    fig5 = configure_plots(
        fig5,
        "",
        "",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig5),
        "stochastic_dominance",
        driver=driver,
    )

    # Save distribution dfs for avg and seq
    cum_df_avg.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/distribution_reduction_avg.csv",
        index=False,
    )
    cum_df_seq.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/distribution_reduction_seq.csv",
        index=False,
    )
