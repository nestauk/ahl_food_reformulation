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
    kcal_est = kantar.kcal_reduction()

    # define cats
    broader_category = "rst_4_market_sector"
    granular_category = "rst_4_extended"

    # read in shortliested products
    with open(f"{PROJECT_DIR}/outputs/reports/detailed_products.json") as f:
        chosen_cats_average = pd.DataFrame(json.load(f)).melt(
            var_name=broader_category, value_name=granular_category
        )

    with open(f"{PROJECT_DIR}/outputs/reports/detailed_products_sequential.json") as f:
        # with open(f"{PROJECT_DIR}/outputs/reports/detailed_products_average.json") as f:
        chosen_cats_seq = pd.DataFrame(json.load(f)).melt(
            var_name=broader_category, value_name=granular_category
        )

    logging.info("Merging data - Averages")

    purch_recs_comb_scenarios_avg = kcal.make_impact(
        chosen_cats_average,
        kcal_est,
        val_fields,
        pur_recs,
        prod_codes,
        prod_vals,
        prod_meta,
        nut_rec,
    )

    logging.info("Merging data - Sequential")

    purch_recs_comb_scenarios_seq = kcal.make_impact(
        chosen_cats_seq,
        kcal_est,
        val_fields,
        pur_recs,
        prod_codes,
        prod_vals,
        prod_meta,
        nut_rec,
    )

    hh_kcal_filter_avg = pd.concat(
        [
            kcal.kcal_day(
                purch_recs_comb_scenarios_avg,
                pan_ind,
                panel_weight,
                "Gross_up_kcal",
                0.05,
                0.95,
            ),
            kcal.kcal_day(
                purch_recs_comb_scenarios_avg,
                pan_ind,
                panel_weight,
                "Gross_up_kcal_min",
                0.05,
                0.95,
            ),
            kcal.kcal_day(
                purch_recs_comb_scenarios_avg,
                pan_ind,
                panel_weight,
                "Gross_up_kcal_max",
                0.05,
                0.95,
            ),
        ],
        axis=1,
    )

    hh_kcal_filter_seq = pd.concat(
        [
            kcal.kcal_day(
                purch_recs_comb_scenarios_seq,
                pan_ind,
                panel_weight,
                "Gross_up_kcal",
                0.05,
                0.95,
            ),
            kcal.kcal_day(
                purch_recs_comb_scenarios_seq,
                pan_ind,
                panel_weight,
                "Gross_up_kcal_min",
                0.05,
                0.95,
            ),
            kcal.kcal_day(
                purch_recs_comb_scenarios_seq,
                pan_ind,
                panel_weight,
                "Gross_up_kcal_max",
                0.05,
                0.95,
            ),
        ],
        axis=1,
    )

    driver = google_chrome_driver_setup()

    df = pd.read_csv(f"{PROJECT_DIR}/outputs/data/impact_on_kcal.csv")

    df.columns = ["index", "0% Reformulation", "5% Reformulation", "10% Reformulation"]

    dfT = df.T
    dfT.columns = dfT.iloc[0]
    dfT = dfT[1:]
    pl_df = dfT.reset_index()

    fig1 = (
        alt.LayerChart(pl_df)
        .encode(
            x=alt.X("index:N", title=""),
            tooltip=["min:Q", "25%:Q", "mean:Q", "75%:Q", "max:Q"],
        )
        .add_layers(
            alt.Chart().mark_rule().encode(y="min:Q", y2="max:Q"),
            alt.Chart()
            .mark_bar(width=15, color=pu.NESTA_COLOURS[0])
            .encode(y="25%:Q", y2="75%:Q"),
            alt.Chart().mark_tick(color="white", width=15).encode(y="mean:Q"),
        )
    )

    fig1 = configure_plots(fig1, "Distribution of Daily Calorie Intake", "", 16, 20, 16)

    save_altair(
        altair_text_resize(fig1).properties(width=250, height=250),
        "boxplot",
        driver=driver,
    )

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

    fig2 = (
        alt.Chart(diff_avg)
        .mark_line()
        .encode(
            x=alt.X("index", axis=alt.Axis(format="%", title="Percentile")),
            y=alt.Y("value", axis=alt.Axis(title="kcal/day")),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig2 = configure_plots(
        fig2,
        "Distribution of Reduction in Per Capita Daily Calorie Compared to No Reformulation",
        "Average",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig2).properties(width=250, height=250),
        "cumplot_average",
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

    fig3 = (
        alt.Chart(diff_seq)
        .mark_line()
        .encode(
            x=alt.X("index", axis=alt.Axis(format="%", title="Percentile")),
            y=alt.Y("value", axis=alt.Axis(title="kcal/day")),
            color=alt.Color(
                "Reformulation Level",
                scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])},
            ),
        )
    )

    fig3 = configure_plots(
        fig3,
        "Distribution of Reduction in DPer Capita aily Calorie Compared to No Reformulation",
        "Sequential",
        16,
        20,
        16,
    )

    save_altair(
        altair_text_resize(fig3).properties(width=250, height=250),
        "cumplot_sequential",
        driver=driver,
    )

    # Save distribution dfs for avg and seq
    diff_avg.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/distribution_reduction_avg.csv",
        index=False,
    )
    diff_seq.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/distribution_reduction_seq.csv",
        index=False,
    )
