# Import libraries
from ahl_food_reformulation.getters import kantar
import logging
import json
import pandas as pd
from ahl_food_reformulation import PROJECT_DIR
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

    hh_kcal_filter_avg = kcal.compare_scenarios(
        panel_weight, purch_recs_comb_scenarios_avg, pan_ind
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

    hh_kcal_filter_seq = kcal.compare_scenarios(
        panel_weight, purch_recs_comb_scenarios_seq, pan_ind
    )

    logging.info("Generating Descriptive Stats")

    kcal.kcal_day_describe(hh_kcal_filter_avg).reset_index(level=0).to_csv(
        f"{PROJECT_DIR}/outputs/data/impact_on_kcal_avg.csv",
        float_format="%.3f",
        index=False,
    )

    kcal.kcal_day_describe(hh_kcal_filter_seq).reset_index(level=0).to_csv(
        f"{PROJECT_DIR}/outputs/data/impact_on_kcal_seq.csv",
        float_format="%.3f",
        index=False,
    )
