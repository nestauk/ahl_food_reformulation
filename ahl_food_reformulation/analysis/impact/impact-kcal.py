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
        chosen_cats = pd.DataFrame(json.load(f)).melt(
            var_name=broader_category, value_name=granular_category
        )

    logging.info("Merging data")

    purch_recs_comb_scenarios = kcal.make_impact(
        chosen_cats,
        kcal_est,
        pan_ind,
        val_fields,
        pur_recs,
        prod_codes,
        prod_vals,
        prod_meta,
        nut_rec,
    )

    logging.info("Generating Descriptive Stats")

    pd.concat(
        [
            kcal.kcal_day(
                purch_recs_comb_scenarios,
                pan_ind,
                panel_weight,
                "Gross_up_kcal",
                0.05,
                0.95,
            ),
            kcal.kcal_day(
                purch_recs_comb_scenarios,
                pan_ind,
                panel_weight,
                "Gross_up_kcal_min",
                0.05,
                0.95,
            ),
            kcal.kcal_day(
                purch_recs_comb_scenarios,
                pan_ind,
                panel_weight,
                "Gross_up_kcal_max",
                0.05,
                0.95,
            ),
        ],
        axis=1,
    ).to_csv(
        f"{PROJECT_DIR}/outputs/data/impact_on_kcal.csv",
        float_format="%.3f",
    )
