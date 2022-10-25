# Import libraries
from ahl_food_reformulation.getters import kantar
import logging
import json
import pandas as pd
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.pipeline import kcal_impact as kcal


if __name__ == "__main__":

    logging.info("Reading data")

    pur_recs = kantar.purchase_records_updated()
    nut_rec = kantar.nutrition()
    pan_ind = kantar.household_ind()
    prod_mast = kantar.product_master()
    val_fields = kantar.val_fields()
    uom = kantar.uom()
    prod_codes = kantar.product_codes()
    prod_vals = kantar.product_values()
    panel_weight = kantar.panel_weights_year()
    cl_kcal_share = kantar.cluster_kcal_share()
    cl_adj_size = kantar.cluster_adj_size()
    prod_meta = kantar.prod_meta_update()
    kcal_est = kantar.kcal_reduction()

    # define cats
    att_num = 2907  # code corresponding to rst_4_extended
    broader_category = "rst_4_market_sector"
    granular_category = "rst_4_extended"

    # read in shortliested products
    with open(f"{PROJECT_DIR}/outputs/reports/detailed_products.json") as f:
        chosen_cats = pd.DataFrame(json.load(f)).melt(
            var_name=broader_category, value_name=granular_category
        )

    logging.info("Merging data")
    # merge files to get calorie reduction for product categories
    target_red = chosen_cats.merge(kcal_est, on="rst_4_market_sector")

    # generate conversion factor
    pan_conv = transform.hh_size_conv(pan_ind)

    # Purchase and product info combined
    comb_files = transform.combine_files(
        val_fields, pur_recs, prod_codes, prod_vals, att_num
    ).drop("att_vol", axis=1)

    comb_update = comb_files.merge(
        prod_meta[["product_code", "rst_4_extended", "rst_4_market_sector"]],
        left_on="Product Code",
        right_on="product_code",
    )

    comb_update.rename(columns={"rst_4_extended": "att_vol"}, inplace=True)

    # sum of all kcal purchased by category by an household
    purch_recs_comb = transform.make_purch_records(nut_rec, comb_update, ["att_vol"])

    purch_recs_comb_scenarios = purch_recs_comb.merge(
        target_red, right_on="rst_4_extended", left_on="att_vol", how="left"
    ).fillna(0)

    # remove outliers
    # q_low = purch_recs_comb_scenarios["Gross_up_kcal"].quantile(0.05)
    # q_hi  = purch_recs_comb_scenarios["Gross_up_kcal"].quantile(0.95)

    # pur_rec_filter = purch_recs_comb_scenarios[(purch_recs_comb_scenarios["Gross_up_kcal"] < q_hi) & (purch_recs_comb_scenarios["Gross_up_kcal"] > q_low)].copy()

    purch_recs_comb_scenarios["Gross_up_kcal_min"] = purch_recs_comb_scenarios[
        "Gross_up_kcal"
    ] * (1 - purch_recs_comb_scenarios["min"])
    purch_recs_comb_scenarios["Gross_up_kcal_max"] = purch_recs_comb_scenarios[
        "Gross_up_kcal"
    ] * (1 - purch_recs_comb_scenarios["max"])

    logging.info("Generating Descriptive Stats")

    pd.concat(
        [
            kcal.kcal_day(
                purch_recs_comb_scenarios, pan_conv, panel_weight, "Gross_up_kcal"
            ),
            kcal.kcal_day(
                purch_recs_comb_scenarios, pan_conv, panel_weight, "Gross_up_kcal_min"
            ),
            kcal.kcal_day(
                purch_recs_comb_scenarios, pan_conv, panel_weight, "Gross_up_kcal_max"
            ),
        ],
        axis=1,
    ).to_csv(
        f"{PROJECT_DIR}/outputs/data/impact_on_kcal.csv",
        float_format="%.3f",
    )
