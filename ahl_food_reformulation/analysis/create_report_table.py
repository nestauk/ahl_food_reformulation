# Import libraries
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import report_tables as report
from ahl_food_reformulation.pipeline.cluster_analysis import cluster_table
import logging
from pathlib import Path
from ahl_food_reformulation import PROJECT_DIR

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
    cl_kcal_share = kantar.cluster_kcal_share()
    cl_adj_size = kantar.cluster_adj_size()

    # Defining categories
    # 2827 = market, 2828 = market sector
    broad_cat_number = 2827
    broad_cat_str = "rst_4_market"
    granular_cat_num = 2907
    granular_cat_str = "rst_4_extended"

    logging.info("Creating tables")
    # Create tables
    logging.info("Granular")
    granular_table = report.create_report_table(
        report.kcal_contr_table(
            granular_cat_str, pan_ind, pur_recs, nut_recs, prod_meta, panel_weight
        ),
        report.kcal_density_table(
            granular_cat_str,
            pur_recs,
            nut_recs,
            prod_meta,
            prod_meas,
            50,  # sample size
        ),
        cluster_table(
            val_fields,
            pur_recs,
            prod_codes,
            prod_vals,
            nut_recs,
            prod_meta,
            panel_weight,
            cl_kcal_share,
            cl_adj_size,
            pan_ind,
            granular_cat_num,
            granular_cat_str,
            sig_level=0.05,
            top=0.25,
        ),
    )
    logging.info("Broader")
    broader_table = report.create_report_table(
        report.kcal_contr_table(
            broad_cat_str, pan_ind, pur_recs, nut_recs, prod_meta, panel_weight
        ),
        report.kcal_density_table(
            broad_cat_str,
            pur_recs,
            nut_recs,
            prod_meta,
            prod_meas,
            50,  # sample size
        ),
        cluster_table(
            val_fields,
            pur_recs,
            prod_codes,
            prod_vals,
            nut_recs,
            prod_meta,
            panel_weight,
            cl_kcal_share,
            cl_adj_size,
            pan_ind,
            broad_cat_number,
            broad_cat_str,
            sig_level=0.05,
            top=0.25,
        ),
    )
    # Unique categories combined
    unique_cats = (
        prod_meta[[broad_cat_str, granular_cat_str]]
        .drop_duplicates(subset=[granular_cat_str])
        .set_index(granular_cat_str)
    )

    # If using rst market sector cat then comment this section out
    # Unique categories combined
    broader_table.index.names = [broad_cat_str]
    unique_cats_broad = prod_meta[
        ["rst_4_market_sector", broad_cat_str]
    ].drop_duplicates(subset=[broad_cat_str])
    broader_table = broader_table.reset_index().merge(
        unique_cats_broad, how="left", on=broad_cat_str
    )

    # Removing cat with less than 50 products (broader cat)
    broader_table = broader_table[broader_table["number_products"] > 25].copy()

    logging.info("Saving tables")
    # Save tables
    Path(f"{PROJECT_DIR}/outputs/data/decision_table/").mkdir(
        parents=True, exist_ok=True
    )
    granular_table.merge(unique_cats, left_index=True, right_index=True).to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/decision_table_"
        + granular_cat_str
        + ".csv",
        float_format="%.3f",
    )
    broader_table.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/decision_table_"
        + broad_cat_str
        + ".csv",
        float_format="%.3f",
        index=False,
    )
