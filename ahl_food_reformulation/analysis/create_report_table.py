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

    # Define categories
    granular_category = "rst_4_extended"  # Granular category
    broader_category = "rst_4_market_sector"  # Broader category

    logging.info("Creating tables")
    # Create tables
    logging.info("RST extended table (granular)")
    granular_table = report.create_report_table(
        report.kcal_contr_table(
            granular_category, pan_ind, pur_recs, nut_recs, prod_meta, panel_weight
        ),
        report.kcal_density_table(
            granular_category,
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
            att_num=2907,
            sig_level=0.05,
            top=0.25,
        ),
    )
    logging.info("RST market sector table (broader)")
    broader_table = report.create_report_table(
        report.kcal_contr_table(
            broader_category, pan_ind, pur_recs, nut_recs, prod_meta, panel_weight
        ),
        report.kcal_density_table(
            broader_category,
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
            att_num=2828,
            sig_level=0.05,
            top=0.25,
        ),
    )
    # Unique categories combined
    unique_cats = (
        prod_meta[[broader_category, granular_category]]
        .drop_duplicates(subset=[granular_category])
        .set_index(granular_category)
    )

    # Removing cat with less than 50 products (broader cat)
    broader_table = broader_table[broader_table["number_products"] > 50].copy()

    logging.info("Saving tables")
    # Save tables
    Path(f"{PROJECT_DIR}/outputs/data/decision_table/").mkdir(
        parents=True, exist_ok=True
    )
    granular_table.merge(unique_cats, left_index=True, right_index=True).to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/decision_table_"
        + granular_category
        + ".csv",
        float_format="%.3f",
    )
    broader_table.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/decision_table_"
        + broader_category
        + ".csv",
        float_format="%.3f",
    )
