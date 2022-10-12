# Import libraries
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import report_tables as report
import logging
from pathlib import Path
from ahl_food_reformulation import PROJECT_DIR

if __name__ == "__main__":

    logging.info("Reading data")
    # read data
    pur_recs = kantar.purchase_records_updated()
    nut_recs = kantar.nutrition()
    prod_mast = kantar.product_master()
    val_fields = kantar.val_fields()
    uom = kantar.uom()
    prod_meta = kantar.product_metadata()
    prod_meas = kantar.product_measurement()
    prod_codes = kantar.product_codes()
    prod_vals = kantar.product_values()
    pan_ind = kantar.household_ind()

    # Define categories
    granular_category = [2907, "rst_4_extended"]  # Granular category
    broader_category = [2828, "rst_4_market_sector"]  # Broader category

    logging.info("Creating tables")
    # Create tables
    logging.info("RST extended table (granular)")
    granular_table = report.create_report_table(
        report.kcal_contr_table(
            granular_category[0],
            pan_ind,
            val_fields,
            pur_recs,
            prod_codes,
            prod_vals,
            nut_recs,
        ),
        report.kcal_density_table(
            granular_category[1],
            pur_recs,
            nut_recs,
            prod_meta,
            prod_meas,
            50,  # sample size
        ),
    )
    logging.info("RST market sector table (broader)")
    broader_table = report.create_report_table(
        report.kcal_contr_table(
            broader_category[0],
            pan_ind,
            val_fields,
            pur_recs,
            prod_codes,
            prod_vals,
            nut_recs,
        ),
        report.kcal_density_table(
            broader_category[1],
            pur_recs,
            nut_recs,
            prod_meta,
            prod_meas,
            50,  # sample size
        ),
    )
    # Unique categories combined
    unique_cats = (
        prod_meta[[broader_category[1], granular_category[1]]]
        .drop_duplicates(subset=[granular_category[1]])
        .set_index(granular_category[1])
    )

    logging.info("Saving tables")
    # Save tables
    Path(f"{PROJECT_DIR}/outputs/data/decision_table/").mkdir(
        parents=True, exist_ok=True
    )
    granular_table.merge(unique_cats, left_index=True, right_index=True).to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/table_"
        + granular_category[1]
        + ".csv",
        float_format="%.3f",
    )
    broader_table.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/table_"
        + broader_category[1]
        + ".csv",
        float_format="%.3f",
    )
