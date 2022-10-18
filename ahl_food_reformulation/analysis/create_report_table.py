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
    prod_meta = kantar.prod_meta_update()
    prod_meas = kantar.product_measurement()
    pan_ind = kantar.household_ind()

    # Define categories
    granular_category = "rst_4_extended"  # Granular category
    broader_category = "rst_4_market_sector"  # Broader category

    logging.info("Creating tables")
    # Create tables
    logging.info("RST extended table (granular)")
    granular_table = report.create_report_table(
        report.kcal_contr_table(
            granular_category, pan_ind, pur_recs, nut_recs, prod_meta
        ),
        report.kcal_density_table(
            granular_category,
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
            broader_category, pan_ind, pur_recs, nut_recs, prod_meta
        ),
        report.kcal_density_table(
            broader_category,
            pur_recs,
            nut_recs,
            prod_meta,
            prod_meas,
            50,  # sample size
        ),
    )
    # Unique categories combined
    unique_cats = (
        prod_meta[[broader_category, granular_category]]
        .drop_duplicates(subset=[granular_category])
        .set_index(granular_category)
    )

    logging.info("Saving tables")
    # Save tables
    Path(f"{PROJECT_DIR}/outputs/data/decision_table/").mkdir(
        parents=True, exist_ok=True
    )
    granular_table.merge(unique_cats, left_index=True, right_index=True).to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/table_"
        + granular_category
        + ".csv",
        float_format="%.3f",
    )
    broader_table.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/table_" + broader_category + ".csv",
        float_format="%.3f",
    )
