# Import libraries
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import report_tables as report
import logging
from pathlib import Path
from ahl_food_reformulation import PROJECT_DIR

if __name__ == "__main__":

    logging.info("Reading data")
    # read data
    pur_recs = kantar.purchase_records()
    nut_recs = kantar.nutrition()
    prod_mast = kantar.product_master()
    val_fields = kantar.val_fields()
    uom = kantar.uom()
    prod_meta = kantar.product_metadata()
    prod_meas = kantar.product_measurement()

    # Define categories
    granular_category = "rst_4_extended"  # Granular category
    broader_category = "rst_4_market_sector"  # Broader category

    logging.info("Creating tables")
    # Create tables
    granular_table = report.create_pop_table_cat(
        granular_category,
        val_fields,
        prod_mast,
        uom,
        pur_recs,
        nut_recs,
        prod_meta,
        prod_meas,
    )
    broader_table = report.create_pop_table_cat(
        broader_category,
        val_fields,
        prod_mast,
        uom,
        pur_recs,
        nut_recs,
        prod_meta,
        prod_meas,
    )
    unique_cats = prod_meta[[broader_category, granular_category]].drop_duplicates(
        subset=[granular_category]
    )

    logging.info("Saving tables")
    # Save tables
    Path(f"{PROJECT_DIR}/outputs/data/decision_table/").mkdir(
        parents=True, exist_ok=True
    )
    granular_table.merge(unique_cats, on=granular_category).to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/table_"
        + granular_category
        + ".csv",
        index=False,
    )
    broader_table.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/table_" + broader_category + ".csv",
        index=False,
    )
