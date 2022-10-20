# Import libraries
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import nutrient_metrics_funcs as nutrient
import logging
from pathlib import Path
from ahl_food_reformulation import PROJECT_DIR

if __name__ == "__main__":

    logging.info("Reading data")
    # Read data
    nut_recs = kantar.nutrition()
    pur_recs = kantar.purchase_records_updated()
    prod_meta = kantar.prod_meta_update()

    # Define categories
    granular_category = "rst_4_extended"  # Granular category
    broader_category = "rst_4_market_sector"  # Broader category

    logging.info("Creating tables")
    # Create tables
    broad_macro_nut = nutrient.macro_nutrient_table(
        pur_recs, prod_meta, nut_recs, broader_category
    )
    gran_macro_nut = nutrient.macro_nutrient_table(
        pur_recs, prod_meta, nut_recs, granular_category
    )

    logging.info("Saving files")
    # Create folder if doesn't exist
    Path(f"{PROJECT_DIR}/outputs/data/decision_table/").mkdir(
        parents=True, exist_ok=True
    )
    # Saving files
    broad_macro_nut.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/macro_nutrients_"
        + broader_category
        + ".csv",
        float_format="%.3f",
        index=False,
    )
    gran_macro_nut.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/macro_nutrients_"
        + granular_category
        + ".csv",
        float_format="%.3f",
        index=False,
    )
