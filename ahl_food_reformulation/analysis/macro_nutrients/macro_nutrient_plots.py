# Import libraries
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import nutrient_metrics_funcs as nutrient
from ahl_food_reformulation.utils.altair_save_utils import (
    save_altair,
    google_chrome_driver_setup,
    altair_text_resize,
)
import logging
import json
import pandas as pd
from ahl_food_reformulation import PROJECT_DIR

if __name__ == "__main__":
    # Set driver for altair saving
    driver = google_chrome_driver_setup()

    logging.info("Reading data")
    # Read data
    nut_recs = kantar.nutrition()
    pur_recs = kantar.purchase_records_updated()
    prod_meta = kantar.prod_meta_update()

    # define cats
    broader_category = "rst_4_market_sector"
    granular_category = "rst_4_extended"

    # Get chosen categories as dataframe
    with open(f"{PROJECT_DIR}/outputs/reports/detailed_products.json") as f:
        chosen_cats = pd.DataFrame(json.load(f)).melt(
            var_name=broader_category, value_name=granular_category
        )

    logging.info("Creating tables")
    # Create tables
    broad_macro_nut = nutrient.macro_nutrient_table(
        pur_recs, prod_meta, nut_recs, broader_category
    )
    gran_macro_nut = nutrient.macro_nutrient_table(
        pur_recs, prod_meta, nut_recs, granular_category
    )

    # Broad cats df
    # Unique list of broad cats
    broad_cats = list(chosen_cats.rst_4_market_sector.drop_duplicates())
    broad_macro_nut_subset = broad_macro_nut[
        broad_macro_nut.rst_4_market_sector.isin(broad_cats)
    ][["rst_4_market_sector", "Carb_prop", "Prot_prop", "Fat_prop"]].copy()
    broad_macro_nut_subset.columns = ["Categories", "Carbohydrates", "Protein", "Fat"]
    broad_plot_df = broad_macro_nut_subset.melt(
        id_vars=["Categories"], var_name="Macro nutrients", value_name="proportions"
    )
    # Granular cats df
    gran_macro_nut_subset = chosen_cats.merge(
        gran_macro_nut[["rst_4_extended", "Carb_prop", "Prot_prop", "Fat_prop"]],
        on="rst_4_extended",
        how="left",
    )
    gran_macro_nut_subset.columns = [
        "Market sector",
        "Categories",
        "Carbohydrates",
        "Protein",
        "Fat",
    ]
    gran_plot_df = gran_macro_nut_subset.melt(
        id_vars=["Market sector", "Categories"],
        var_name="Macro nutrients",
        value_name="proportions",
    )
    # Divide by 100 so can be displayed in % format in plot
    gran_plot_df["proportions"] = gran_plot_df["proportions"] / 100
    broad_plot_df["proportions"] = broad_plot_df["proportions"] / 100

    logging.info("Creating plots")
    # Produce plots
    fig_broad, fig_gran = nutrient.plot_macro_proportions(
        broad_plot_df, gran_plot_df, driver, broad_cats
    )

    logging.info("Saving plots")
    # Save plots
    save_altair(
        altair_text_resize(fig_broad).properties(width=250, height=250),
        "macro_prop_broad",
        driver=driver,
    )
    save_altair(
        altair_text_resize(fig_gran)
        .properties(width=200, height=150)
        .resolve_scale(x="independent", y="independent"),
        "macro_prop_gran",
        driver=driver,
    )
