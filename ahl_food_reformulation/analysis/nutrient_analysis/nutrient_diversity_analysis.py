# Import libraries
from unicodedata import category
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import energy_density as energy
from ahl_food_reformulation.pipeline import nutrient_metrics_funcs as nutrient
import logging
from pathlib import Path

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

    logging.info("Create energy density df")
    prod_cat = "rst_4_market_sector"  # Update to category of interest
    # run function with chosen cat
    df_prod_ed = energy.prod_energy_100(
        prod_cat,
        val_fields,
        prod_mast,
        uom,
        pur_recs,
        nut_recs,
        prod_meta,
        prod_meas,
    )[[prod_cat, "chosen_unit", "kcal_100g_ml"]].copy()

    logging.info("Create nutrient diversity df")
    # Removing 'Frozen Poultry+Game' as only 2 products
    to_keep = (
        df_prod_ed[prod_cat]
        .value_counts()[df_prod_ed[prod_cat].value_counts() > 50]
        .index
    )
    df_prod_ed_reduce = df_prod_ed[df_prod_ed[prod_cat].isin(to_keep)].copy()
    # Create nutrient diversity df
    df_diversity = nutrient.create_diversity_df(df_prod_ed_reduce, prod_cat, 10, 50)

    logging.info("Plot heatmaps")
    # Plot heatmaps
    nutrient.diversity_heatmaps(
        df_diversity, prod_cat, "entropy", "variance", "unsampled_values"
    )
    nutrient.diversity_heatmaps(
        df_diversity,
        prod_cat,
        "entropy_size_adj",
        "variance_size_adj",
        "sampled_values",
    )

    logging.info("Save table of results")
    # Save df to outputs/data
    # Add folder if not already created
    Path(f"{PROJECT_DIR}/outputs/data/nutrient_diversity/").mkdir(
        parents=True, exist_ok=True
    )
    df_diversity.to_csv(
        f"{PROJECT_DIR}/outputs/data/nutrient_diversity/nutrient_diversity_"
        + prod_cat
        + ".csv",
        index=False,
    )
