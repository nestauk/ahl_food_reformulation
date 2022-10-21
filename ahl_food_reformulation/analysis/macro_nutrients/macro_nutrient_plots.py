# Import libraries
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import nutrient_metrics_funcs as nutrient
from ahl_food_reformulation.utils.plotting import configure_plots
import logging
import altair as alt
import json
import pandas as pd
from pathlib import Path
from ahl_food_reformulation import PROJECT_DIR

logging.info("Reading data")
# Read data
nut_recs = kantar.nutrition()
pur_recs = kantar.purchase_records_updated()
prod_meta = kantar.prod_meta_update()
# Get chosen cats
with open(f"{PROJECT_DIR}/outputs/data/decision_table/detailed_products.txt") as f:
    data = json.load(f)
chosen_cats = pd.DataFrame(data)

# define cats
broader_category = "rst_4_market_sector"
granular_category = "rst_4_extended"

chosen_cats = chosen_cats.melt(var_name=broader_category, value_name=granular_category)

# Create tables
broad_macro_nut = nutrient.macro_nutrient_table(
    pur_recs, prod_meta, nut_recs, broader_category
)
gran_macro_nut = nutrient.macro_nutrient_table(
    pur_recs, prod_meta, nut_recs, granular_category
)

broad_cats = list(chosen_cats.rst_4_market_sector.drop_duplicates())

broad_macro_nut_subset = broad_macro_nut[
    broad_macro_nut.rst_4_market_sector.isin(broad_cats)
][["rst_4_market_sector", "Carb_prop", "Prot_prop", "Fat_prop"]].copy()
broad_macro_nut_subset.columns = ["Categories", "Carbohydrates", "Protein", "Fat"]

# Plot top candidates
source = broad_macro_nut_subset.melt(
    id_vars=["Categories"], var_name="Macro nutrients", value_name="proportions"
)
fig = (
    alt.Chart(source)
    .mark_bar()
    .encode(
        y=alt.Y("Categories", title="Categories", axis=alt.Axis(titlePadding=20)),
        x="sum(proportions)",
        color="Macro nutrients",
    )
    .properties(width=250, height=250)
)
configure_plots(
    fig,
    "Macronutrient proportions for top candidates",
    "",
    16,
    20,
    12,
)

# Granular plots
gran_macro_nut_subset = chosen_cats.merge(
    gran_macro_nut[["rst_4_extended", "Carb_prop", "Prot_prop", "Fat_prop"]],
    on="rst_4_extended",
    how="left",
)
gran_facet = gran_macro_nut_subset.copy()
gran_facet.columns = ["Market sector", "Categories", "Carbohydrates", "Protein", "Fat"]

source = gran_facet.melt(
    id_vars=["Market sector", "Categories"],
    var_name="Macro nutrients",
    value_name="proportions",
)
fig = (
    alt.Chart(source)
    .mark_bar()
    .encode(
        y=alt.Y("Categories", title="Categories", axis=alt.Axis(titlePadding=20)),
        x="sum(proportions)",
        color="Macro nutrients",
        facet=alt.Facet("Market sector:N", columns=2),
    )
    .properties(width=200, height=150)
)
configure_plots(
    fig,
    "Macronutrient proportions for " + broad_cats[0],
    "",
    16,
    20,
    12,
)
fig.resolve_scale(y="independent")
