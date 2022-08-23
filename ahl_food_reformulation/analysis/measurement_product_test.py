# -*- coding: utf-8 -*-
# %%
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.utils import lookups as lps
from ahl_food_reformulation import PROJECT_DIR
import logging
import pandas as pd

# %%
logging.info("loading data")
# Reading in data
pur_recs = kantar.purchase_records()
nut_recs = kantar.nutrition()
prod_mast = kantar.product_master()
val_fields = kantar.val_fields()
uom = kantar.uom()

# %%
conv = pd.read_csv(
    f"{PROJECT_DIR}/inputs/data/Nesta - Units, Grams, Millilitres, Servings All Products.txt",
    encoding="ISO-8859-1",
)

# %%
# Adding volume measurement
pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)

# %% [markdown]
# Convert from Volume Type 1 to Volume Type 2
#
# - (‘Volume Type 1 from purchase file’ / ‘Volume Type 1 from reference file’) * ‘Volume Type 2 from reference file’

# %%
pur_rec_vol["Reported Volume"].value_counts()

# %%
conv.set_index("PRODUCT", inplace=True)
conv_meas = (
    conv.groupby([conv.index, "VOLUME TYPE"])["VALUE"].first().unstack().reset_index()
)

# %%
conv_meas.columns = ["Product Code", "Grams", "Millilitres", "Servings", "Units"]
conv_meas["Litres"] = conv_meas["Millilitres"] / 1000

# %%
conv_meas.head(5)


# %%
def conv_grams(pur_rec_vol, conv_meas, measure):
    pur_rec_meas = (
        pur_rec_vol[pur_rec_vol["Reported Volume"] == measure]
        .copy()
        .merge(
            conv_meas[["Product Code", measure, "Grams"]], how="left", on="Product Code"
        )
    )
    pur_rec_meas["Updated_volume"] = (
        (pur_rec_meas["Volume"] / pur_rec_meas[measure]) * pur_rec_meas["Grams"]
    ) / 1000
    pur_rec_meas["Updated_volume"] = np.where(
        (pur_rec_meas["Grams"].isnull() | pur_rec_meas[measure].isnull()),
        float("NAN"),
        pur_rec_meas["Updated_volume"],
    )
    return pur_rec_meas


# %%
pur_rec_units = conv_grams(pur_rec_vol, conv_meas, "Units")
pur_rec_litres = conv_grams(pur_rec_vol, conv_meas, "Litres")
pur_rec_servings = conv_grams(pur_rec_vol, conv_meas, "Servings")

# %%
pur_rec_kilos = pur_rec_vol[pur_rec_vol["Reported Volume"] == "Kilos"].copy()
pur_rec_kilos["Updated_volume"] = pur_rec_kilos["Volume"]
pur_recs_full = pd.concat(
    [pur_rec_kilos, pur_rec_units, pur_rec_litres, pur_rec_servings], ignore_index=True
)

# %%
pur_rec_kilos.shape

# %%
# Some values must of not have a reported volume...
print(pur_recs_full.shape, pur_rec_vol.shape)

# %%
print("Total purchase records volume not converted to kilos")
(pur_recs_full["Updated_volume"].isna().sum() / 36387653) * 100

# %%
print("Total purchase records volume previously not kilos")
((36387653 - 27330035) / 36387653) * 100

# %% [markdown]
# ### Unique product table

# %%
logging.info("Slice by Kilos")
# Slicing by only kilos
pur_rec_kilos = pur_rec_vol[pur_rec_vol["Reported Volume"] == "Kilos"].copy()

# %%
logging.info("Getting unique product list")
# Applying function to get unique product list with selected per 100g nutritional info
unique_prods_nut = lps.products_per_100g(
    ["Energy KJ", "Protein KG"], pur_rec_kilos, nut_recs
)

# %%
# Checking a few examples
unique_prods_nut.head(5)

# %%
# Check for duplicates
unique_prods_nut["Product Code"].is_unique
