from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.utils import lookups as lps
from ahl_food_reformulation import PROJECT_DIR
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


pd.options.display.max_columns = None


# read data
pur_recs = kantar.purchase_records()
nut_recs = kantar.nutrition()
prod_mast = kantar.product_master()
val_fields = kantar.val_fields()
uom = kantar.uom()
prod_meta = kantar.product_metadata()
prod_meas = kantar.product_measurement()


# merge files to include a colume that specifies the unit of measurement of the product
pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)


pur_rec_vol.shape[0]


# add file with additional conversions (for units and servings)
tbl = lps.measure_table(prod_meas)


# merge to extract additional measures
pur_rec_vol_extra = pur_rec_vol.merge(tbl, on="Product Code", how="left")


pur_rec_vol_extra.shape[0]


# conditional expression to select volume
conds = [
    (pur_rec_vol_extra["Reported Volume"] == "Litres")
    | (pur_rec_vol_extra["Reported Volume"] == "Kilos"),
    (
        (pur_rec_vol_extra["Reported Volume"] == "Servings")
        | (pur_rec_vol_extra["Reported Volume"] == "Units")
    )
    & ~np.isnan(pur_rec_vol_extra["Grams"]),
    (
        (pur_rec_vol_extra["Reported Volume"] == "Servings")
        | (pur_rec_vol_extra["Reported Volume"] == "Units")
    )
    & np.isnan(pur_rec_vol_extra["Grams"])
    & ~np.isnan(pur_rec_vol_extra["Litres"]),
]

choices = [
    pur_rec_vol_extra["Reported Volume"],
    "Kilos",
    "Liters",
]

choice_volume = [
    pur_rec_vol_extra["Volume"],
    pur_rec_vol_extra["Quantity"] * pur_rec_vol_extra["Grams"] / 1000,
    pur_rec_vol_extra["Quantity"] * pur_rec_vol_extra["Litres"],
]

volume_per = [
    pur_rec_vol_extra["Volume"] / pur_rec_vol_extra["Quantity"],
    pur_rec_vol_extra["Grams"] / 1000,
    pur_rec_vol_extra["Litres"],
]

# Updated volume label
pur_rec_vol_extra["reported_volume_up"] = np.select(
    conds, choices, pur_rec_vol_extra["Reported Volume"]
)

# Updated volume
pur_rec_vol_extra["volume_up"] = np.select(
    conds, choice_volume, pur_rec_vol_extra["Volume"]
)

# create volume per - if it is missing then -1
pur_rec_vol_extra["volume_per"] = np.select(conds, volume_per, -1)


pur_rec_vol_extra["reported_volume_up"].value_counts()


pur_rec_vol_extra.head()


pur_rec_vol_extra.shape[0]


metadata = prod_meta[
    [
        "product_code",
        "rst_4_extended",
        "rst_4_market",
        "rst_4_market_sector",
        "rst_4_sub_market",
        "rst_4_trading_area",
    ]
].drop_duplicates()


# merge with product metadata
pur_recs_meta = pur_rec_vol_extra.merge(
    metadata, left_on="Product Code", right_on="product_code", how="left"
).drop_duplicates()


pur_recs_meta.shape[0]


# slice data to include products with reported volume of unit only (from the original variable)
pur_rec_unit = pur_recs_meta[pur_recs_meta["Reported Volume"] == "Units"]


pur_rec_not_unit = pur_recs_meta[pur_recs_meta["Reported Volume"] != "Units"]


pur_rec_not_unit["Reported Volume"].value_counts()


pur_rec_unit.size


# this will get updated as we go
pur_rec_unit_update = pur_rec_unit.copy()


pur_rec_unit_update.shape[0]


pur_rec_valid = pur_rec_unit[pur_rec_unit["volume_per"] > 0]


# generate averages by category


rst_4_extended = (
    pur_rec_valid.groupby(["rst_4_extended", "reported_volume_up"])["volume_per"]
    .mean()
    .copy()
    .reset_index(name="mean_ext")
)
rst_4_extended.head()


rst_4_sub_market = (
    pur_rec_valid.groupby(["rst_4_sub_market", "reported_volume_up"])["volume_per"]
    .mean()
    .copy()
    .reset_index(name="mean_sub")
)
rst_4_sub_market.head()


rst_4_market = (
    pur_rec_valid.groupby(["rst_4_market", "reported_volume_up"])["volume_per"]
    .mean()
    .copy()
    .reset_index(name="mean_mkt")
)
rst_4_market.head()


categories = prod_meta[
    ["rst_4_extended", "rst_4_sub_market", "rst_4_market"]
].drop_duplicates()


categories = (
    categories.merge(
        rst_4_extended[["rst_4_extended", "mean_ext"]], on="rst_4_extended", how="left"
    )
    .merge(
        rst_4_sub_market[["rst_4_sub_market", "mean_sub"]],
        on="rst_4_sub_market",
        how="left",
    )
    .merge(rst_4_market[["rst_4_market", "mean_mkt"]], on="rst_4_market", how="left")
)


categories.head()


pur_rec_unit_update = pur_rec_unit_update.merge(
    categories, on=["rst_4_extended", "rst_4_sub_market", "rst_4_market"]
)


pur_rec_unit_update.head()


condition = [
    (pur_rec_unit_update["reported_volume_up"] == "Kilos"),
    (pur_rec_unit_update["reported_volume_up"] == "Units")
    & ~np.isnan(pur_rec_unit_update["mean_ext"]),
    (pur_rec_unit_update["reported_volume_up"] == "Units")
    & (np.isnan(pur_rec_unit_update["mean_ext"]))
    & (~np.isnan(pur_rec_unit_update["mean_sub"])),
    (pur_rec_unit_update["reported_volume_up"] == "Units")
    & (np.isnan(pur_rec_unit_update["mean_ext"]))
    & (np.isnan(pur_rec_unit_update["mean_sub"]))
    & (~np.isnan(pur_rec_unit_update["mean_mkt"])),
    (pur_rec_unit_update["rst_4_sub_market"] == "Eggs Hens"),
    (pur_rec_unit_update["rst_4_sub_market"] == "Eggs Duck"),
    (pur_rec_unit_update["rst_4_sub_market"] == "Eggs Quail"),
]

volume_per = [
    pur_rec_unit_update["volume_per"],
    pur_rec_unit_update["mean_ext"],
    pur_rec_unit_update["mean_sub"],
    pur_rec_unit_update["mean_mkt"],
    50 / 1000,
    70 / 1000,
    10 / 1000,
]

type_conv = ["A", "B", "C", "D", "E", "F", "G"]


pur_rec_unit_update["volume_per_2"] = np.select(condition, volume_per, -1)
pur_rec_unit_update["volume_up_2"] = (
    pur_rec_unit_update["volume_per_2"] * pur_rec_unit_update["Quantity"]
)
pur_rec_unit_update["reported_volume_up_2"] = "Kilos"
pur_rec_unit_update["type_conv"] = np.select(condition, type_conv, "Z")


pur_rec_unit_update.shape[0]


pur_rec_unit_update["type_conv"].value_counts()


a_type = pur_rec_unit_update[pur_rec_unit_update["type_conv"] == "A"]


a_type.head()


pur_rec_unit_update.drop(columns=["volume_per", "volume_up", "reported_volume_up"])


pur_rec_unit_update.rename(
    columns={
        "volume_per_2": "volume_per",
        "volume_up_2": "volume_up",
        "reported_volume_up_2": "reported_volume_up",
    }
)


final = pd.concat([pur_rec_not_unit, pur_rec_unit_update])


final.shape[0]
