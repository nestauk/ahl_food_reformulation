from ahl_food_reformulation.pipeline.preprocessing import transform_data as transform
from ahl_food_reformulation.utils import lookups as lps
import pandas as pd
import numpy as np
from ahl_food_reformulation.getters import get_data


from ahl_food_reformulation import PROJECT_DIR


# read data
pur_recs = get_data.purchase_records()
prod_mast = get_data.product_master()
val_fields = get_data.val_fields()
uom = get_data.uom()
prod_meta = get_data.product_metadata()
prod_meas = get_data.product_measurement()


# extract gram weight from size field
cond_1 = prod_meta["size"].astype(str).str.contains("Gm")
cond_2 = prod_meta["size"].astype(str).str.contains("X")
unit_products = prod_meta.loc[(cond_1) & (~cond_2)].copy()
unit_products["Gram_weight"] = unit_products["size"].str.extract("(\d+)").astype(float)
unit_products = unit_products[["product_code", "Gram_weight"]]
unit_products.rename(columns={"product_code": "Product Code"}, inplace=True)


# merge files to include a colume that specifies the unit of measurement of the product
pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)


# add file with additional conversions (for units and servings)
tbl = lps.measure_table(prod_meas)


# merge to extract additional measures
pur_rec_vol_extra = pur_rec_vol.merge(tbl, on="Product Code", how="left")

# merge to include the gram weight from the size description
pur_rec_vol_extra = pur_rec_vol_extra.merge(
    unit_products, on="Product Code", how="left"
)

# create a combined gram measure
cond = [
    (~np.isnan(pur_rec_vol_extra["Grams"])),
    (
        np.isnan(pur_rec_vol_extra["Grams"])
        & ~np.isnan(pur_rec_vol_extra["Gram_weight"])
    ),
]

value = [pur_rec_vol_extra["Grams"], pur_rec_vol_extra["Gram_weight"]]

pur_rec_vol_extra["gram_comb"] = np.select(cond, value, np.nan)


# conditional expression to select volume
conds = [
    (pur_rec_vol_extra["Reported Volume"] == "Litres")
    | (pur_rec_vol_extra["Reported Volume"] == "Kilos"),
    (
        (pur_rec_vol_extra["Reported Volume"] == "Servings")
        | (pur_rec_vol_extra["Reported Volume"] == "Units")
    )
    & ~np.isnan(pur_rec_vol_extra["gram_comb"]),
    (
        (pur_rec_vol_extra["Reported Volume"] == "Servings")
        | (pur_rec_vol_extra["Reported Volume"] == "Units")
    )
    & np.isnan(pur_rec_vol_extra["gram_comb"])
    & ~np.isnan(pur_rec_vol_extra["Litres"]),
]

choices = [
    pur_rec_vol_extra["Reported Volume"],
    "Kilos",
    "Liters",
]

choice_volume = [
    pur_rec_vol_extra["Volume"],
    pur_rec_vol_extra["Quantity"] * pur_rec_vol_extra["gram_comb"] / 1000,
    pur_rec_vol_extra["Quantity"] * pur_rec_vol_extra["Litres"],
]

volume_per = [
    pur_rec_vol_extra["Volume"] / pur_rec_vol_extra["Quantity"],
    pur_rec_vol_extra["gram_comb"] / 1000,
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


metadata = prod_meta[
    [
        "product_code",
        "rst_4_extended",
        "rst_4_market",
        "rst_4_market_sector",
        "rst_4_sub_market",
        "rst_4_trading_area",
    ]
].copy()


metadata.rename(columns={"product_code": "Product Code"}, inplace=True)


# merge with product metadata
pur_recs_meta = pur_rec_vol_extra.merge(metadata, on="Product Code", how="left")


# slice data to include products with reported volume of unit only (from the original variable)
pur_rec_unit = pur_recs_meta[pur_recs_meta["Reported Volume"] == "Units"]


pur_rec_not_unit = pur_recs_meta[pur_recs_meta["Reported Volume"] != "Units"]


# this will get updated as we go
pur_rec_unit_update = pur_rec_unit.copy()

# only select products with valid measurement for imputation
pur_rec_valid = pur_rec_unit[pur_rec_unit["volume_per"] > 0]


# generate averages by category


rst_4_extended = (
    pur_rec_valid.groupby(["rst_4_extended", "reported_volume_up"])["volume_per"]
    .mean()
    .copy()
    .reset_index(name="mean_ext")
)


rst_4_sub_market = (
    pur_rec_valid.groupby(["rst_4_sub_market", "reported_volume_up"])["volume_per"]
    .mean()
    .copy()
    .reset_index(name="mean_sub")
)


rst_4_market = (
    pur_rec_valid.groupby(["rst_4_market", "reported_volume_up"])["volume_per"]
    .mean()
    .copy()
    .reset_index(name="mean_mkt")
)


categories = prod_meta[
    ["rst_4_extended", "rst_4_sub_market", "rst_4_market"]
].drop_duplicates()

# merge all files with aggregate weight by category
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


pur_rec_unit_update = pur_rec_unit_update.merge(
    categories, on=["rst_4_extended", "rst_4_sub_market", "rst_4_market"]
)

# impute weight based on category


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
pur_rec_unit_update["type_conv"] = np.select(condition, type_conv, "Z")


# updated volume depending on imputation method
pur_rec_unit_update["volume_up_2"] = np.where(
    (pur_rec_unit_update["type_conv"] == "E")
    | (pur_rec_unit_update["type_conv"] == "F")
    | (pur_rec_unit_update["type_conv"] == "G"),
    (pur_rec_unit_update["volume_per_2"] * pur_rec_unit_update["Volume"]),
    (pur_rec_unit_update["volume_per_2"] * pur_rec_unit_update["Quantity"]),
)

pur_rec_unit_update["reported_volume_up_2"] = "Kilos"


# drops old volume colums and subsitute with new ones
pur_rec_unit_update.drop(
    columns=["volume_per", "volume_up", "reported_volume_up"], inplace=True
)
pur_rec_unit_update.rename(
    columns={
        "volume_per_2": "volume_per",
        "volume_up_2": "volume_up",
        "reported_volume_up_2": "reported_volume_up",
    },
    inplace=True,
)

# concatenate files to obtain full purchsae record
pur_rec_unit_update = pur_rec_unit_update[pur_rec_not_unit.columns]

final = pd.concat([pur_rec_not_unit, pur_rec_unit_update])

out = final[
    [
        "Panel Id",
        "Product Code",
        "Store Code",
        "Quantity",
        "Spend",
        "Volume",
        "Promo Code",
        "PurchaseId",
        "Period",
        "Purchase Date",
        "Worldpanel Year",
        "Worldpanel Week",
        "Worldpanel Day",
        "Gross Up Weight",
        "Reported Volume",
        "reported_volume_up",
        "volume_up",
        "volume_per",
    ]
]

out.to_csv(f"{PROJECT_DIR}/inputs/data/pur_rec_volume.csv")
