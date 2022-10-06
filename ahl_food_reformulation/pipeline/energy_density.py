from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.utils import lookups as lps
import pandas as pd
import numpy as np


def prod_energy_100(
    cat: str,
    val_fields,
    prod_mast,
    uom,
    pur_recs,
    nut_recs,
    prod_meta,
    prod_meas,
):
    """
    Return simple and weighted kcal/100ml(g) by product (with category information)
    Args:
        cat (str): one product category
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        prod_mast (pd.DataFrame): Pandas dataframe unique product list
        uom (pd.DataFrame): Pandas dataframe contains product measurement information
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
        prod_meas (pd.DataFrame): Pandas dataframe with additional conversions to g and ml for unit and serving products
    Returns:
        pd.DataFrame: Dataframe with average kcal/100ml(gr) simple and weighted by sales (for the year) and reported volume
    """

    # add standardised volume measurement
    pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)

    # add file with additional conversions (for units and servings)
    tbl = lps.measure_table(prod_meas)

    # merge to extract additional measures
    pur_rec_vol = pur_rec_vol.merge(tbl, on="Product Code")

    # conditional expression to select volume
    conds = [
        (pur_rec_vol["Reported Volume"] == "Litres")
        | (pur_rec_vol["Reported Volume"] == "Kilos"),
        (
            (pur_rec_vol["Reported Volume"] == "Servings")
            | (pur_rec_vol["Reported Volume"] == "Units")
        )
        & ~np.isnan(pur_rec_vol["Grams"]),
        (
            (pur_rec_vol["Reported Volume"] == "Servings")
            | (pur_rec_vol["Reported Volume"] == "Units")
        )
        & np.isnan(pur_rec_vol["Grams"])
        & ~np.isnan(pur_rec_vol["Litres"]),
    ]

    choices = [
        pur_rec_vol["Reported Volume"],
        "Kilos",
        "Liters",
    ]

    choice_volume = [
        pur_rec_vol["Volume"],
        pur_rec_vol["Quantity"] * pur_rec_vol["Grams"] / 1000,
        pur_rec_vol["Quantity"] * pur_rec_vol["Litres"],
    ]

    # Updated volume label
    pur_rec_vol["reported_volume_up"] = np.select(conds, choices, "missing")

    # Updated volume
    pur_rec_vol["volume_up"] = np.select(conds, choice_volume, pur_rec_vol["Volume"])

    # scaled gross up weight - this converts the wegith from quantities to volumes (either kg or l)

    pur_rec_vol["scaled_factor"] = (
        pur_rec_vol["Gross Up Weight"] * pur_rec_vol["volume_up"]
    )

    # create unique list of products with total sales
    pur_recs_agg = (
        pur_rec_vol.groupby(["Product Code", "reported_volume_up"])["scaled_factor"]
        .sum()
        .reset_index(name="total_sale")
    )

    # merge with product metadata
    pur_recs_meta = pur_recs_agg.merge(
        prod_meta, left_on="Product Code", right_on="product_code", how="left"
    )

    # check distribution of reported volume within category
    level = (
        pur_recs_meta.groupby([cat, "reported_volume_up"])
        .size()
        .copy()
        .reset_index(name="count")
    )
    level_pivot = (
        pd.pivot(level, index=[cat], columns="reported_volume_up", values="count")
        .fillna(0)
        .reset_index()
    )

    # determine which measurement is used to generate average
    level_pivot["tot"] = (
        level_pivot["Kilos"] + level_pivot["Litres"] + level_pivot["missing"]
    )
    level_pivot["kilo_share"] = level_pivot["Kilos"] / level_pivot["tot"]
    level_pivot["litre_share"] = level_pivot["Litres"] / level_pivot["tot"]
    level_pivot["chosen_unit"] = np.where(
        level_pivot["litre_share"] >= 0.9,
        "Litres",
        np.where(level_pivot["kilo_share"] >= 0.9, "Kilos", "none"),
    )

    # merge with product metadata
    pur_rec_conv = pur_recs_agg.merge(
        prod_meta, left_on="Product Code", right_on="product_code", how="left"
    ).merge(level_pivot[[cat, "chosen_unit"]], on=cat)

    # subset to products where the reported volume is equal to the chosen unit based on 90% rule
    pur_rec_select = pur_rec_conv[
        pur_rec_conv["reported_volume_up"] == pur_rec_conv["chosen_unit"]
    ]

    # generate nutritional info to merge into the aggregate data
    # Convert to datetime format
    pur_rec_vol["Purchase Date"] = pd.to_datetime(
        pur_rec_vol["Purchase Date"], format="%d/%m/%Y"
    )

    # Get unique and most recent products
    pur_recs_latest = (
        pur_rec_vol.sort_values(by=["Purchase Date"], ascending=False)
        .drop_duplicates(subset="Product Code", keep="first")
        .merge(
            nut_recs[["Purchase Number", "Purchase Period", "Energy KCal"]],
            how="left",
            left_on=["PurchaseId", "Period"],
            right_on=["Purchase Number", "Purchase Period"],
        )
        .drop(["Purchase Number", "Purchase Period"], axis=1)
    )
    # generate value of kcal per 100ml(g)
    pur_recs_latest["kcal_100g_ml"] = pur_recs_latest["Energy KCal"] / (
        pur_recs_latest["volume_up"] * 10
    )

    # anything with more than 900kcal per 100ml(g) is implausible because of the energy density of fat being 9kcal/g
    pur_recs_latest = pur_recs_latest[pur_recs_latest["kcal_100g_ml"] <= 900].copy()

    # unique dataframe of product with kcal into
    density_prod = pur_recs_latest[["Product Code", "kcal_100g_ml"]].drop_duplicates(
        subset="Product Code"
    )

    # merge kcal info with sales
    return pur_rec_select.merge(density_prod, on="Product Code")


def cat_energy_100(
    cat: str,
    val_fields: pd.DataFrame,
    prod_mast: pd.DataFrame,
    uom: pd.DataFrame,
    pur_recs: pd.DataFrame,
    nut_recs: pd.DataFrame,
    prod_meta: pd.DataFrame,
    prod_meas: pd.DataFrame,
):
    """
    Return simple and weighted kcal/100ml(g) aggregate by product category

    Args:
        cat (str): one product category
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        prod_mast (pd.DataFrame): Pandas dataframe unique product list
        uom (pd.DataFrame): Pandas dataframe contains product measurement information
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
        prod_meas (pd.DataFrame): Pandas dataframe with additional conversions to g and ml for unit and serving products

    Returns:
        pd.DataFrame: Dataframe with average kcal/100ml(gr) simple and weighted by sales (for the year) and reported volume
    """

    # add standardised volume measurement
    pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)

    # add file with additional conversions (for units and servings)
    tbl = lps.measure_table(prod_meas)

    # merge to extract additional measures
    pur_rec_vol = pur_rec_vol.merge(tbl, on="Product Code")

    # conditional expression to select volume
    conds = [
        (pur_rec_vol["Reported Volume"] == "Litres")
        | (pur_rec_vol["Reported Volume"] == "Kilos"),
        (
            (pur_rec_vol["Reported Volume"] == "Servings")
            | (pur_rec_vol["Reported Volume"] == "Units")
        )
        & ~np.isnan(pur_rec_vol["Grams"]),
        (
            (pur_rec_vol["Reported Volume"] == "Servings")
            | (pur_rec_vol["Reported Volume"] == "Units")
        )
        & np.isnan(pur_rec_vol["Grams"])
        & ~np.isnan(pur_rec_vol["Litres"]),
    ]

    choices = [
        pur_rec_vol["Reported Volume"],
        "Kilos",
        "Liters",
    ]

    choice_volume = [
        pur_rec_vol["Volume"],
        pur_rec_vol["Quantity"] * pur_rec_vol["Grams"] / 1000,
        pur_rec_vol["Quantity"] * pur_rec_vol["Litres"],
    ]

    # Updated volume label
    pur_rec_vol["reported_volume_up"] = np.select(conds, choices, "missing")

    # Updated volume
    pur_rec_vol["volume_up"] = np.select(conds, choice_volume, pur_rec_vol["Volume"])

    # scaled gross up weight - this converts the wegith from quantities to volumes (either kg or l)

    pur_rec_vol["scaled_factor"] = (
        pur_rec_vol["Gross Up Weight"] * pur_rec_vol["volume_up"]
    )

    # create unique list of products with total sales
    pur_recs_agg = (
        pur_rec_vol.groupby(["Product Code", "reported_volume_up"])["scaled_factor"]
        .sum()
        .reset_index(name="total_sale")
    )

    # merge with product metadata
    pur_recs_meta = pur_recs_agg.merge(
        prod_meta, left_on="Product Code", right_on="product_code", how="left"
    )

    # check distribution of reported volume within category
    level = (
        pur_recs_meta.groupby([cat, "reported_volume_up"])
        .size()
        .copy()
        .reset_index(name="count")
    )
    level_pivot = (
        pd.pivot(level, index=[cat], columns="reported_volume_up", values="count")
        .fillna(0)
        .reset_index()
    )

    # determine which measurement is used to generate average
    level_pivot["tot"] = (
        level_pivot["Kilos"] + level_pivot["Litres"] + level_pivot["missing"]
    )
    level_pivot["kilo_share"] = level_pivot["Kilos"] / level_pivot["tot"]
    level_pivot["litre_share"] = level_pivot["Litres"] / level_pivot["tot"]
    level_pivot["chosen_unit"] = np.where(
        level_pivot["litre_share"] >= 0.9,
        "Litres",
        np.where(level_pivot["kilo_share"] >= 0.9, "Kilos", "none"),
    )

    # merge with product metadata
    pur_rec_conv = pur_recs_agg.merge(
        prod_meta, left_on="Product Code", right_on="product_code", how="left"
    ).merge(level_pivot[[cat, "chosen_unit"]], on=cat)

    # subset to products where the reported volume is equal to the chosen unit based on 90% rule
    pur_rec_select = pur_rec_conv[
        pur_rec_conv["reported_volume_up"] == pur_rec_conv["chosen_unit"]
    ]

    # generate nutritional info to merge into the aggregate data
    # Convert to datetime format
    pur_rec_vol["Purchase Date"] = pd.to_datetime(
        pur_rec_vol["Purchase Date"], format="%d/%m/%Y"
    )

    # Get unique and most recent products
    pur_recs_latest = (
        pur_rec_vol.sort_values(by=["Purchase Date"], ascending=False)
        .drop_duplicates(subset="Product Code", keep="first")
        .merge(
            nut_recs[["Purchase Number", "Purchase Period", "Energy KCal"]],
            how="left",
            left_on=["PurchaseId", "Period"],
            right_on=["Purchase Number", "Purchase Period"],
        )
        .drop(["Purchase Number", "Purchase Period"], axis=1)
    )
    # generate value of kcal per 100ml(g)
    pur_recs_latest["kcal_100g_ml"] = pur_recs_latest["Energy KCal"] / (
        pur_recs_latest["volume_up"] * 10
    )

    # anything with more than 900kcal per 100ml(g) is implausible because of the energy density of fat being 9kcal/g
    pur_recs_latest = pur_recs_latest[pur_recs_latest["kcal_100g_ml"] <= 900].copy()

    # unique dataframe of product with kcal into
    density_prod = pur_recs_latest[["Product Code", "kcal_100g_ml"]].drop_duplicates(
        subset="Product Code"
    )

    # merge kcal info with sales
    pur_final = pur_rec_select.merge(density_prod, on="Product Code")

    # simple mean
    s_mean = (
        pur_final.groupby([cat, "chosen_unit"])["kcal_100g_ml"]
        .mean()
        .reset_index(name="kcal_100_s")
    )

    # weighted mean
    pur_final["cross_prd"] = (
        pur_final["kcal_100g_ml"] * pur_final["total_sale"]
    )  # cross product
    kcal = (
        pur_final.groupby([cat])["cross_prd"].sum().reset_index(name="sum_cross_prd")
    )  # sum of cross product by cat
    sale = (
        pur_final.groupby([cat])["total_sale"].sum().reset_index(name="sum_total_sale")
    )  # sum of sales by cat
    w_mean = kcal.merge(sale, on=cat)  # merge sum of cross and total
    w_mean["kcal_100_w"] = (
        w_mean["sum_cross_prd"] / w_mean["sum_total_sale"]
    )  # weighted mean

    # generate final output
    return s_mean.merge(w_mean[[cat, "kcal_100_w"]], on=cat)
