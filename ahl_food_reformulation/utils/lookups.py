# Import modules
from typing import List
import pandas as pd
import numpy as np
from ahl_food_reformulation import PROJECT_DIR


def map_codes(df: pd.DataFrame, col: str):
    """
    Maps code numbers to values

    Args:
        df (pd.DataFrame): Pandas dataframe
        col (str): Name of column to map

    Returns:
        pd.DateFrame:
    """
    di = {
        2: "Urban-Rural",
        3: "Social Class",
        4: "Council Tax Band",
        5: "Region",
        7: "Newspaper Read",
        8: "Life Stage",
        9: "Household Income",
        11: "Ethnicity",
        12: "Education Level",
    }
    return df[col].map(di)


def hh_demographic_table(
    demog_coding: pd.DataFrame, demog_val: pd.DataFrame, pan_mast: pd.DataFrame
):
    """
    Creates a dataframe with combined demographic information about each household

    Args:
        demog_coding (pd.DataFrame): Pandas dataframe codes per household that links to demographic information
        demog_val (pd.DataFrame): Pandas dataframe demographic values per code
        pan_mast (pd.DataFrame): Pandas dataframe of household information

    Returns:
        pd.DateFrame: Dataframe with demographic information for each household
    """
    demog_val.columns = ["Demog Id", "Demog Value", "Demog Description"]
    hh_demogs = demog_coding.merge(
        demog_val,
        left_on=["Demog Id", "Demog Value"],
        right_on=["Demog Id", "Demog Value"],
        how="left",
    )
    hh_demogs.drop(["Demog Value"], axis=1, inplace=True)
    hh_demogs["Demog Id"] = map_codes(hh_demogs, "Demog Id")
    hh_demogs.set_index("Panel Id", inplace=True)
    hh_demogs = hh_demogs.pivot_table(
        values="Demog Description",
        index=hh_demogs.index,
        columns="Demog Id",
        aggfunc="first",
    )
    return pd.merge(
        hh_demogs, pan_mast.set_index("Panel Id"), left_index=True, right_index=True
    )


def product_table(
    val_fields: pd.DataFrame,
    prod_mast: pd.DataFrame,
    uom: pd.DataFrame,
    prod_codes: pd.DataFrame,
    prod_vals: pd.DataFrame,
    prod_att: pd.DataFrame,
):
    """
    Creates a dataframe with information for each product

    Args:
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        prod_mast (pd.DataFrame): Panadas dataframe unique product list
        uom (pd.DataFrame): Panadas dataframe contains product measurement information
        prod_codes (pd.DataFrame): Panadas dataframe contains the codes to link products to category information
        prod_vals (pd.DataFrame): Panadas dataframe contains the product category information
        prod_att (pd.DataFrame): Panadas dataframe containing description for each attribute

    Returns:
        pd.DateFrame: Dataframe with information for each product
    """
    # Get volume
    val_fields.drop_duplicates(inplace=True)  # Remove duplicates
    prod_vol = (
        prod_mast[["Product Code", "Validation Field"]]
        .merge(
            val_fields[["VF", "UOM"]],
            left_on="Validation Field",
            right_on="VF",
            how="left",
        )
        .merge(uom[["UOM", "Reported Volume"]], on="UOM", how="left")
        .drop(["Validation Field", "VF", "UOM"], axis=1)
    )
    # Get product info (including categories)
    att_dict = dict()
    for ix, row in prod_att.iterrows():
        att_dict[row["Attribute Number"]] = row["Attribute Description"]
    prod_codes["Attribute"] = prod_codes["Attribute Number"].apply(
        lambda x: att_dict[x]
    )
    combined_prod_att = prod_codes.merge(
        prod_vals, left_on="Attribute Value", right_on="Attribute Value", how="left"
    )
    combined_prod_att.set_index("Product Code", inplace=True)
    combined_prod_att = combined_prod_att.pivot_table(
        values="Attribute Value Description",
        index=combined_prod_att.index,
        columns="Attribute",
        aggfunc="first",
    )
    return pd.merge(
        combined_prod_att,
        prod_vol.set_index("Product Code"),
        left_index=True,
        right_index=True,
    ).reset_index()


def products_per_100g(nut_list: list, pur_recs: pd.DataFrame, nut_recs: pd.DataFrame):
    """
    Creates a dataframe of unique products and selected per 100g nutritional information

    Args:
        nut_list (list): List of nutritional columns to convert
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information

    Returns:
        pd.DateFrame: Dataframe with per 100g nutritional info for each product
    """
    # Convert to datetime format
    pur_recs["Purchase Date"] = pd.to_datetime(
        pur_recs["Purchase Date"], format="%d/%m/%Y"
    )
    # Get unique and most recent products
    pur_recs_latest = (
        pur_recs.sort_values(by=["Purchase Date"], ascending=False)
        .drop_duplicates(subset="Product Code", keep="first")
        .merge(
            nut_recs[["Purchase Number", "Purchase Period"] + nut_list],
            how="left",
            left_on=["PurchaseId", "Period"],
            right_on=["Purchase Number", "Purchase Period"],
        )
        .drop(["Purchase Number", "Purchase Period"], axis=1)
    )
    # Add per 100g info for selected nutrients
    for nut in nut_list:
        pur_recs_latest[nut + "_100g"] = pur_recs_latest[nut] / (
            pur_recs_latest["Volume"] * 10
        )
    return pur_recs_latest[["Product Code"] + [sub + "_100g" for sub in nut_list]]


def measure_table(conv: pd.DataFrame):
    """
    Creates a table of products and measurements

    Args:
        conv (pd.DataFrame): Pandas dataframe with product measurements

    Returns:
        pd.DateFrame: Dataframe with table of products and measurements
    """
    conv_update = conv.copy()
    conv_update.set_index("PRODUCT", inplace=True)
    conv_meas = (
        conv_update.groupby([conv_update.index, "VOLUME TYPE"])["VALUE"]
        .first()
        .unstack()
        .reset_index()
    )
    conv_meas.columns = ["Product Code", "Grams", "Millilitres", "Servings", "Units"]
    conv_meas["Litres"] = conv_meas["Millilitres"] / 1000
    return conv_meas


def conv_kilos(pur_rec_vol: pd.DataFrame, conv_meas: pd.DataFrame, measures: list):
    """
    Converts selected measurements into kilos

    Args:
        pur_rec_vol (pd.DataFrame): Pandas dataframe of purchase records (with volume measurement field)
        conv_meas (pd.DataFrame): Pandas dataframe of product measurements
        measures (list): List of measurements to convert

    Returns:
        pd.DateFrame: Dataframe with kilo products for selected measurements (and existing kilos)
    """
    meas_dfs = []
    for measure in measures:
        pur_rec_meas = (
            pur_rec_vol[pur_rec_vol["Reported Volume"] == measure]
            .copy()
            .merge(
                conv_meas[["Product Code", measure, "Grams"]],
                how="left",
                on="Product Code",
            )
        )
        pur_rec_meas["Volume"] = (
            (pur_rec_meas["Volume"] / pur_rec_meas[measure]) * pur_rec_meas["Grams"]
        ) / 1000
        pur_rec_meas.dropna(subset=["Grams", measure], inplace=True)
        pur_rec_meas["Reported Volume"] = "Kilos"
        meas_dfs.append(pur_rec_meas)

    pur_rec_kilos = pur_rec_vol[pur_rec_vol["Reported Volume"] == "Kilos"].copy()
    meas_dfs.append(pur_rec_kilos)
    kilo_df = pd.concat(meas_dfs, ignore_index=True)
    return kilo_df.drop(["Grams", "Litres", "Servings", "Units"], axis=1)
