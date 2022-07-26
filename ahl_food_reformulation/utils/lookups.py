# Import modules
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
