# Import libraries
from pyclbr import Function
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import os.path
from ahl_food_reformulation import PROJECT_DIR


def combine_files(
    val_fields: pd.DataFrame,
    pur_recs: pd.DataFrame,
    prod_mast: pd.DataFrame,
    uom: pd.DataFrame,
    prod_codes: pd.DataFrame,
    prod_vals: pd.DataFrame,
    att_num: int,
):
    """
    Performs multiple merges and a few cleaning functions to combine the following files into one:
    val_fields, pur_records, prod_mast, uom, prod_codes, prod_vals

    Args:
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        prod_mast (pd.DataFrame): Pandas dataframe unique product list
        uom (pd.DataFrame): Panadas dataframe contains product measurement information
        prod_codes (pd.DataFrame): Pandas dataframe contains the codes to link products to category information
        prod_vals (pd.DataFrame): Pandas dataframe contains the product category information
        att_num (int): Product category type code number

    Returns:
        pur_recs (pandas.DateFrame): Merged pandas dataframe
    """
    val_fields.drop_duplicates(inplace=True)  # Remove duplicates
    pur_recs = pur_recs[
        ["PurchaseId", "Panel Id", "Period", "Product Code", "Volume", "Quantity"]
    ].merge(
        prod_mast[["Product Code", "Validation Field"]], on="Product Code", how="left"
    )
    pur_recs = pur_recs.merge(
        val_fields[["VF", "UOM"]], left_on="Validation Field", right_on="VF", how="left"
    )
    pur_recs = pur_recs.merge(uom[["UOM", "Reported Volume"]], on="UOM", how="left")
    rst_4_ext = prod_codes[prod_codes["Attribute Number"] == att_num].copy()
    prod_code_vals = rst_4_ext.merge(prod_vals, on="Attribute Value", how="left")
    pur_recs = pur_recs.merge(
        prod_code_vals[["Product Code", "Attribute Value Description"]],
        on="Product Code",
        how="left",
    )
    pur_recs = pur_recs[
        pur_recs["Reported Volume"].notna()
    ]  # Remove purchases with no volume
    pur_recs["att_vol"] = (
        pur_recs["Attribute Value Description"] + "_" + pur_recs["Reported Volume"]
    )
    return pur_recs


def nutrition_merge(nutrition: pd.DataFrame, purch_recs: pd.DataFrame, cols: list):
    """Merges the purchase records and nutrition file

    Args:
        nutrition (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        cols (list): List of columns names to merge from the nutrition dataset

    Returns:
        (pandas.DateFrame): Merged pandas dataframe
    """
    # Add unique purchase ID
    nutrition["pur_id"] = (
        nutrition["Purchase Number"].astype(str)
        + "_"
        + nutrition["Purchase Period"].astype(str)
    )
    purch_recs["pur_id"] = (
        purch_recs["PurchaseId"].astype(str) + "_" + purch_recs["Period"].astype(str)
    )
    # Merge datasets
    return purch_recs.merge(nutrition[["pur_id"] + cols], on="pur_id", how="left")


def total_product_hh_purchase(purch_recs: pd.DataFrame):
    """Groups by household, measurement and product and sums the volume and kcal content.

    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data

    Returns:
        (pandas.DateFrame): groupby pandas dataframe
    """
    # Remove cases where volume is zero (8 cases)
    purch_recs = purch_recs[purch_recs["Volume"] != 0].copy()
    return (
        purch_recs.groupby(["Panel Id", "Reported Volume", "att_vol"])[
            ["Volume", "Energy KCal", "Quantity"]
        ]
        .sum()
        .reset_index()
    )


def norm_variable(data: pd.Series):
    """normalise variable between 0 and 1.

    Args:
        data (pd.Series): Pandas series, volume for different measurements.

    Returns:
        pd.Series: Normalised column
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def hh_total_categories(df: pd.DataFrame):
    """
    Groups by the household id and category summing the volume.

    Args:
        df (pd.DataFrame): Pandas dataframe of purchase records (noramlised by volume measurement)

    Returns:
        pd.DateFrame: Household totals per food category
    """
    return (
        df.groupby(["Panel Id", "att_vol"])["Volume"]
        .sum()
        .unstack(["att_vol"])
        .fillna(0)
    )


def make_purch_records(
    purchases: pd.DataFrame,
    nutrition: pd.DataFrame,
    val_fields: pd.DataFrame,
    prod_mast: pd.DataFrame,
    uom: pd.DataFrame,
    prod_codes: pd.DataFrame,
    prod_vals: pd.DataFrame,
):
    """
    Merges dataframes to create purchase records df with food category and nutrition information

    Args:
        purchases (pd.DataFrame): Pandas dataframe of purchase records (noramlised by volume measurement)
        nutrition (pd.DataFrame): Pandas dataframe of purchase level nutritional information
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        prod_mast (pd.DataFrame): Pandas dataframe unique product list
        uom (pd.DataFrame): Panadas dataframe contains product measurement information
        prod_codes (pd.DataFrame): Pandas dataframe contains the codes to link products to category information
        prod_vals (pd.DataFrame): Pandas dataframe contains the product category information

    Returns:
        pd.DateFrame: Household totals per food category
    """
    purchases_comb = combine_files(
        val_fields, purchases, prod_mast, uom, prod_codes, prod_vals, 2907
    )
    purchases_nutrition = nutrition_merge(nutrition, purchases_comb, ["Energy KCal"])
    return total_product_hh_purchase(purchases_nutrition)


def kcal_contribution(purch_recs: pd.DataFrame):
    """
    Calculates the kcal contribution to the volume and normalises by measurement.

    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data

    Returns:
        pd.DateFrame: Kcal / volume ratio per household, scaled by measurement
    """
    return (
        purch_recs.pipe(
            lambda df: df.assign(
                kcal_vol=(df["Energy KCal"] / df["Volume"]) * df["Quantity"]
            )
        )
        .pipe(
            lambda df: df.assign(
                vol_scaled=df.groupby("Reported Volume")["kcal_vol"].apply(
                    norm_variable
                )
            )
            .set_index(["Panel Id", "att_vol"])[["vol_scaled"]]
            .unstack(["att_vol"])
            .fillna(0)
        )
        .droplevel(0, axis=1)
    )


def hh_kcal_per_category(purch_recs: pd.DataFrame):
    """
    Unstacks df to show total kcal per product per household then normalises by household (rows)

    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data

    Returns:
        (pd.DateFrame): Kcal totals per product per household normalised by total household kcal
    """
    purch_recs = (
        purch_recs.set_index(["Panel Id", "att_vol"])[["Energy KCal"]]
        .unstack(["att_vol"])
        .fillna(0)
    )
    purch_recs.columns = purch_recs.columns.droplevel()
    return scale_hh(purch_recs, MinMaxScaler())  # Scale the hh purchases 0 to 1


def scale_hh(df: pd.DataFrame, scaler: Function):
    """
    Applies a scaler to each row of household purchases.

    Args:
        df (pd.DataFrame): Pandas dataframe household purchases by food category
        scaler (function): Sklearn scaler to apply

    Returns:
        pd.DateFrame: Household totals scaled by rows.
    """
    return pd.DataFrame(
        scaler.fit_transform(df.T).T, columns=list(df.columns), index=df.index
    )


def proportion_hh(df: pd.DataFrame):
    """
    Transforms total values of categories into proportions  of the total values for each household

    Args:
        df (pd.DataFrame): Pandas dataframe household purchases by food category

    Returns:
        pd.DateFrame: Household purchases per category as proportions of total purchases
    """
    return df.div(df.sum(axis=1), axis=0)


def food_cat_represent(df: pd.DataFrame):
    """
    Transforms household puchases to show how over / under-represented a category is for a household.

    Args:
        df (pd.DataFrame): Pandas dataframe household purchases by food category

    Returns:
        pd.DateFrame: Household purchases per category
    """
    return (df.div(df.sum(axis=1), axis=0)).div(
        list(df.sum() / (df.sum().sum())), axis=1
    )


def total_nutrition_intake(cluster: pd.DataFrame):
    """
    Get total nutritional volume per category per cluster.

    Args:
        df (pd.DataFrame): Pandas dataframe of nutritional intake for cluster

    Returns:
        pd.DateFrame: Total intake per nurtitional group
    """
    c_total = cluster.groupby(by=["Attribute Value Description"]).sum()[
        [
            "gross_up_vol",
            "Energy KJ",
            "Energy KCal",
            "Protein KG",
            "Carbohydrate KG",
            "Sugar KG",
            "Fat KG",
            "Saturates KG",
            "Fibre KG Flag",
            "Sodium KG",
        ]
    ]
    return c_total.loc[:, c_total.columns != "Volume"].multiply(
        c_total["Volume"], axis="index"
    )


def percent_demog_group(df: pd.DataFrame, col: str, clusters: str):
    """
    Percent of value per demographic group.

    Args:
        df (pd.DataFrame): Pandas dataframe
        col (str): Name of column to create percentage values
        clusters (str): Name of column with cluster labels

    Returns:
        perc_demographic (pd.DateFrame): Table of percentages per value of demographic group
    """
    df["Percent"] = 1
    perc_demographic = (df.groupby([clusters, col])["Percent"].sum()) / (
        df.groupby([clusters])["Percent"].sum()
    )
    perc_demographic = perc_demographic.reset_index()
    perc_demographic["Percent"] = perc_demographic["Percent"] * 100
    return perc_demographic


def bmi_households(pan_ind_mast: pd.DataFrame):
    """
    Creates table showing the size, number of adults with high BMI and BMI missing per household/

    Args:
        df (pd.DataFrame): Pandas dataframe

    Returns:
        pd.DateFrame: Table of BMI info per household
    """
    pan_ind_mast["adults"] = np.where(
        (pan_ind_mast["BMI"] > 0) & (pan_ind_mast["Age"] > 16), 1, 0
    )
    pan_ind_mast["high_bmi_adult"] = np.where(
        (pan_ind_mast["BMI"] >= 25) & (pan_ind_mast["Age"] > 16), 1, 0
    )
    pan_ind_mast["bmi_missing"] = np.where(pan_ind_mast["BMI"] == 0, 1, 0)

    pan_ind_bmi = pan_ind_mast[
        ["Panel Id", "high_bmi_adult", "bmi_missing", "adults"]
    ].copy()
    pan_ind_bmi["household_size"] = 1
    return pan_ind_bmi.groupby(by=["Panel Id"]).sum()
