# Import libraries
from pyclbr import Function
import pandas as pd
import numpy as np
from ahl_food_reformulation.utils import lookups as lps


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


def add_energy_density(pur_rec_kilos: pd.DataFrame, nut_recs: pd.DataFrame):
    """
    Adds four columns to the purchase record:  energy_density (kcal per 1g),  energy_density_cat ('very_low', 'low', 'medium', 'high' based on thresholds), Reported Volume, kcal per 100g
    Args:
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe contains the nutritional information of specified data
    Returns:
        pd.DateFrame: Dataframe that is a copy of pur_rec with two additional columns with energy density info
    """
    # Convert to datetime format
    pur_rec_kilos["Purchase Date"] = pd.to_datetime(
        pur_rec_kilos["Purchase Date"], format="%d/%m/%Y"
    )

    # generate unique list of products
    unique_prods_nut = lps.products_per_100g(["Energy KCal"], pur_rec_kilos, nut_recs)

    # generate energy density variable
    unique_prods_nut["energy_density"] = unique_prods_nut["Energy KCal_100g"] / 100

    # generate energy density category variable based on standard thresholds
    unique_prods_nut["energy_density_cat"] = pd.cut(
        unique_prods_nut["energy_density"],
        bins=[0, 0.6, 1.5, 4, float("Inf")],
        labels=["very_low", "low", "medium", "high"],
    )

    # remove implausible values
    unique_prods_nut = unique_prods_nut[unique_prods_nut["energy_density"] < 20]

    # merge with purchase record
    return pur_rec_kilos.merge(unique_prods_nut, on="Product Code", how="left")


def perc_variable(data: pd.Series):
    """normalise variable between 0 and 1.

    Args:
        data (pd.Series): Pandas series, volume for different measurements.

    Returns:
        pd.Series: Normalised column
    """
    return (data / data.sum()) * 100
