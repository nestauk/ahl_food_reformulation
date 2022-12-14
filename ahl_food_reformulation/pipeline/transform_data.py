# Import libraries
from pyclbr import Function
import pandas as pd
import numpy as np
from ahl_food_reformulation.utils import lookups as lps


def combine_files(
    val_fields: pd.DataFrame,
    pur_recs: pd.DataFrame,
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
        [
            "PurchaseId",
            "Panel Id",
            "Period",
            "Product Code",
            "Volume",
            "Quantity",
            "Reported Volume",
            "Gross Up Weight",
        ]
    ]  # .merge(
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
    pur_recs["att_vol"] = pur_recs["Attribute Value Description"]
    return pur_recs


def vol_for_purch(
    pur_recs: pd.DataFrame,
    val_fields: pd.DataFrame,
    prod_mast: pd.DataFrame,
    uom: pd.DataFrame,
):
    """Adds volume meausrement to purchase records

    Args:
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        prod_mast (pd.DataFrame): Pandas dataframe unique product list
        uom (pd.DataFrame): Panadas dataframe contains product measurement information

    Returns:
        (pandas.DateFrame): Merged pandas dataframe
    """
    val_fields.drop_duplicates(inplace=True)  # Remove duplicates
    return (
        pur_recs.merge(
            prod_mast[["Product Code", "Validation Field"]],
            on="Product Code",
            how="left",
        )
        .merge(
            val_fields[["VF", "UOM"]],
            left_on="Validation Field",
            right_on="VF",
            how="left",
        )
        .merge(uom[["UOM", "Reported Volume"]], on="UOM", how="left")
        .drop(["Validation Field", "VF", "UOM"], axis=1)
    )


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


def total_product_hh_purchase(purch_recs: pd.DataFrame, cols):
    """Groups by household, measurement and product and sums the volume and kcal content.
    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        cols (list): List of cols to group (different for kcal and volume representations)
    Returns:
        (pandas.DateFrame): groupby pandas dataframe
    """
    # Remove cases where volume is zero (8 cases)
    purch_recs = purch_recs[purch_recs["Volume"] != 0].copy()
    purch_recs["Gross_up_kcal"] = (
        purch_recs["Energy KCal"] * purch_recs["Gross Up Weight"]
    )
    return (
        purch_recs.groupby(["Panel Id"] + cols)[
            ["Volume", "Energy KCal", "Quantity", "Gross Up Weight", "Gross_up_kcal"]
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
    nutrition: pd.DataFrame, purchases_comb: pd.DataFrame, cols: list
):
    """
    Merges dataframes to create purchase records df with food category and nutrition information
    Args:
        nutrition (pd.DataFrame): Pandas dataframe of purchase level nutritional information
        purchases_comb (pd.DataFrame): Combined files to give product informaion to purchases
        cols (list): Columns to use for groupby
    Returns:
        pd.DateFrame: Household totals per food category
    """
    purchases_nutrition = nutrition_merge(nutrition, purchases_comb, ["Energy KCal"])
    return total_product_hh_purchase(purchases_nutrition, cols)


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


def hh_kcal_per_prod(pur_recs: pd.DataFrame, kcal_col: str):
    """
    Unstacks df to show total kcal per product per household then normalises by household (rows)

    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        kcal_col (str): Energy Kcal column (weighted or unweighted)

    Returns:
        (pd.DateFrame): Kcal totals per product per household
    """
    purch_recs = (
        pur_recs.set_index(["Panel Id", "att_vol"])[[kcal_col]]
        .unstack(["att_vol"])
        .fillna(0)
    )
    purch_recs.columns = purch_recs.columns.droplevel()
    return purch_recs


def hh_kcal_per_prod_red(purch_recs: pd.DataFrame, kcal_col: str):
    """
    Unstacks df to show total kcal per product per household then normalises by household (rows)

    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        kcal_col (str): Energy Kcal column (weighted or unweighted)

    Returns:
        (pd.DateFrame): Kcal totals per product per household
    """
    prod_kcal = hh_kcal_per_prod(purch_recs, kcal_col)
    prod_kcal.drop(list(prod_kcal.filter(regex="Oil")), axis=1, inplace=True)
    prod_kcal.drop(list(prod_kcal.filter(regex="Rice")), axis=1, inplace=True)
    return prod_kcal


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


def scale_df(scaler: Function, df: pd.DataFrame):
    array_reshaped = df.to_numpy().reshape(-1, 1)
    scaled = scaler.fit_transform(array_reshaped).reshape(len(df), df.shape[1])
    return pd.DataFrame(scaled, columns=list(df.columns), index=df.index)


def hh_kcal_per_category(
    nut: pd.DataFrame,
    scaler_type: Function,
    comb_files: pd.DataFrame,
):
    """
    Unstacks df to show total kcal per product per household then normalises by household (rows)
    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut (pd.DataFrame): Pandas dataframe contains nutritional information per purchase record
        scaler_type: Scaler function to apply to normalise data
        cat (int): Number ID of product category
        comb_files (pd.DataFrame): Combined purchase and product info
    Returns:
        (pd.DateFrame): Kcal totals per product per household normalised by total household kcal
    """
    purch_recs_comb = make_purch_records(nut, comb_files, ["att_vol"])
    return scale_hh(
        hh_kcal_per_prod(purch_recs_comb, "Energy KCal"), scaler_type
    )  # Scale the hh purchases 0 to 1


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


def hh_size_conv(pan_ind: pd.DataFrame):
    """
    Applies a scaler to each row of household purchases.

    Args:
        pan_ind (pd.DataFrame): Pandas dataframe of household members

    Returns:
        pd.DateFrame: Household converted total size
    """
    d = {
        "age_group": [
            "0-1",
            "1-3",
            "4-6",
            "7-10",
            "11-14",
            "15-18",
            "19-24",
            "25-50",
            "51+",
            "11-14",
            "15-18",
            "19-24",
            "25-50",
            "51+",
        ],
        "group": [
            "children",
            "children",
            "children",
            "children",
            "M",
            "M",
            "M",
            "M",
            "M",
            "F",
            "F",
            "F",
            "F",
            "F",
        ],
        "conversion": [
            0.29,
            0.51,
            0.71,
            0.78,
            0.98,
            1.18,
            1.14,
            1.14,
            0.90,
            0.86,
            0.86,
            0.86,
            0.86,
            0.75,
        ],
    }
    conversion_table = pd.DataFrame(data=d)

    bins = [0, 1, 4, 7, 11, 15, 19, 25, 51, 150]
    labels = ["0-1", "1-3", "4-6", "7-10", "11-14", "15-18", "19-24", "25-50", "51+"]

    pan_ind["age_group"] = pd.cut(pan_ind["Age"], bins=bins, labels=labels, right=False)
    pan_ind["group"] = np.where(pan_ind["Age"] < 11, "children", pan_ind["Gender"])
    pan_ind_conv = pan_ind.merge(
        conversion_table, how="left", on=["age_group", "group"]
    ).copy()
    return pan_ind_conv.groupby(["Panel Id"])["conversion"].sum().reset_index()


def apply_hh_conv(hh_kcal: pd.DataFrame, pan_conv: pd.DataFrame):
    """
    Applies a scaler to each row of household purchases.

    Args:
        hh_kcal (pd.DataFrame): Pandas dataframe of households kcal totals
        pan_conv (pd.DataFrame): Pandas dataframe of households converted totals

    Returns:
        pd.DateFrame: Household converted total kcal purchased
    """
    df_merged = hh_kcal.merge(
        pan_conv.set_index("Panel Id"), how="left", left_index=True, right_index=True
    )
    return df_merged.loc[:, df_merged.columns != "conversion"].div(
        df_merged.conversion, axis=0
    )


def hh_kcal_volume_converted(
    nut: pd.DataFrame,
    pan_conv: pd.DataFrame,
    scaler: Function,
    comb_files: pd.DataFrame,
):
    """
    Converts the Applies a scaler to each row of household purchases.
    Args:
        nut (pd.DataFrame): Nutritional info per purchase
        pan_conv (pd.DataFrame): Pandas dataframe of households converted totals
        scaler (function): Normalising scaler
        comb_files (pd.DataFrame): Combined purchase/product file
    Returns:
        pd.DateFrame: Household converted total kcal purchased scaled
    """
    purch_recs_comb = make_purch_records(nut, comb_files, ["att_vol"])
    hh_kcal = hh_kcal_per_prod(purch_recs_comb, "Energy KCal")
    hh_kcal_conv = apply_hh_conv(hh_kcal, pan_conv)
    return scale_df(scaler, hh_kcal_conv)


def rst_4_market_sector_update(df: pd.DataFrame):
    """
    Updates rst_4_market_sector values to split out kilo/litre groups of products

    Args:
        df (pd.DataFrame): Df of products with rst_4_market_sector and 'rst_4_market' values

    Returns:
        df (pd.DataFrame): prod_meta df with updated rst_4_market_sector values

    """
    conditions = [
        df["rst_4_market_sector"].eq("Dairy Products")
        & df["rst_4_market"].eq("Total Cheese"),
        df["rst_4_market_sector"].eq("Dairy Products")
        & df["rst_4_market"].isin(["Eggs", "Butter", "Margarine"]),
        df["rst_4_market_sector"].eq("Dairy Products")
        & df["rst_4_market"].isin(["Yoghurt", "Fromage Frais"]),
        df["rst_4_market_sector"].eq("Dairy Products")
        & df["rst_4_market"].isin(["Total Milk", "Fresh Cream"]),
        df["rst_4_market_sector"].eq("Frozen Confectionery")
        & df["rst_4_market"].eq("Total Ice Cream"),
    ]
    choices = ["Cheese", "Eggs and Butter", "Yoghurt", "Milk and Cream", "Ice Cream"]
    return np.select(conditions, choices, default=df["rst_4_market_sector"])


def rst_4_market_update(df: pd.DataFrame):
    """
    Updates rst_4_market values to split out bread and cakes from particular cat

    Args:
        df (pd.DataFrame): Df of products with rst_4_market and rst_4_extended values

    Returns:
        df (pd.DataFrame): prod_meta df with updated rst_4_market_sector values

    """
    conditions = [
        df["rst_4_market"].eq("Morning Goods")
        & df["rst_4_extended"].isin(
            [
                "Crusty Bread Rolls",
                "Lavash Bread",
                "Morning Goods Bagels",
                "Morning Goods Baguettes",
                "Morning Goods Ciabatta",
                "Morning Goods Flatbread",
                "Morning Goods Sub Roll",
                "Morning Goods Tortilla Wraps",
                "Naan Bread",
                "Other Bread Rolls/Baps",
                "Part Baked Bread",
                "Pitta Bread",
                "Soft Bread Rolls",
            ]
        ),
    ]
    choices = ["Total Bread"]
    return np.select(conditions, choices, default=df["rst_4_market"])
