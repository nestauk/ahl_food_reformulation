# Returns the files from the kantar dataset
import pandas as pd
import re
import os.path
from ahl_food_reformulation import PROJECT_DIR
from typing import Tuple, Dict
from toolz import pipe
from ahl_food_reformulation.getters.miscelaneous import postcode_region_lookup
from ahl_food_reformulation.utils.lookups import product_table
from ahl_food_reformulation.pipeline.transform_data import (
    rst_4_market_sector_update,
    rst_4_market_update,
)


def purchase_records():
    """Reads all the purchase records"""

    return pd.read_csv(f"{PROJECT_DIR}/inputs/data/purchase_records.csv")


def purchase_subsets(date_period):
    """
      Reads in purchase_records.csv and creates subset of purchase records file by defined month.
      First checks if files exists before creating.

    Args:
        date_period (int): Year and month to subset data (must match format in dataset - YYYYMM)

    Returns:
        subset (pd.DataFrame): Purchase records dataframe sliced by date_period.
    """
    file_path = f"{PROJECT_DIR}/outputs/data/pur_rec_" + str(date_period) + ".csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    else:
        subset_records = purchase_records().query(f"Period == {date_period}")
        subset_records.to_csv(file_path, index=False)
        return subset_records


def nutrition_subsets(date_period):
    """
      Reads in the nutrition.csv and creates subset of by defined month.
      First checks if files exists before creating.

    Args:
        date_period (int): Year and month to subset data (must match format in dataset - YYYYMM)

    Returns:
        subset (pd.DataFrame): Nutrition dataframe sliced by date_period.
    """
    file_path = f"{PROJECT_DIR}/outputs/data/nut_" + str(date_period) + ".csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    else:
        pur_recs = pd.read_csv(f"{PROJECT_DIR}/inputs/data/nutrition_data.csv")
        subset = pur_recs[pur_recs["Purchase Period"] == date_period]
        subset.to_csv(file_path, index=False)
        return subset


def purchase_records():
    """Reads in dataset of purchase level information

    Args: None

    Returns: pd.DataFrame: purchase records dataframe
    """
    return pd.read_csv(PROJECT_DIR / "inputs/data/purchase_records.csv")


def product_master():
    """Reads in dataset of unique products used by households.

    Args: None

    Returns: pd.DataFrame: product master dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/product_master.csv", encoding="ISO-8859-1"
    )


def val_fields():
    """Reads in dataset of codes to merge product master and uom information

    Args: None

    Returns: pd.DataFrame: validation fields dataframe
    """
    return pd.read_csv(PROJECT_DIR / "inputs/data/validation_field.csv")


def uom():
    """Reads in dataset of product measurement information

    Args: None

    Returns: pd.DataFrame: uom dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/uom.csv",
        header=0,
        names=["UOM", "Reported Volume"],
    )


def product_codes():
    """Reads in dataset which contains the codes to link products to category information

    Args: None

    Returns: pd.DataFrame: product codes dataframe
    """
    return pd.read_csv(PROJECT_DIR / "inputs/data/product_attribute_coding.csv")


def product_values():
    """Reads in dataset containing the product category information

    Args: None

    Returns: pd.DataFrame: product values dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/product_attribute_values.csv", encoding="ISO-8859-1"
    )


def product_attribute():
    """Reads in dataset containing information on the product attributes.

    Args: None

    Returns: pd.DataFrame: product values dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/product_attribute.csv", encoding="ISO-8859-1"
    )


def panel_clusters():
    """Reads in dataset containing the cluster labels per household

    Args: None

    Returns: pd.DataFrame: panel clusters dataframe
    """
    return pd.read_csv(PROJECT_DIR / "outputs/data/panel_clusters.csv")


def nutrition():
    """Reads in dataset of purchase level nutritional information

    Args: None

    Returns: pd.DataFrame: nutrition dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/nutrition_data.csv", encoding="ISO-8859-1"
    )


def household_master():
    """Reads in dataset of household information

    Args: None

    Returns: pd.DataFrame: household master dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/panel_household_master.csv", encoding="ISO-8859-1"
    )


def household_ind():
    """Reads in dataset of information about each household member

    Args: None

    Returns: pd.DataFrame: household individual dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/panel_individual_master.csv", encoding="ISO-8859-1"
    )


def demog_coding():
    """Reads in dataset of codes per household that links to demographic information

    Args: None

    Returns: pd.DataFrame: demographic coding dataframe
    """
    return pd.read_csv(PROJECT_DIR / "inputs/data/panel_demographic_coding.csv")


def demog_val():
    """Reads in dataset of demographic values per code

    Args: None

    Returns: pd.DataFrame: demographic values dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/panel_demographic_values.csv", encoding="ISO-8859-1"
    )


def household_demog():
    """Reads in dataset of household demographic information

    Args: None

    Returns: pd.DataFrame: household demographic dataframe
    """
    return pd.read_csv(PROJECT_DIR / "outputs/data/panel_demographic_table.csv")


def demog_clean() -> Tuple[pd.DataFrame, Dict]:
    """
    Reads a cleaned version of the kantar dataset
    """

    demog = household_demog()
    demog_col_lookup = {name: re.sub(" ", "_", name.lower()) for name in demog.columns}

    demog.columns = [demog_col_lookup[name] for name in demog.columns]

    demog = demog.rename(columns={"postocde_district": "postcode_district"})

    demog_clean_names = {v: k for k, v in demog_col_lookup.items()}

    return demog.assign(
        cluster=lambda df: df["panel_id"].map(
            panel_clusters().set_index("Panel Id")["clusters"].to_dict()
        )
    ).assign(region=lambda df: df["postcode_district"].map(postcode_region_lookup()))


def product_metadata() -> pd.DataFrame:
    """Table combining all the product metadata"""

    return pipe(
        product_table(
            val_fields(),
            product_master(),
            uom(),
            product_codes(),
            product_values(),
            product_attribute(),
        ),
        lambda df: df.rename(
            columns={c: re.sub(" ", "_", c.lower()) for c in df.columns}
        ),
    )


def product_measurement():
    """File containing all available measurements per product

    Args: None

    Returns: pd.DataFrame: list of product measurements
    """

    return pd.read_csv(
        f"{PROJECT_DIR}/inputs/data/Nesta - Units, Grams, Millilitres, Servings All Products.txt",
        encoding="ISO-8859-1",
    )


def panel_weights():
    """Reads the panel weights file"""

    return pd.read_csv(
        f"{PROJECT_DIR}/inputs/data/panel_demographic_weights_period.csv"
    )


def purchase_records_volume():
    """
    Getter for the copy of purchase record with imputed weights.

    Args:
        None

    Returns:
        df (pd.DataFrame): pruchase records with additional columns for volumes

    """

    return pd.read_csv(f"{PROJECT_DIR}/inputs/data/pur_rec_volume.csv").iloc[:, 1:]


def purchase_records_updated():
    """
    Getter for the copy of purchase record with imputed weights.
    Cleaned to format of purchase records but with reported volume added.

    Args:
        None

    Returns:
        df (pd.DataFrame): purchase records with imputed volumes

    """
    pur_recs_updated = purchase_records_volume()
    return pur_recs_updated.drop(
        ["Reported Volume", "volume_per", "Volume"], axis=1
    ).rename({"reported_volume_up": "Reported Volume", "volume_up": "Volume"}, axis=1)


def prod_meta_update():
    """
    Getter for the copy of prod_meta df with updated rst_4_market_sector values.

    Args:
        None

    Returns:
        df (pd.DataFrame): prod_meta df with updated rst_4_market_sector values

    """
    prod_meta = product_metadata()
    prod_meta["rst_4_market_sector"] = rst_4_market_sector_update(prod_meta)
    prod_meta["rst_4_market"] = rst_4_market_update(prod_meta)
    prod_meta.dropna(subset=["rst_4_extended"], inplace=True)

    return prod_meta[prod_meta["rst_4_market_sector"] != "Dairy Products"]


def panel_weights_year():
    """Reads the panel weights file"""

    return pd.read_csv(f"{PROJECT_DIR}/inputs/data/panel_demographic_weights_year.csv")


def cluster_kcal_share():
    """
    Getter for the lookup of households to cluster assignments according to the kcal share method

    Args:
        None

    Returns:
        df (pd.DataFrame): panel ids mapped to cluster labels


    """

    return pd.read_csv(
        f"{PROJECT_DIR}/outputs/data/alternative_clusters/panel_clusters_kcal_share.csv"
    )


def cluster_adj_size():
    """
    Getter for the lookup of households to cluster assignments according to the hh adjustment size method

    Args:
        None

    Returns:
        df (pd.DataFrame): panel ids mapped to cluster labels

    """
    return pd.read_csv(
        f"{PROJECT_DIR}/outputs/data/alternative_clusters/panel_clusters_adj_size.csv"
    )
