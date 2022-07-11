# Returns the files from the kantar dataset
import pandas as pd
import os.path
from ahl_food_reformulation import PROJECT_DIR


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
        pur_recs = pd.read_csv(f"{PROJECT_DIR}/inputs/data/purchase_records.csv")
        subset = pur_recs[pur_recs["Period"] == date_period]
        subset.to_csv(file_path, index=False)
        return subset


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
        names=["UOM", "Measure Description", "Factor", "Reported Volume"],
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
    """ """
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
    return pd.read_csv(PROJECT_DIR / "outputs/data/panel_demographic_table_202110.csv")
