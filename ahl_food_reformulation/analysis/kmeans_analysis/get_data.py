# %%
# Returns the files from the kantar dataset
import pandas as pd
import re
import os.path
from ahl_food_reformulation import PROJECT_DIR


# %%
def purchase_records():
    """Reads all the purchase records"""

    return pd.read_csv(f"{PROJECT_DIR}/inputs/data/purchase_records.csv")


# %%
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


# %%
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


# %%
def product_master():
    """Reads in dataset of unique products used by households.

    Args: None

    Returns: pd.DataFrame: product master dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/product_master.csv", encoding="ISO-8859-1"
    )


# %%
def val_fields():
    """Reads in dataset of codes to merge product master and uom information

    Args: None

    Returns: pd.DataFrame: validation fields dataframe
    """
    return pd.read_csv(PROJECT_DIR / "inputs/data/validation_field.csv")


# %%
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


# %%
def product_codes():
    """Reads in dataset which contains the codes to link products to category information

    Args: None

    Returns: pd.DataFrame: product codes dataframe
    """
    return pd.read_csv(PROJECT_DIR / "inputs/data/product_attribute_coding.csv")


# %%
def product_values():
    """Reads in dataset containing the product category information

    Args: None

    Returns: pd.DataFrame: product values dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/product_attribute_values.csv", encoding="ISO-8859-1"
    )


# %%
def t_tests():
    """Reads in dataset containing the product category information

    Args: None

    Returns: pd.DataFrame: product values dataframe
    """
    return pd.read_csv(PROJECT_DIR / "outputs/data/t_tests_features.csv")
