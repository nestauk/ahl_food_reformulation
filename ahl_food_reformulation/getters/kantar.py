# Returns the files from the kantar dataset
import pandas as pd
import os.path
from ahl_food_reformulation import PROJECT_DIR


def purchase_subsets(date_period):
    """Creates subset from the purchase records of defined month. First checks if files exists before creating.
    The purchase records file describes the unique purchases of products.
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
    """Dataset of unique products used by households."""
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/product_master.csv", encoding="ISO-8859-1"
    )


def val_fields():
    """ """
    return pd.read_csv(PROJECT_DIR / "inputs/data/validation_field.csv")


def uom():
    """ """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/uom.csv",
        header=0,
        names=["UOM", "Measure Description", "Factor", "Reported Volume"],
    )


def product_codes():
    """ """
    return pd.read_csv(PROJECT_DIR / "inputs/data/product_attribute_coding.csv")


def product_values():
    """ """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/product_attribute_values.csv", encoding="ISO-8859-1"
    )


def panel_clusters():
    """ """
    return pd.read_csv(PROJECT_DIR / "outputs/data/panel_clusters.csv")


def nutrition():
    """ """
    return pd.read_csv(
        PROJECT_DIR / "inputs/data/nutrition_data.csv", encoding="ISO-8859-1"
    )


def household_demog():
    """ """
    return pd.read_csv(PROJECT_DIR / "outputs/data/panel_demographic_table_202110.csv")
