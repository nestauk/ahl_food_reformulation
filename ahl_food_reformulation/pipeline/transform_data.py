# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: ahl_food_reformulation
#     language: python
#     name: ahl_food_reformulation
# ---

# %%

# Import libraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import os.path
from ahl_food_reformulation import PROJECT_DIR


def create_subsets(date_period):
    """
    Creates subset of defined month. First checks if files exists before creating.
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


def combine_files(val_fields, pur_recs, prod_mast, uom, prod_codes, prod_vals, att_num):
    """
    Performs multiple merges and a few cleaning functions to combine the following files into one:
    val_fields, pur_records, prod_mast, uom, prod_codes, prod_vals
    """
    val_fields.drop_duplicates(inplace=True)  # Remove duplicates
    # Need to confirm with Kantar on the use of the weights
    pur_recs["gross_up_vol"] = pur_recs["Volume"]
    #  pur_recs["Volume"] * pur_recs["Gross Up Weight"]
    # )  # Gross up volume (uk)
    # Merge files
    pur_recs = pur_recs[
        ["PurchaseId", "Panel Id", "Period", "Product Code", "gross_up_vol"]
    ].merge(
        prod_mast[["Product Code", "Validation Field"]], on="Product Code", how="left"
    )
    pur_recs = pur_recs.merge(
        val_fields[["VF", "UOM"]], left_on="Validation Field", right_on="VF", how="left"
    )
    pur_recs = pur_recs.merge(uom[["UOM", "Reported Volume"]], on="UOM", how="left")
    rst_4_ext = prod_codes[prod_codes["Attribute Number"] == att_num].copy()
    prod_code_vals = rst_4_ext.merge(
        prod_vals, left_on="Attribute Value", right_on="Attribute Code", how="left"
    )
    pur_recs = pur_recs.merge(
        prod_code_vals[["Product Code", "Attribute Code Description"]],
        on="Product Code",
        how="left",
    )
    pur_recs = pur_recs[
        pur_recs["Reported Volume"].notna()
    ]  # Remove purchases with no volume
    pur_recs["att_vol"] = (
        pur_recs["Attribute Code Description"] + "_" + pur_recs["Reported Volume"]
    )
    return pur_recs


class GroupByScaler(BaseEstimator, TransformerMixin):
    """
    Scales subsets of a column based on the values in another column.
    Scaling done using MinMaxScaler() (values from 0-1)
    Taken from: https://stackoverflow.com/questions/68356000/how-to-standardize-scikit-learn-by-group
    """

    def __init__(self, by=None):
        self.scalers = dict()
        self.by = by

    def fit(self, X, y=None):
        self.cols = X.select_dtypes(exclude=["object"]).columns
        for val in X[self.by].unique():
            mask = X[self.by] == val
            X_sub = X.loc[mask, self.cols]
            self.scalers[val] = MinMaxScaler().fit(X_sub)
        return self

    def transform(self, X, y=None):
        for val in X[self.by].unique():
            mask = X[self.by] == val
            X.loc[mask, self.cols] = self.scalers[val].transform(X.loc[mask, self.cols])

        return X


def norm_variable(data):
    """normalise variable between 0 and 1"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# %%
# Groupby to get category totals per household
def hh_total_categories(df):
    """
    Groups by the household id and category summing the grossed up volume.
    """
    return (
        df.groupby(["Panel Id", "att_vol"])["gross_up_vol"]
        .sum()
        .unstack(["att_vol"])
        .fillna(0)
    )


# %%
def scale_hh(df, scaler):
    """
    Applies a scaler to each row of household purchases.
    """
    return pd.DataFrame(
        scaler.fit_transform(df.T).T, columns=list(df.columns), index=df.index
    )


def proportion_hh(df):
    """
    Transforms total values of categories into proportions  of the total values for each household
    """
    return df.div(df.sum(axis=1), axis=0)


def food_cat_represent(df):
    return (df.div(df.sum(axis=1), axis=0)).div(
        list(df.sum() / (df.sum().sum())), axis=1
    )


# %%
