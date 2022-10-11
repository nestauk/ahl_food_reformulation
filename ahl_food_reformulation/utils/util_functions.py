import pandas as pd
import numpy as np


def reduce_df(df: pd.DataFrame, min_number: int, prod_category: str):
    """
    Slices df by categories with > min_number of products

    Args:
        df (pd.DataFrame): Dataframe of product and category information
        min_number (int): Min number of products to be in df (greater than)
        prod_category (str): Name of product category

    Returns:
        (pd.DateFrame): Sliced df
    """
    to_keep = (
        df[prod_category]
        .value_counts()[df[prod_category].value_counts() > min_number]
        .index
    )

    return df[df[prod_category].isin(to_keep)].copy()


def number_prods_cat(df: pd.DataFrame, prod_category: str):
    """
    Get total products per category

    Args:
        df (pd.DataFrame): Dataframe of product and category information
        prod_category (str): Name of product category

    Returns:
        (pd.DateFrame): Groupby total of products per category
    """
    return (
        df.groupby(prod_category)
        .size()
        .reset_index()
        .rename({0: "number_products"}, axis=1)
    )


def iqr(data: list):
    """
    calculate interquartile range (IQR) of a list

    Args:
        data (list): List of numbers

    Returns:
        (int): IQR value
    """
    q3, q1 = np.percentile(data, [75, 25])
    return q3 - q1
