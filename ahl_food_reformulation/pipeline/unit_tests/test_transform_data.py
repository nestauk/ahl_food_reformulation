## Unit tests for transform_data functions in pipeline

# Import libraries
import pytest
import pandas as pd
from ahl_food_reformulation.pipeline import transform_data

# Set fixture dfs to use in tests
@pytest.fixture()
def df():
    df = pd.DataFrame(
        [[50, 50], [400, 600]], index=["hh1", "hh2"], columns=["cat1", "cat2"]
    )

    return df


@pytest.fixture()
def purchases():
    purchases = pd.DataFrame(
        [
            [5, 50, 100, "panel_1", 3, "xx1"],
            [0, 100, 100, "panel_2", 3, "xx1"],
            [10, 150, 100, "panel_1", 3, "xx1"],
            [5, 200, 100, "panel_1", 3, "xx2"],
            [1, 3, 100, "panel_3", 3, "xx4"],
        ],
        columns=[
            "Volume",
            "Energy KCal",
            "Gross Up Weight",
            "Panel Id",
            "Quantity",
            "Category_id",
        ],
    )
    return purchases


# hh_total_categories
def test_hh_total_categories(purchases):
    """Tests the total_product_hh_purchase function using dummy dataset"""
    # Expected
    df_exp = pd.DataFrame(
        [
            [15.0, 5.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        columns=["xx1", "xx2", "xx4"],
        index=["panel_1", "panel_2", "panel_3"],
    )
    df_exp.index.name = "Panel Id"
    # test the result
    assert transform_data.hh_total_categories(purchases, "Category_id").equals(df_exp)


# total_product_hh_purchase
def test_total_product_hh_purchase(purchases):
    """Tests the total_product_hh_purchase function using dummy dataset"""
    # Expected
    df_exp = pd.DataFrame(
        [
            ["panel_1", "xx1", 15, 200, 6, 200, 20000],
            ["panel_1", "xx2", 5, 200, 3, 100, 20000],
            ["panel_3", "xx4", 1, 3, 3, 100, 300],
        ],
        columns=[
            "Panel Id",
            "Category_id",
            "Volume",
            "Energy KCal",
            "Quantity",
            "Gross Up Weight",
            "Gross_up_kcal",
        ],
    )
    # test the result
    assert transform_data.total_product_hh_purchase(purchases, ["Category_id"]).equals(
        df_exp
    )


# proportion_hh
def test_proportion_hh(df):
    """Tests the proportion_hh function using dummy dataset"""
    # Expected
    df_exp = pd.DataFrame(
        [[0.5, 0.5], [0.4, 0.6]], index=["hh1", "hh2"], columns=["cat1", "cat2"]
    )
    # test the result
    assert transform_data.proportion_hh(df).equals(df_exp)


# scale_hh
