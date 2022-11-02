from ahl_food_reformulation.pipeline import transform_data as transform
from functools import reduce
import pandas as pd


def make_impact(
    chosen_cats,
    kcal_est,
    val_fields,
    pur_recs,
    prod_codes,
    prod_vals,
    prod_meta,
    nut_rec,
):
    """
    Generate data needed to determine impact of reformulation on the population.

    Parameters
    ----------
    chosen_cats : pd.DataFrame
        pandas DataFrame with list of detailed categories nested within broader categories.
    kcal_est : pd.DataFrame
        pandas DataFrame with values for calorie reduction in case of reformulation
    val_fields : pd.DataFrame
        Pandas dataframe with codes to merge product master and uom dfs
    pur_recs : pd.DataFrame
        Pandas dataframe contains the purchase records of specified data
    prod_codes : pd.DataFrame
        Pandas dataframe contains the codes to link products to category information
    prod_vals : pd.DataFrame
        Pandas dataframe contains the product category information
    prod_meta : pd.DataFrame
        Pandas dataframe contains the product information
    nut_rec : pd.DataFrame
        Pandas dataframe contains the nutritional information of specified data.

    Returns
    -------
    pandas dataframe with values of calories per household under different scenarios

    """

    target_red = chosen_cats.merge(kcal_est, on="rst_4_market_sector")

    # Purchase and product info combined
    comb_files = transform.combine_files(
        val_fields, pur_recs, prod_codes, prod_vals, 2907
    ).drop("att_vol", axis=1)

    comb_update = comb_files.merge(
        prod_meta[["product_code", "rst_4_extended", "rst_4_market_sector"]],
        left_on="Product Code",
        right_on="product_code",
    )

    comb_update.rename(columns={"rst_4_extended": "att_vol"}, inplace=True)

    # get number of periods each hh is present
    period_n = (
        comb_update[["Panel Id", "Period"]]
        .drop_duplicates()
        .groupby(["Panel Id"])["Period"]
        .size()
        .reset_index(name="period_n")
    )
    period_n["days"] = period_n["period_n"] * 28

    # sum of all kcal purchased by category by an household
    purch_recs_comb = transform.make_purch_records(nut_rec, comb_update, ["att_vol"])

    purch_recs_comb_scenarios = (
        purch_recs_comb.merge(
            target_red, right_on="rst_4_extended", left_on="att_vol", how="left"
        )
        .fillna(0)
        .merge(period_n, on="Panel Id")
    )

    purch_recs_comb_scenarios["Gross_up_kcal_min"] = purch_recs_comb_scenarios[
        "Gross_up_kcal"
    ] * (1 - purch_recs_comb_scenarios["min"])
    purch_recs_comb_scenarios["Gross_up_kcal_max"] = purch_recs_comb_scenarios[
        "Gross_up_kcal"
    ] * (1 - purch_recs_comb_scenarios["max"])

    return purch_recs_comb_scenarios


def kcal_day(purch_recs_comb_scenarios, pan_ind, panel_weight, scenario, low, high):
    """
    Returns summary statistics for a given scenario (no reformulation, high reformulation, low reformulation)

    Parameters
    ----------
    purch_recs_comb : pd.DataFrame
        pandas DataFrame with sum of all kcal purchased by category by an household
    pan_ind : pd.DataFrame
        pandas DataFrame with conversion factor.
    panel_weight : pd.DataFrame
        pandas DataFrame with demographic weights.
    scenario : LIST
        ["Gross_up_kcal", "Gross_up_kcal_max", "Gross_up_kcal_min"]
    low : float
        percentile for trimming outliers at the bottom
    high : float
        percentile for trimming outliers at the top

    Returns
    -------
    pandas Series with descriptive statistics for daily kcal distribution

    """

    panel_weight.rename(columns={"panel_id": "Panel Id"}, inplace=True)

    # generate conversion factor
    pan_conv = transform.hh_size_conv(pan_ind)

    pan_conv_weighted = pan_conv.merge(panel_weight, on="Panel Id", how="inner")
    pan_conv_weighted["conversion"] = (
        pan_conv_weighted["conversion"] * pan_conv_weighted["demographic_weight"]
    )
    pan_conv_weighted = pan_conv_weighted[["Panel Id", "conversion"]]

    hh_kcal = (
        purch_recs_comb_scenarios.groupby(["Panel Id", "days"])[scenario]
        .sum()
        .reset_index(name=scenario)
    )

    hh_kcal_weighted = hh_kcal.merge(pan_conv_weighted, on="Panel Id", how="inner")

    hh_kcal_weighted[scenario + "_daily"] = (
        hh_kcal_weighted[scenario]
        / hh_kcal_weighted["conversion"]
        / hh_kcal_weighted["days"]
    )

    q_hi = hh_kcal_weighted[scenario + "_daily"].quantile(high)
    q_low = hh_kcal_weighted[scenario + "_daily"].quantile(low)

    hh_kcal_filter = hh_kcal_weighted[
        (hh_kcal_weighted[scenario + "_daily"] < q_hi)
        & (hh_kcal_weighted[scenario + "_daily"] > q_low)
    ].copy()

    return hh_kcal_filter[["Panel Id", scenario + "_daily"]]


def kcal_day_describe(hh_kcal_filter):
    """
    Descriptive statistics for the data parsed

    Parameters
    ----------
    hh_kcal_filter : pd.DataFrame
        pandas dataframe containing clean data of kcal/day under 3 scenarios

    Returns
    -------
    pandas Dataframe with descriptive statistics

    """

    return hh_kcal_filter.describe()


def compare_scenarios(panel_weight, purch_recs_comb_scenarios, pan_ind):
    """


    Parameters
    ----------
    panel_weight : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    panel_weight.rename(columns={"panel_id": "Panel Id"}, inplace=True)

    data_frames = [
        kcal_day(
            purch_recs_comb_scenarios,
            pan_ind,
            panel_weight,
            "Gross_up_kcal",
            0.05,
            0.95,
        ),
        kcal_day(
            purch_recs_comb_scenarios,
            pan_ind,
            panel_weight,
            "Gross_up_kcal_min",
            0.05,
            0.95,
        ),
        kcal_day(
            purch_recs_comb_scenarios,
            pan_ind,
            panel_weight,
            "Gross_up_kcal_max",
            0.05,
            0.95,
        ),
        panel_weight,
    ]

    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=["Panel Id"], how="outer"),
        data_frames,
    )

    df_merged["pos_diff_5"] = (
        df_merged["Gross_up_kcal_daily"] - df_merged["Gross_up_kcal_min_daily"]
    )
    df_merged["pos_diff_10"] = (
        df_merged["Gross_up_kcal_daily"] - df_merged["Gross_up_kcal_max_daily"]
    )

    return df_merged
