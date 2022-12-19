from doctest import OutputChecker
from ahl_food_reformulation.utils import util_functions as util_func
from ahl_food_reformulation.pipeline.preprocessing import energy_density as energy
from ahl_food_reformulation.pipeline import nutrient_metrics_funcs as nutrient
from ahl_food_reformulation.pipeline.preprocessing import transform_data as transform
import pandas as pd


def hh_kcal_weight(
    prod_cat: int,
    pur_recs: pd.DataFrame,
    nut_recs: pd.DataFrame,
    prod_meta: pd.DataFrame,
):
    """
    Create weighted hh kcal per cat

    Args:
        prod_category (int): one product category
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
    Returns:
        pd.DataFrame: Table with metrics based on kcal contribution per category
    """
    comb_files = pur_recs.merge(
        prod_meta[["product_code", prod_cat]],
        left_on=["Product Code"],
        right_on="product_code",
        how="left",
    )
    comb_files = comb_files[
        comb_files["Reported Volume"].notna()
    ]  # Remove purchases with no volume
    comb_files["att_vol"] = comb_files[prod_cat]
    comb_files.drop("product_code", axis=1, inplace=True)
    # Make household representations
    purch_recs_comb = transform.make_purch_records(nut_recs, comb_files, ["att_vol"])
    return transform.hh_kcal_per_prod(purch_recs_comb, "Gross_up_kcal")


def kcal_contr_table(
    prod_cat: int,
    pan_ind: pd.DataFrame,
    pur_recs: pd.DataFrame,
    nut_recs: pd.DataFrame,
    prod_meta: pd.DataFrame,
    panel_weight: pd.DataFrame,
):
    """
    Create kcal contribution metrics table based on chosen category

    Args:
        prod_category (int): one product category
        pan_ind (pd.DataFrame): Pandas dataframe of individual household member info
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
        panel_weight (pd.Dataframe): Pandas dataframe of demographic weights
    Returns:
        pd.DataFrame: Table with metrics based on kcal contribution per category
    """
    hh_kcal_weighted = hh_kcal_weight(prod_cat, pur_recs, nut_recs, prod_meta)

    # Converted household size
    pan_conv = transform.hh_size_conv(pan_ind)
    pan_conv_weighted = pan_conv.merge(
        panel_weight, left_on="Panel Id", right_on="panel_id", how="inner"
    )
    pan_conv_weighted["conversion"] = (
        pan_conv_weighted["conversion"] * pan_conv_weighted["demographic_weight"]
    )
    pan_conv_weighted = pan_conv_weighted[["Panel Id", "conversion"]]

    hh_kcal_conv_weighted = transform.apply_hh_conv(
        hh_kcal_weighted, pan_conv_weighted
    ).dropna(axis=0)

    # Create table
    kcal_cont_df = pd.concat(
        [
            (hh_kcal_weighted.sum() / hh_kcal_weighted.sum().sum()) * 100,
            (hh_kcal_conv_weighted.sum() / hh_kcal_conv_weighted.sum().sum()) * 100,
            (hh_kcal_conv_weighted.median()) / 365,
            (hh_kcal_conv_weighted.mean()) / 365,
            (hh_kcal_conv_weighted.apply(util_func.iqr)) / 365,
        ],
        axis=1,
    )
    kcal_cont_df.columns = [
        "percent_kcal_contrib_weighted",
        "percent_kcal_contrib_size_adj_weighted",
        "median_kcal_size_adj_weighted",
        "mean_kcal_size_adj_weighted",
        "IQR_kcal_size_adj_weighted",
    ]
    return kcal_cont_df


def kcal_density_table(
    prod_category: str,
    pur_recs: pd.DataFrame,
    nut_recs: pd.DataFrame,
    prod_meta: pd.DataFrame,
    prod_meas: pd.DataFrame,
    sample_size: int,
):
    """
    Create kcal density metrics table based on chosen category

    Args:
        prod_category (str): one product category
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
        prod_meas (pd.DataFrame): Pandas dataframe with additional conversions to g and ml for unit and serving products
        sample_size (int): Number of samples to use for entropy and variance averages
    Returns:
        pd.DataFrame: Table with metrics based on energy density scores per category
    """
    df_prod_ed = energy.prod_energy_100(
        prod_category,
        pur_recs,
        nut_recs,
        prod_meta,
        prod_meas,
    )
    energy_dens_agg = energy.cat_energy_100(prod_category, df_prod_ed)
    df_prod_ed["energy_density_cat"] = energy.energy_density_score(
        df_prod_ed["kcal_100g_ml"]
    )

    ed_cats_sales = (
        df_prod_ed.groupby([prod_category, "energy_density_cat"])["total_sale"].sum()
        / df_prod_ed.groupby([prod_category])["total_sale"].sum()
    ) * 100
    ed_cats_sales = ed_cats_sales.reset_index().rename(
        {"total_sale": "percent_high_ed_sales_weighted"}, axis=1
    )

    ed_cats_num = (
        df_prod_ed.groupby([prod_category, "energy_density_cat"]).size()
        / df_prod_ed.groupby([prod_category]).size()
    ) * 100
    ed_cats_num = ed_cats_num.reset_index().rename({0: "percent_high_ed"}, axis=1)

    metrics_table = (
        ed_cats_sales[ed_cats_sales.energy_density_cat == "high"]
        .copy()
        .merge(
            ed_cats_num[ed_cats_num.energy_density_cat == "high"].copy(),
            on=[prod_category, "energy_density_cat"],
        )
        .merge(energy_dens_agg, on=prod_category)
    )

    df_prod_ed_reduce = util_func.reduce_df(df_prod_ed, sample_size, prod_category)[
        [prod_category, "chosen_unit", "kcal_100g_ml"]
    ]
    df_diversity = nutrient.create_diversity_df(
        df_prod_ed_reduce, prod_category, 100, sample_size
    )
    num_prods_cat = util_func.number_prods_cat(df_prod_ed, prod_category)

    return (
        metrics_table.merge(df_diversity, on=prod_category, how="left")
        .merge(num_prods_cat, on=prod_category)
        .drop(["count", "energy_density_cat"], axis=1)
    ).set_index(prod_category)


def create_report_table(
    kcal_density_df: pd.DataFrame, kcal_cont_df: pd.DataFrame, clust_table: pd.DataFrame
):
    """
    Merges kcal density and kcal contribution tables

    Args:
        kcal_density_df (pd.DataFrame): Pandas dataframe with kcal density metrics
        kcal_cont_df (pd.DataFrame): Pandas dataframe with kcal contribution metrics
        clust_table (pd.DataFrame): Pandas dataframe with cluster metrics
    Returns:
        pd.DataFrame: Merged table of metrics
    """
    return pd.merge(
        kcal_cont_df, kcal_density_df, left_index=True, right_index=True, how="outer"
    ).merge(clust_table, left_index=True, right_index=True, how="outer")
