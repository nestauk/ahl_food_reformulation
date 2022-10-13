from doctest import OutputChecker
from ahl_food_reformulation.utils import util_functions as util_func
from ahl_food_reformulation.pipeline import energy_density as energy
from ahl_food_reformulation.pipeline import nutrient_metrics_funcs as nutrient
from ahl_food_reformulation.pipeline import transform_data as transform
import pandas as pd


def kcal_contr_table(
    prod_cat: int,
    pan_ind: pd.DataFrame,
    pur_recs: pd.DataFrame,
    nut_recs: pd.DataFrame,
    prod_meta: pd.DataFrame,
):
    """
    Create kcal contribution metrics table based on chosen category

    Args:
        prod_category (int): one product category
        pan_ind (pd.DataFrame): Pandas dataframe of individual household member info
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
    Returns:
        pd.DataFrame: Table with metrics based on kcal contribution per category
    """
    # Converted household size
    pan_conv = transform.hh_size_conv(pan_ind)
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
    hh_kcal = transform.hh_kcal_per_prod(purch_recs_comb)
    hh_kcal_conv = transform.apply_hh_conv(hh_kcal, pan_conv)
    # Create table
    kcal_cont_df = pd.concat(
        [
            (hh_kcal.sum() / hh_kcal.sum().sum()) * 100,
            (hh_kcal_conv.sum() / hh_kcal_conv.sum().sum()) * 100,
            hh_kcal_conv.median(),
            hh_kcal_conv.mean(),
            hh_kcal_conv.apply(util_func.iqr),
        ],
        axis=1,
    )
    kcal_cont_df.columns = [
        "percent_kcal_contrib",
        "percent_kcal_contrib_size_adj",
        "median_kcal_size_adj",
        "mean_kcal_size_adj",
        "IQR_kcal_size_adj",
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
        df_prod_ed_reduce, prod_category, 10, sample_size
    )
    num_prods_cat = util_func.number_prods_cat(df_prod_ed, prod_category)

    return (
        metrics_table.merge(df_diversity, on=prod_category, how="left")
        .merge(num_prods_cat, on=prod_category)
        .drop(["count", "energy_density_cat"], axis=1)
    ).set_index(prod_category)


def create_report_table(kcal_density_df: pd.DataFrame, kcal_cont_df: pd.DataFrame):
    """
    Merges kcal density and kcal contribution tables

    Args:
        kcal_density_df (pd.DataFrame): Pandas dataframe with kcal density metrics
        kcal_cont_df (pd.DataFrame): Pandas dataframe with kcal contribution metrics
    Returns:
        pd.DataFrame: Merged table of metrics
    """
    return pd.merge(
        kcal_cont_df, kcal_density_df, left_index=True, right_index=True, how="outer"
    )
