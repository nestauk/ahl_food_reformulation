from ahl_food_reformulation.utils import util_functions as util_func
from ahl_food_reformulation.pipeline import energy_density as energy
from ahl_food_reformulation.pipeline import nutrient_metrics_funcs as nutrient


def create_pop_table_cat(
    prod_category, val_fields, prod_mast, uom, pur_recs, nut_recs, prod_meta, prod_meas
):
    """
    Create population metrics table based on chosen category

    Args:
        prod_category (str): one product category
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        prod_mast (pd.DataFrame): Pandas dataframe unique product list
        uom (pd.DataFrame): Pandas dataframe contains product measurement information
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
        prod_meas (pd.DataFrame): Pandas dataframe with additional conversions to g and ml for unit and serving products
    Returns:
        pd.DataFrame: Table with metrics based on energy density scores per category
    """
    df_prod_ed = energy.prod_energy_100(
        prod_category,
        val_fields,
        prod_mast,
        uom,
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

    df_prod_ed_reduce = util_func.reduce_df(df_prod_ed, 50, prod_category)[
        [prod_category, "chosen_unit", "kcal_100g_ml"]
    ]
    df_diversity = nutrient.create_diversity_df(
        df_prod_ed_reduce, prod_category, 10, 50
    )
    num_prods_cat = util_func.number_prods_cat(df_prod_ed, prod_category)

    return (
        metrics_table.merge(df_diversity, on=prod_category, how="left")
        .merge(num_prods_cat, on=prod_category)
        .drop(["count"], axis=1)
    )
