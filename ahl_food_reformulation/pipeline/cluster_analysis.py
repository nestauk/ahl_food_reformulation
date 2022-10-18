from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.pipeline import hh_income_class
import pandas as pd
from patsy.contrasts import Sum
import statsmodels.api as sm
import patsy
from ahl_food_reformulation.pipeline import cluster_analysis as cl
import logging


def mk_reg_df_share(
    val_fields: pd.DataFrame,
    pur_recs: pd.DataFrame,
    prod_codes: pd.DataFrame,
    prod_vals: pd.DataFrame,
    nut_rec: pd.DataFrame,
    prod_meta: pd.DataFrame,
    att_num: int,
):
    """
    Generates dataset to use for regression analysis. It is a dataset in wide format where each row is a hh and each column the share of kcal consumed by that household for a category
    Args:
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut (pd.DataFrame): Pandas dataframe of purchase level nutritional information
        prod_codes (pd.DataFrame): Pandas dataframe contains the codes to link products to category information
        prod_vals (pd.DataFrame): Pandas dataframe contains the product category information
        att_num (int): Product category type code number

    Returns
    -------
    purch_recs_wide_share (pd.DataFrame): Share of kcal consumed by hh by categoty

    """
    # Purchase and product info combined
    comb_files = transform.combine_files(
        val_fields, pur_recs, prod_codes, prod_vals, att_num
    ).drop("att_vol", axis=1)

    comb_update = comb_files.merge(
        prod_meta[["product_code", "rst_4_extended", "rst_4_market_sector"]],
        left_on="Product Code",
        right_on="product_code",
    )

    if att_num == 2907:
        comb_update.rename(columns={"rst_4_extended": "att_vol"}, inplace=True)
    elif att_num == 2828:
        comb_update.rename(columns={"rst_4_market_sector": "att_vol"}, inplace=True)

    purch_recs_comb = transform.make_purch_records(nut_rec, comb_update, ["att_vol"])

    # for each hh and category create share of kcal
    purch_recs_comb["share"] = purch_recs_comb["Energy KCal"] / purch_recs_comb.groupby(
        ["Panel Id"]
    )["Energy KCal"].transform("sum")

    # clean category names
    purch_recs_comb["att_vol"] = (
        purch_recs_comb["att_vol"]
        .str.replace(" ", "", regex=True)
        .str.replace("/", "", regex=True)
        .str.replace("-", "", regex=True)
        .str.replace("1", "One", regex=True)
        .str.replace("2", "Two", regex=True)
        .str.replace("+", "", regex=True)
        .str.replace("&", "", regex=True)
        .str.replace(".", "", regex=True)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
    )

    # create file with share for regression

    purch_recs_wide_share = (
        purch_recs_comb.set_index(["Panel Id", "att_vol"])[["share"]]
        .unstack(["att_vol"])
        .fillna(0)
    )
    purch_recs_wide_share.columns = purch_recs_wide_share.columns.droplevel()

    return purch_recs_wide_share


def mk_reg_df_adj(
    pan_ind: pd.DataFrame,
    val_fields: pd.DataFrame,
    pur_recs: pd.DataFrame,
    prod_codes: pd.DataFrame,
    prod_vals: pd.DataFrame,
    nut_rec: pd.DataFrame,
    prod_meta: pd.DataFrame,
    att_num: int,
):
    """
    Generates dataset to use for regression analysis. It is a dataset in wide format where each row is a hh and each column the absolute value of kcal consumed by that household for a category


    Parameters
    ----------
    pan_ind (pd.DataFrame): Pandas dataframe of household members
    nut (pd.DataFrame): Pandas dataframe of purchase level nutritional information
    val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
    pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
    prod_codes (pd.DataFrame): Pandas dataframe contains the codes to link products to category information
    prod_vals (pd.DataFrame): Pandas dataframe contains the product category information
    att_num (int): Product category type code number

    Returns
    -------
    purch_recs_wide_abs (pd.DataFrame): Absolute adjusted kcal consumed by hh by categoty

    """

    # Converted household size
    pan_conv = transform.hh_size_conv(pan_ind)

    # Purchase and product info combined
    comb_files = transform.combine_files(
        val_fields, pur_recs, prod_codes, prod_vals, att_num
    ).drop("att_vol", axis=1)

    comb_update = comb_files.merge(
        prod_meta[["product_code", "rst_4_extended", "rst_4_market_sector"]],
        left_on="Product Code",
        right_on="product_code",
    )

    if att_num == 2907:
        comb_update.rename(columns={"rst_4_extended": "att_vol"}, inplace=True)
    elif att_num == 2828:
        comb_update.rename(columns={"rst_4_market_sector": "att_vol"}, inplace=True)

    purch_recs_comb = transform.make_purch_records(
        nut_rec, comb_update, ["att_vol"]
    ).merge(pan_conv, on="Panel Id")

    # for each hh and category create adjusted kcal
    purch_recs_comb["abs_adj"] = (
        purch_recs_comb["Energy KCal"] / purch_recs_comb["conversion"]
    )

    # clean category names
    purch_recs_comb["att_vol"] = (
        purch_recs_comb["att_vol"]
        .str.replace(" ", "", regex=True)
        .str.replace("/", "", regex=True)
        .str.replace("-", "", regex=True)
        .str.replace("1", "One", regex=True)
        .str.replace("2", "Two", regex=True)
        .str.replace("+", "", regex=True)
        .str.replace("&", "", regex=True)
        .str.replace(".", "", regex=True)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
    )

    # create file with absolute adjusted values for regression

    purch_recs_wide_abs = (
        purch_recs_comb.set_index(["Panel Id", "att_vol"])[["abs_adj"]]
        .unstack(["att_vol"])
        .fillna(0)
    )
    purch_recs_wide_abs.columns = purch_recs_wide_abs.columns.droplevel()

    return purch_recs_wide_abs


def reg_share(
    cl_kcal_share: pd.DataFrame,
    panel_weight: pd.DataFrame,
    purch_recs_wide_share: pd.DataFrame,
    sig_level: float,
    top: float,
):
    """
    Runs multiple regressions using the deviant coding method with share of kcal for a category as dependent variable and cluster assignment dummies as independent variables. Retains only coefficients that are significant and positive.

    Parameters
    ----------
    cl_kcal_share : pd.DataFrame
        cluster assignment based on share of kcal.
    panel_weight : pd.DataFrame
        demographic weights.
    purch_recs_wide_share : pd.DataFrame
        Share of kcal consumed by hh by categoty.
    sig_level : float
        significance level to retain coefficients (e.g 0.05 for 5%).
    top : float
        percentile to identify top x% poorest clusters. E.g. to get the top 20% poorest clusters input 0.2

    Returns
    -------
    final: pd.DataFrame
        Pandas dataframe aggregated by category with the number of clusters that consume significantly more of a category, the population in the clusters and the percenatge of population it represents

    """

    # merge panel weights with cluster assignments
    panel_w = panel_weight.merge(cl_kcal_share, left_on="panel_id", right_on="Panel Id")

    # calculate total population size
    pop_size = panel_weight["demographic_weight"].sum()

    # calculate total population size within clusters
    cluster_w = (
        panel_w.groupby(["clusters"])["demographic_weight"]
        .sum()
        .reset_index(name="tot")
    )

    # merge cluster assignment into regression file
    kcal_share = purch_recs_wide_share.merge(cl_kcal_share, on="Panel Id")

    # list of categories
    val = purch_recs_wide_share.columns.tolist()

    # run regressions
    coeffs = []
    for cat in val:
        y, X = patsy.dmatrices(
            str(cat) + " ~ C(clusters, Sum)", kcal_share, return_type="matrix"
        )
        mod = sm.OLS(y, X)
        res = mod.fit()
        coeffs.append([cat, mod.data.xnames, res.params, res.pvalues])

    # flatten list of lists into dataframe
    flat = pd.DataFrame(
        coeffs, columns=["category", "clusters", "coeffs", "pvalues"]
    ).explode(["clusters", "coeffs", "pvalues"], ignore_index=True)

    # remove the intercept coefficient
    flat.query("clusters != 'Intercept'", inplace=True)

    # retain only positive and statistiucally significant coefficients (5%)
    out = flat[(flat.pvalues < sig_level) & (flat.coeffs > 0)]

    # create cluster variable
    out = out.assign(clusters=out["clusters"].str.extract("(\d+)").astype(int))

    # merge in weights
    out = out.merge(cluster_w, on="clusters")

    # read in income classifier
    income_class = hh_income_class.income_class_share(top)

    # merge with income classifier
    out_income = out.merge(income_class, on="clusters")

    # create final dataset by merging aggregate
    final = pd.merge(
        pd.merge(
            out.groupby(["category"])["clusters"].size().reset_index(name="cluster_no"),
            out.groupby(["category"])["tot"].sum().reset_index(name="pop"),
            on="category",
        ),
        out_income.groupby(["category"])["low"]
        .sum()
        .reset_index(name="cluster_low_income"),
        on="category",
    )

    # share wrt to population
    final["share"] = final["pop"] / pop_size

    return final


def reg_adj(
    cl_adj_size: pd.DataFrame,
    panel_weight: pd.DataFrame,
    purch_recs_wide_abs: pd.DataFrame,
    sig_level: float,
    top: float,
):
    """
    Runs multiple regressions using the deviant coding method with absolute adjusted kcal for a category as dependent variable and cluster assignment dummies as independent variables. Retains only coefficients that are significant and positive.

    Parameters
    ----------
    cl_adj_size : pd.DataFrame
        cluster assignment based on adjusted kcal
    panel_weight : pd.DataFrame
        demographic weights.
    purch_recs_wide_abs : pd.DataFrame
        Absolute adjusted kcal consumed by hh by categoty.
    sig_level : float
        significance level to retain coefficients (e.g 0.05 for 5%).
    top : float
        percentile to identify top x% poorest clusters. E.g. to get the top 20% poorest clusters input 0.2


    Returns
    -------
    final: pd.DataFrame
        Pandas dataframe aggregated by category with the number of clusters that consume significantly more of a category, the population in the clusters and the percenatge of population it represents

    """

    # merge panel weights with cluster assignments
    panel_w = panel_weight.merge(cl_adj_size, left_on="panel_id", right_on="Panel Id")

    # calculate total population size
    pop_size = panel_weight["demographic_weight"].sum()

    # calculate total population size within clusters
    cluster_w = (
        panel_w.groupby(["clusters"])["demographic_weight"]
        .sum()
        .reset_index(name="tot")
    )

    # merge cluster assignment into regression file
    kcal_adj = purch_recs_wide_abs.merge(cl_adj_size, on="Panel Id")

    # list of categories
    val = purch_recs_wide_abs.columns.tolist()

    # run regressions
    coeffs = []
    for cat in val:
        y, X = patsy.dmatrices(
            str(cat) + " ~ C(clusters, Sum)", kcal_adj, return_type="matrix"
        )
        mod = sm.OLS(y, X)
        res = mod.fit()
        coeffs.append([cat, mod.data.xnames, res.params, res.pvalues])

    # flatten list of lists into dataframe
    flat = pd.DataFrame(
        coeffs, columns=["category", "clusters", "coeffs", "pvalues"]
    ).explode(["clusters", "coeffs", "pvalues"], ignore_index=True)

    # remove the intercept coefficient
    flat.query("clusters != 'Intercept'", inplace=True)

    # retain only positive and statistiucally significant coefficients (5%)
    out = flat[(flat.pvalues < sig_level) & (flat.coeffs > 0)]

    # create cluster variable
    out = out.assign(clusters=out["clusters"].str.extract("(\d+)").astype(int))

    # merge in weights
    out = out.merge(cluster_w, on="clusters")

    # income classifier
    income_class = hh_income_class.income_class_adj(top)

    # merge in classifier
    out_income = out.merge(income_class, on="clusters")

    # create final dataset by merging aggregate
    final = pd.merge(
        pd.merge(
            out.groupby(["category"])["clusters"].size().reset_index(name="cluster_no"),
            out.groupby(["category"])["tot"].sum().reset_index(name="pop"),
            on="category",
        ),
        out_income.groupby(["category"])["low"]
        .sum()
        .reset_index(name="cluster_low_income"),
        on="category",
    )

    # share wrt to population
    final["share"] = final["pop"] / pop_size

    return final


def cluster_table(
    val_fields: pd.DataFrame,
    pur_recs: pd.DataFrame,
    prod_codes: pd.DataFrame,
    prod_vals: pd.DataFrame,
    nut_rec: pd.DataFrame,
    prod_meta: pd.DataFrame,
    panel_weight: pd.DataFrame,
    cl_kcal_share: pd.DataFrame,
    cl_adj_size: pd.DataFrame,
    pan_ind: pd.DataFrame,
    att_num: int,
    sig_level: float,
    top: float,
):
    """

    Generates data, run regression and merges final outputs

    Parameters
    ----------
    val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
    pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
    prod_codes (pd.DataFrame): Pandas dataframe contains the codes to link products to category information
    prod_vals (pd.DataFrame): Pandas dataframe contains the product category information
    nut_rec (pd.DataFrame): Pandas dataframe of purchase level nutritional information
    prod_meta (pd.DataFrame) : Pandas data frame of products with categories
    att_num (int): Product category type code number
    sig_level (float): significance level to retain coefficients (e.g 0.05 for 5%).
    top (float): percentile to identify top x% poorest clusters. E.g. to get the top 20% poorest clusters input 0.2
    panel_weight (pd.DataFrame) demographic weights.
    purch_recs_wide_share (pd.DataFrame) Share of kcal consumed by hh by categoty.
    cl_kcal_share (pd.DataFrame) cluster assignment based on share of kcal.

    Returns
    -------
    pd.DataFrame aggregate table by category

    """
    logging.info("Generating regression data")

    purch_recs_wide_share = cl.mk_reg_df_share(
        val_fields,
        pur_recs,
        prod_codes,
        prod_vals,
        nut_rec,
        prod_meta,
        att_num,
    )

    purch_recs_wide_abs = cl.mk_reg_df_adj(
        pan_ind,
        val_fields,
        pur_recs,
        prod_codes,
        prod_vals,
        nut_rec,
        prod_meta,
        att_num,
    )

    logging.info("Running regression")

    share_table = cl.reg_share(
        cl_kcal_share, panel_weight, purch_recs_wide_share, sig_level, top
    )[["category", "share", "cluster_no", "cluster_low_income"]]

    adj_table = cl.reg_adj(
        cl_adj_size, panel_weight, purch_recs_wide_abs, sig_level, top
    )[["category", "share", "cluster_no", "cluster_low_income"]]

    if att_num == 2907:
        cat = "rst_4_extended"
    elif att_num == 2828:
        cat = "rst_4_market_sector"

    logging.info("Mergin tables")

    tbl = pd.DataFrame(prod_meta[cat].unique(), columns=[cat])

    tbl["att_vol"] = (
        tbl[cat]
        .str.replace(" ", "", regex=True)
        .str.replace("/", "", regex=True)
        .str.replace("-", "", regex=True)
        .str.replace("1", "One", regex=True)
        .str.replace("2", "Two", regex=True)
        .str.replace("+", "", regex=True)
        .str.replace("&", "", regex=True)
        .str.replace(".", "", regex=True)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
    )

    adj_table.rename(
        columns={
            "category": "att_vol",
            "share": "share_adj",
            "cluster_no": "cluster_no_adj",
            "cluster_low_income": "cluster_low_adj",
        },
        inplace=True,
    )

    share_table.rename(
        columns={
            "category": "att_vol",
            "share": "share_sh",
            "cluster_no": "cluster_no_sh",
            "cluster_low_income": "cluster_low_sh",
        },
        inplace=True,
    )

    return (
        tbl.merge(share_table, on="att_vol", how="left")
        .merge(adj_table, on="att_vol", how="left")
        .drop("att_vol", axis=1)
    )
