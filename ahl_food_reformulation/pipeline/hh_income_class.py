from ahl_food_reformulation.getters import kantar
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def income_class_share(top: float):
    """
    Runs a logistic regression of hh income dummy on cluster and region dummy and identifies poorest clusters based on input. CLuster classification based on the share method.

    Parameters
    ----------
    top: float
        percentile to identify top x% poorest clusters. E.g. to get the top 20% poorest clusters input 0.2

    Returns
    -------
    pd.DataFrame: Pandas dataframe with a lookup of clusters and low income indicator


    """
    # read demographic file
    demog_clean = kantar.demog_clean()

    # s ubset to variables needed
    demog_sub = demog_clean[["panel_id", "household_income", "region"]]

    # filter to known income
    remove = ["Did not want to answer", "Unknown"]
    demog_sub.query("household_income not in @remove", inplace=True)

    # filter to known region
    demog_sub = demog_sub[demog_sub["region"].notna()].copy()

    # generate low income indicator
    demog_sub["hh_ind"] = np.where(
        (demog_sub["household_income"] == "£0 - £9,999 pa")
        | (demog_sub["household_income"] == "£10,000 - £19,999 pa"),
        1,
        0,
    )

    # merge with cluster assignment (share)
    cl_kcal_share = kantar.cluster_kcal_share()

    # merge cluatser assignment with demographic file
    demog_share = demog_sub.merge(
        cl_kcal_share, left_on="panel_id", right_on="Panel Id"
    )

    # run logistic regression
    log_reg = smf.logit("hh_ind ~ C(clusters, Sum) + C(region)", data=demog_share).fit()

    # extract coefficients and identify top poorest quntiles
    coefs = pd.DataFrame(log_reg.params, columns=["coef"]).reset_index()
    coefs = coefs[coefs["index"].str.contains("clusters")]
    coefs["clusters"] = coefs["index"].str.extract("(\d+)").astype(int)
    coefs["low"] = np.where(coefs["coef"] >= coefs["coef"].quantile(1 - top), 1, 0)

    return coefs[["clusters", "low"]]


def income_class_adj(top: float):
    """
    Runs a logistic regression of hh income dummy on cluster and region dummy and identifies poorest clusters based on input. CLuster classification based on the share method.

    Parameters
    ----------
    top: float
        percentile to identify top x% poorest clusters. E.g. to get the top 20% poorest clusters input 0.2

    Returns
    -------
    pd.DataFrame: Pandas dataframe with a lookup of clusters and low income indicator


    """
    # read demographic file
    demog_clean = kantar.demog_clean()

    # s ubset to variables needed
    demog_sub = demog_clean[["panel_id", "household_income", "region"]]

    # filter to known income
    remove = ["Did not want to answer", "Unknown"]
    demog_sub.query("household_income not in @remove", inplace=True)

    # filter to known region
    demog_sub = demog_sub[demog_sub["region"].notna()].copy()

    # generate low income indicator
    demog_sub["hh_ind"] = np.where(
        (demog_sub["household_income"] == "£0 - £9,999 pa")
        | (demog_sub["household_income"] == "£10,000 - £19,999 pa"),
        1,
        0,
    )

    # merge with cluster assignment (adj)
    cl_adj_size = kantar.cluster_adj_size()

    # merge cluatser assignment with demographic file
    demog_adj = demog_sub.merge(cl_adj_size, left_on="panel_id", right_on="Panel Id")

    # run logistic regression
    log_reg = smf.logit("hh_ind ~ C(clusters, Sum) + C(region)", data=demog_adj).fit()

    # extract coefficients and identify top poorest quntiles
    coefs = pd.DataFrame(log_reg.params, columns=["coef"]).reset_index()
    coefs = coefs[coefs["index"].str.contains("clusters")]
    coefs["clusters"] = coefs["index"].str.extract("(\d+)").astype(int)
    coefs["low"] = np.where(coefs["coef"] >= coefs["coef"].quantile(1 - top), 1, 0)

    return coefs[["clusters", "low"]]
