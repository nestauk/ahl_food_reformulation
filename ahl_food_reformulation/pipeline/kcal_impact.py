def kcal_day(purch_recs_comb, pan_conv, panel_weight, scenario):
    """
    Returns summary statistics for a given scenario (no reformulation, high reformulation, low reformulation)

    Parameters
    ----------
    purch_recs_comb : pd.DataFrame
        pandas DataFrame with sum of all kcal purchased by category by an household
    pan_conv : pd.DataFrame
        pandas DataFrame with conversion factor.
    panel_weight : pd.DataFrame
        pandas DataFrame with demographic weights.
    scenario : LIST
        ["Gross_up_kcal", "Gross_up_kcal_max", "Gross_up_kcal_min"]

    Returns
    -------
    pandas Series with descriptive statistics for daily kcal distribution

    """

    pan_conv_weighted = pan_conv.merge(
        panel_weight, left_on="Panel Id", right_on="panel_id", how="inner"
    )
    pan_conv_weighted["conversion"] = (
        pan_conv_weighted["conversion"] * pan_conv_weighted["demographic_weight"]
    )
    pan_conv_weighted = pan_conv_weighted[["Panel Id", "conversion"]]

    hh_kcal = (
        purch_recs_comb.groupby(["Panel Id"])[scenario].sum().reset_index(name=scenario)
    )

    hh_kcal_weighted = hh_kcal.merge(pan_conv_weighted, on="Panel Id", how="inner")

    hh_kcal_weighted[scenario + "_daily"] = (
        hh_kcal_weighted[scenario] / hh_kcal_weighted["conversion"] / 365
    )

    return hh_kcal_weighted[scenario + "_daily"].describe()
