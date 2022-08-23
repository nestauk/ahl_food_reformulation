# Functions to explore the cluster data

import re
from functools import partial
import random
from toolz import pipe
from typing import Dict, Any, List, Union, Tuple

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import statsmodels.api as sm

from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar


CATEGORY_VARS = [
    "council_tax_band",
    "education_level",
    "ethnicity",
    "household_income",
    "life_stage",
    "region",
    "social_class",
]


def make_demog_clean_names() -> Dict:
    """Make a dictionary of clean names for demographic variables"""

    return {
        re.sub(" ", "_", name.lower()): name
        for name in kantar.household_demog().columns
    }


DEMOG_CLEAN_NAMES = make_demog_clean_names()


def calculate_cluster_shares(table: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Calculates shares of category in a cluster vs. the average
    Args:
        table: the table with household level demographic information + clustering
        variable: variable whose distributions we want to explore

    Returns:
        Table with shares of activity by cluster and normalised by average shares

    """

    average_share = table[variable].value_counts(normalize=True)

    cluster_share = (
        table.groupby("cluster")[variable]
        .value_counts(normalize=True)
        .reset_index(name="share")
        .assign(total_share=lambda df: df[variable].map(average_share))
        .assign(share_norm=lambda df: (df["share"] / df["total_share"]) - 1)
    )

    # return pd.concat([average_share,cluster_share])
    return cluster_share


def plot_cluster_comparison_cat(
    distr: pd.DataFrame,
    var_name: str,
    drop: Union[None, List],
    height: int = 300,
    width: int = 500,
    var_order: Union[None, List] = None,
    pos_text: int = 5,
    clust_n: int = 18,
) -> alt.Chart:
    """Plot distribution in categorical variable by cluster
    Args:
        distr: chart with distributions
        var_name: variable name
        drop: any variables to drop
        height: height of the chart
        width: width of the chart
        var_order: if we want to sort the levels of the categorical variable
        pos_text: how many "outlying" clusters do we want to label
    """

    distr_ = (
        distr.groupby(var_name)
        .apply(
            lambda df: df.assign(
                text=lambda df_2: [
                    cl if (rank < pos_text) | (rank > clust_n - pos_text) else ""
                    for cl, rank in zip(df_2["cluster"], df_2["share_norm"].rank())
                ]
            )
        )
        .reset_index(drop=False)
    )

    if drop is not None:
        distr_ = distr_.loc[~distr_[var_name].isin(drop)]

    if var_order is None:
        var_order = list(distr_[var_name].unique())

    comp = (
        alt.Chart(distr_)
        .mark_point(filled=True, size=50, stroke="black", strokeWidth=0.5)
        .encode(
            y=alt.Y(f"{var_name}:O", title=DEMOG_CLEAN_NAMES[var_name], sort=var_order),
            x=alt.X(
                "share_norm", title="Share vs. population", axis=alt.Axis(format="%")
            ),
            tooltip=["cluster"],
            color=alt.Color(
                "cluster:N", scale=alt.Scale(scheme="tableau20"), title="Cluster"
            ),
        )
    )

    txt = (
        alt.Chart(distr_)
        .mark_text(dy=-10)
        .encode(
            y=alt.Y(f"{var_name}:O", title=DEMOG_CLEAN_NAMES[var_name], sort=var_order),
            x=alt.X(
                "share_norm", title="Share vs. population", axis=alt.Axis(format="%")
            ),
            tooltip=["cluster"],
            text="text",
            color=alt.Color(
                "cluster:N", scale=alt.Scale(scheme="tableau20"), title="Cluster"
            ),
        )
    )

    line = (
        alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[1, 1]).encode(x="x")
    )

    return (comp + txt + line).properties(height=height, width=width)


def plot_cluster_comparison_non_cat(
    table: pd.DataFrame, var_name: str, n_cols: int = 5
) -> alt.Chart:
    """Plots cluster comparison for non-categorical variables
    Args:
        table: table with household level demographic information + clustering
        var_name: variable name

    """

    distr = (
        table.groupby(["cluster"])[var_name]
        .value_counts(normalize=True)
        .reset_index(name="share")
        .merge(
            table[var_name].value_counts(normalize=True).reset_index(name="avg_share"),
            how="left",
            left_on=var_name,
            right_on="index",
        )
    )

    clust_chart = (
        alt.Chart()
        .mark_line(point=True)
        .encode(
            x=f"{var_name}",
            y=alt.Y("share", axis=alt.Axis(format="%"), title="Household %"),
            color=alt.Color("cluster:N", scale=alt.Scale(scheme="tableau20")),
        )
        .properties(width=100, height=100)
    )

    avg_chart = (
        alt.Chart()
        .mark_line(point=False, color="black", strokeWidth=1)
        .encode(x=f"{var_name}", y=alt.Y("avg_share", axis=alt.Axis(format="%")))
    )

    return alt.layer(clust_chart, avg_chart, data=distr).facet(
        facet=alt.Facet("cluster:N"), columns=n_cols
    )


def make_modelling_dataset(
    demog_table: pd.DataFrame,
) -> Tuple[np.array, np.array, np.array, np.array, pd.DataFrame, np.array]:
    """Creates datasets ready for modelling
    Args:
        demog_table: household characteristics table
    """

    demog_table_no_missing = demog_table.dropna(axis=0, subset=["cluster"])

    num_vars = [
        "main_shopper_age",
        "household_size",
        "number_of_children",
        "high_bmi",
        "bmi_missing",
    ]

    demog_table_model = pd.concat(
        [demog_table_no_missing[num_vars]]
        + [
            pd.get_dummies(demog_table_no_missing[var], prefix=var, drop_first=True)
            for var in CATEGORY_VARS
        ],
        axis=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        demog_table_model.values,
        pd.get_dummies(demog_table_no_missing["cluster"]).values,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        demog_table_model,
        demog_table_no_missing["cluster"].values,
    )


def simple_grid_search(
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
    y_test: np.array,
    C_values: np.array,
) -> None:
    """Fits and evaluates a model with different C scores

    Args:
        training and test data,
        C_values: list of values for the regularisation parameter
    """

    f1_score_micro = []

    for c in C_values:

        print(f"training model with C={c}")

        mod = OneVsRestClassifier(
            LogisticRegression(
                C=c, multi_class="ovr", solver="liblinear", class_weight="balanced"
            )
        )
        fit_mod = mod.fit(X_train, y_train)

        f1_score_micro.append(
            f1_score(fit_mod.predict(X_test), y_test, average="weighted")
        )

    pd.Series(f1_score_micro, index=C_values).plot()


def get_regression_coefficients(
    features: pd.DataFrame, targets: np.array, C: float, top_keep: int = 10
) -> pd.DataFrame:
    """Extracts regression coefficients for an instantiation of the model

    Args:
        features, targets: data for the model
        C: regularisation parameter
        top_keep: top coefficients for each cluster

    """

    number_columns = len(features.columns)

    final_model = OneVsRestClassifier(
        LogisticRegression(
            C=C, multi_class="ovr", solver="liblinear", class_weight="balanced"
        )
    )

    final_model.fit(features.values, pd.get_dummies(targets))

    regression_coefficients = {}

    for n, mod in enumerate(final_model.estimators_):
        labelled_coefs = [
            (name, value) for name, value in zip(features.columns, mod.coef_[0])
        ]
        regression_coefficients[n] = labelled_coefs

    return pd.concat(
        [
            pd.DataFrame(values, columns=["variable", "coefficient"])
            .assign(cluster=k)
            .assign(
                coefficient_top=lambda df: [
                    x if (r <= top_keep) | (r > number_columns - top_keep) else np.nan
                    for x, r in zip(df["coefficient"], df["coefficient"].rank())
                ]
            )
            for k, values in regression_coefficients.items()
        ]
    )


def plot_regression_coeffs(reg_coeff: pd.DataFrame) -> alt.Chart:
    """Heatmap with coefficients"""

    return (
        alt.Chart(reg_coeff)
        .mark_rect(stroke="grey", strokeWidth=0.5)
        .encode(
            x=alt.X("variable", title=None),
            y=alt.Y("cluster:O", title="Cluster"),
            tooltip=["cluster", "variable", alt.Tooltip("coefficient", format=".3f")],
            color=alt.Color(
                "coefficient_top",
                scale=alt.Scale(scheme="redblue"),
                sort="descending",
                title=["Regression", "coefficient"],
            ),
        )
    )


def plot_demog_pipeline(
    demog: pd.DataFrame,
    comp_variable: str,
    order_vars: List,
    drops: List = ["Unknown", "Did not want to answer", "Other"],
    clust_n: int = 18,
) -> alt.Chart:
    """Pipeline to generate demographic plots
    Args:
        demog: household characteristics table
        comp_variable: variable to compare across clusters
        order_vars: order of categories for the variable
        drops: categories to drop

    Returns:
        A plot comparing clusters along a demographic variable

    """
    return pipe(
        demog,
        partial(calculate_cluster_shares, variable=comp_variable),
        partial(
            plot_cluster_comparison_cat,
            drop=drops,
            pos_text=3,
            clust_n=clust_n,
            var_order=order_vars,
            var_name=comp_variable,
        ),
    )


def make_salience_table(
    coefficients_df: pd.DataFrame,
    coeff_name: str = "coefficient",
    comp_var: str = "variable",
    top_n: int = 10,
) -> pd.DataFrame:
    """Create table with top regression scores

    Args:
        coefficients_df: dataframe with regression coefficients
        coeff_name: name of the column with the coefficients
        comp_var: name of the column with the variable
        top_n: number of top coefficients to keep

    Returns:
        A dataframe with the top coefficients
    """
    return (
        coefficients_df.groupby("cluster")
        .apply(
            lambda df: (
                df.assign(
                    high_predictive=lambda df: ", ".join(
                        df.sort_values(coeff_name, ascending=False).head(n=top_n)[
                            comp_var
                        ]
                    )
                )
                .assign(
                    low_predictive=lambda df: ", ".join(
                        df.sort_values(coeff_name, ascending=True).head(n=top_n)[
                            comp_var
                        ]
                    )
                )[["cluster", "high_predictive", "low_predictive"]]
                .drop_duplicates("cluster")
            )
        )
        .reset_index(drop=True)
    )


def item_share(purchase_table: pd.DataFrame, category: str) -> pd.Series:
    """Share of items in a category"""

    return (
        purchase_table.groupby(category).apply(lambda df: df["volume"].sum())
        / purchase_table["volume"].sum()
    )


def plot_item_share(distribution: pd.Series, category: str) -> alt.Chart:
    """Plot share of items in a category"""

    return (
        alt.Chart(distribution.reset_index(name="share"))
        .mark_line(point=True, stroke="red")
        .encode(
            y=alt.Y(category, sort=alt.EncodingSortField("share", order="descending")),
            x=alt.X("share", title="Share of volume", axis=alt.Axis(format="%")),
        )
    )


def make_purchase_shares_normalised(
    purchase_records: pd.DataFrame, category: str, top_n: int = 5, num_clust: int = 18
) -> pd.DataFrame:
    """Creates normalised purchase shares
    Args:
        purchase_records: dataframe with purchase records
        category: category to group by
        top_n: number of top categories to keep
        num_clust: number of clusters

    Returns:
        Table with normalised purchase shares
    """

    purchase_shares_cluster = purchase_records.groupby(["clust"]).apply(
        lambda df: (
            df.groupby(category)["volume"].sum() / df["volume"].sum()
        ).sort_values(ascending=False)
    )

    return (
        purchase_shares_cluster.unstack(level=1)
        .fillna(0)
        .apply(lambda col: zscore(col))
        .stack()
        .reset_index(name="share_normalised")
        .groupby(category)  # This is to extract labels for outlierish clusters
        .apply(
            lambda df: df.assign(
                add_label=[
                    str(text) if (rank <= top_n) | (rank > num_clust - top_n) else " "
                    for text, rank in zip(df["clust"], df["share_normalised"].rank())
                ]
            )
        )
        .reset_index(drop=False)
    )


def plot_shares_normalised(
    purchase_normalised: pd.DataFrame, category: str, width: int = 700
) -> alt.Chart:
    """Plots nornalised shares
    Args:
        purchase_normalised: dataframe with normalised purchase shares
        category: name of the category variable
        width: width of the plot

    Returns:
        Altair chart with normalised purchase shares
    """

    ch = (
        alt.Chart(purchase_normalised)
        .mark_point(filled=True, size=50, stroke="black", strokeWidth=0.5)
        .encode(
            y=category,
            x=alt.X("share_normalised", title="Share of volume, normalised"),
            tooltip=["clust"],
            color=alt.Color("clust:O", scale=alt.Scale(scheme="tableau20")),
        )
    )

    # This shows labels for the top categories
    txt = (
        alt.Chart(purchase_normalised)
        .mark_text(dx=10, dy=-4)
        .encode(
            y=category,
            x=alt.X("share_normalised", title="Share of volume, normalised"),
            tooltip=["clust"],
            text="add_label",
            color=alt.Color("clust:O", scale=alt.Scale(scheme="tableau20")),
        )
    )

    rul = (
        alt.Chart(pd.DataFrame({"x": [1]}))
        .mark_rule(strokeDash=[2, 2], stroke="red")
        .encode(x="x")
    )

    return (ch + rul + txt).properties(width=width)


def make_regression_dataset(
    purchase_records: pd.DataFrame,
    clust_lu: Dict,
    demog_table: pd.DataFrame,
    category: str,
    sample_size: int = 5000,
) -> Tuple[pd.DataFrame]:
    """Creates a regression dataset

    Args:
        purchase_records: dataframe with purchase records
        clust_lu: dictionary with cluster labels
        demog_table: dataframe with demographic data
        category: category to group by
        sample_size: sample size to use

    Returns:
        A tuple with the targets (consumption) and the features (cluster + household size control)

    """

    house_sample = random.sample(
        purchase_records.dropna(axis=0, subset=["clust"])["panel_id"].unique().tolist(),
        sample_size,
    )

    purchase_records_sample = purchase_records.loc[
        purchase_records["panel_id"].isin(house_sample)
    ].reset_index(drop=True)

    print(len(purchase_records_sample))

    househ_shares_target = (
        purchase_records_sample.groupby(["panel_id"])
        .apply(lambda df: df.groupby(category)["volume"].sum() / df["volume"].sum())
        .unstack(level=1)
        .fillna(0)
        .apply(lambda x: zscore(x))
    )

    househ_features = (
        househ_shares_target.reset_index(drop=False)["panel_id"]
        .to_frame()
        .assign(cluster=lambda df: df["panel_id"].map(clust_lu))
        .assign(
            hous_size=lambda df: df["panel_id"].map(
                demog_table.set_index("panel_id")["household_size"].to_dict()
            )
        )
    )

    return househ_shares_target, househ_features


def fit_purchase_regression(
    target: pd.DataFrame, features: pd.DataFrame, category
) -> pd.DataFrame:
    """Regresses share of purchases on household basket on cluster + household size"""

    coeffs = []
    print("Regressing...")
    for cat in target.columns:
        print("\t" + cat)

        for clust in features["cluster"].unique():

            endog = target[cat].to_numpy()
            exog = sm.add_constant(
                features.drop(axis=1, labels=["panel_id"]).assign(
                    cluster=lambda df: (df["cluster"] == clust).astype(int)
                )
            ).to_numpy()

            mod = sm.OLS(endog, exog)
            out = mod.fit()
            coeffs.append([cat, clust, out.params[1], out.pvalues[1]])

    return pd.DataFrame(coeffs, columns=[category, "cluster", "coefficient", "p_value"])


def plot_regression_result(
    regression_table: pd.DataFrame,
    pvalue_thres: float = 0.05,
    category: str = "rst_4_market_sector",
) -> alt.Chart:
    """Plots regression results"""

    return (
        alt.Chart(regression_table.query(f"p_value<{pvalue_thres}"))
        .mark_rect(stroke="darkgrey", strokeWidth=0.5)
        .encode(
            y=category,
            x="cluster:O",
            tooltip=[category, "cluster", alt.Tooltip("coefficient", format=".3f")],
            color=alt.Color(
                "coefficient",
                scale=alt.Scale(scheme="redblue", domainMid=0),
                sort="descending",
                title="Regression coefficient",
            ),
        )
    )


def plot_cluster_counts(panel_clusters: pd.DataFrame, panel_weights: pd.DataFrame):

    """Plots weighted and un-weighted count of households per clusters

    Args:
        panel_clusters: dataframe with household id's and cluster assigned
        panel_weights: weights per household and period

    Returns:
        Plots saved to outputs/figures/png

    """

    cluster_weights = panel_clusters.merge(
        panel_weights, left_on="Panel Id", right_on="panel_id", how="left"
    )
    cluster_weights.drop(["panel_id"], axis=1, inplace=True)
    cluster_weights_size = (
        cluster_weights.groupby(["clusters"])["demographic_weight"].sum().reset_index()
    )

    # Plot total households per cluster (unweighted)
    sns.set(rc={"figure.figsize": (20, 3)}, style="white")
    cluster_counts = (
        panel_clusters["clusters"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Cluster", "clusters": "Households"})
    )
    ax = sns.barplot(
        data=cluster_counts, x="Cluster", y="Households", color="lightblue"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="both", which="major", labelsize=20)
    plt.ylabel("Households", fontsize=25)
    plt.xlabel("Cluster", fontsize=25)
    plt.title("Total households in each cluster (unweighted)", fontsize=25)
    plt.savefig(
        f"{PROJECT_DIR}/outputs/figures/png/unweighted_hh_clusters.png",
        bbox_inches="tight",
    )

    # Plot total households per cluster (weighted)
    sns.set(rc={"figure.figsize": (20, 3)}, style="white")
    ax = sns.barplot(
        data=cluster_weights_size,
        x="clusters",
        y="demographic_weight",
        color="lightblue",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="both", which="major", labelsize=20)
    plt.ylabel("Households", fontsize=25)
    plt.xlabel("Cluster", fontsize=25)
    plt.title("Total households in each cluster (weighted)", fontsize=25)
    plt.savefig(
        f"{PROJECT_DIR}/outputs/figures/png/weighted_hh_clusters.png",
        bbox_inches="tight",
    )

    # Get percentage per cluster
    cluster_counts["households_unweighted"] = (
        cluster_counts["Households"] / cluster_counts["Households"].sum()
    ) * 100
    cluster_weights_size["households_weighted"] = (
        cluster_weights_size["demographic_weight"]
        / cluster_weights_size["demographic_weight"].sum()
    ) * 100

    # Plot percentage of households per cluster - weighted vs unweighted
    cluster_weights_size.merge(
        cluster_counts, left_on="clusters", right_on="Cluster", how="left"
    )[["clusters", "households_weighted", "households_unweighted"]].set_index(
        "clusters"
    ).plot(
        kind="barh", figsize=(10, 10)
    )
    plt.title(
        "Proportion of households in each cluster - weighted compared to un-weighted",
        fontsize=16,
        pad=15,
    )
    plt.xlabel("Percentage", fontsize=14)
    plt.ylabel("Cluster", fontsize=14)
    plt.savefig(
        f"{PROJECT_DIR}/outputs/figures/png/percent_hh_clusters.png",
        bbox_inches="tight",
    )
