# Functions to explore the cluster data

import re
from typing import Dict, Any, List, Union, Tuple

import altair as alt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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
                    cl if (rank < pos_text) | (rank > 20 - pos_text) else ""
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
                    x if (r < top_keep) | (r > number_columns - top_keep) else np.nan
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
