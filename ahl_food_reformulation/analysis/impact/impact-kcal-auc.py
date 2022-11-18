# Read in libraries
from ahl_food_reformulation import PROJECT_DIR
from sklearn import metrics
import pandas as pd

from scipy.integrate import simpson
from numpy import trapz
import numpy as np

import math

# define function to simplify reading large numbers

millnames = ["", " Thousand", " Million", " Billion", " Trillion"]


def millify(n):
    """
    Return an easier to read version of a number

    Parameters
    ----------
    n : float
    number

    Returns
    -------
    string
        number in thousand, million, billion or trillion

    """
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


# Read in tables
diff_avg = pd.read_csv(
    f"{PROJECT_DIR}/outputs/data/decision_table/distribution_reduction_avg.csv"
)
diff_seq = pd.read_csv(
    f"{PROJECT_DIR}/outputs/data/decision_table/distribution_reduction_red.csv"
)


# Separate based on reformulation level
avg_5 = diff_avg[diff_avg["Reformulation Level"] == "5%"].dropna()
avg_10 = diff_avg[diff_avg["Reformulation Level"] == "10%"].dropna()
seq_5 = diff_seq[diff_seq["Reformulation Level"] == "5%"].dropna()
seq_10 = diff_seq[diff_seq["Reformulation Level"] == "10%"].dropna()

# calculate total number of calories removed from population
print("AVERAGE METHOD")
print("-------------")

print(
    "Under 5% kcal reduction the total number of daily calories removed from the population is "
    + millify((avg_5["demographic_weight"] * avg_5["value"]).sum())
)
print(
    "Under 10% kcal reduction the total number of daily calories removed from the population is "
    + millify((avg_10["demographic_weight"] * avg_10["value"]).sum())
)
print("        ")
print("EGGS AND MARGARINE REMOVED")
print("-------------")

print(
    "Under 5% kcal reduction the total number of daily calories removed from the population is "
    + millify((seq_5["demographic_weight"] * seq_5["value"]).sum())
)
print(
    "Under 10% kcal reduction the total number of daily calories removed from the population is "
    + millify((seq_10["demographic_weight"] * seq_10["value"]).sum())
)

# what graphs allow us to read the AUC/AOC as this figure?

# =============================================================================
# IGNORE FOR NOW
# #Generate cumulative distribution of households
#
# # Average
#
# avg_5.sort_values(['value'], inplace=True)
# avg_10.sort_values(['value'], inplace=True)
#
# cdf5_avg = avg_5.groupby(["value"])["demographic_weight"].sum().reset_index()
# cdf5_avg["tot"] = cdf5_avg["demographic_weight"].cumsum()
#
# cdf10_avg = avg_10.groupby(["value"])["demographic_weight"].sum().reset_index()
# cdf10_avg["tot"] = cdf10_avg["demographic_weight"].cumsum()
#
# print(millify(metrics.auc(cdf5_avg["value"].values, cdf5_avg["tot"].values)))
# print(metrics.auc(cdf10_avg["value"].values, cdf10_avg["tot"].values))
#
# # Sequential
#
# seq_5.sort_values(['value'], inplace=True)
# seq_10.sort_values(['value'], inplace=True)
#
#
# cdf5_seq = seq_5.groupby(["value"])["demographic_weight"].sum().reset_index()
# cdf5_seq["tot"] = cdf5_seq["demographic_weight"].cumsum()
#
# cdf10_seq = seq_10.groupby(["value"])["demographic_weight"].sum().reset_index()
# cdf10_seq["tot"] = cdf10_seq["demographic_weight"].cumsum()
#
# print(metrics.auc(cdf5_seq["value"].values, cdf5_seq["tot"].values))
# print(metrics.auc(cdf10_seq["value"].values, cdf10_seq["tot"].values))
#
# # compare
#
# print(metrics.auc(cdf5["value"].values, cdf5["tot"].values) + metrics.auc(cdf10["value"].values, cdf10["tot"].values))
# print(metrics.auc(cdf5_seq["value"].values, cdf5_seq["tot"].values) + metrics.auc(cdf10_seq["value"].values, cdf10_seq["tot"].values))
#
# # stochastic dominance
# # first order: which distribution yields higher return?
# # a distribution dominates another stochastically if cdf is lower (because probability of obtaining lower outcomes is lower)
#
# avg_5["type"] = "avg"
# seq_5["type"] = "seq"
#
# df = pd.concat([avg_5, seq_5])
#
#
# fig5 = alt.Chart(df).mark_line().encode(
#     x = alt.X("value:Q", axis = alt.X(title = "kcal/day")),
#     y=alt.Y("density:Q",axis=alt.Axis(title = "Cumulative Density")),
#     color=alt.Color(
#         "Reformulation Level",
#         scale={"range": list(pu.NESTA_COLOURS[x] for x in [0, 1])}))
#
# fig4b = configure_plots(
#     fig4b,
#     "",
#     "",
#     16,
#     20,
#     16,
# )
#
# save_altair(
#     altair_text_resize(fig4b).properties(width=250, height=250),
#     "cumulative_density_seq",
#     driver=driver,
# )
#
#
# # Testing different methods
# # Sklearn
# print("Avg 5% - Sklearn AUC")
# print(1 - metrics.auc(avg_5["index"].values, avg_5["value"].values))
# # Compute the area using the composite trapezoidal rule.
# area = trapz(avg_5["value"].values, avg_5["index"].values)
# print("Avg 5% - trapezoidal rule")
# print(1 - area)
# # Compute the area using the composite Simpson's rule.
# area = simpson(avg_5["value"].values, avg_5["index"].values)
# print("Avg 5% - Simpson's rule")
# print(1 - area)
#
# # Using AUC
# print("Average method AUC")
# print("--------")
# print("Avg 5%")
# print(
#     1
#     - metrics.auc(avg_5["index"].values, avg_5["value"].values)
# )
# print("Avg 10%")
# print(
#     1
#     - metrics.auc(avg_10["index"].values, avg_10["value"].values)
# )
# print("")
# print("Sequential method AUC")
# print("--------")
# print("Seq 5%")
# print(
#     1
#     - metrics.auc(seq_5["index"].values, seq_5["value"].values)
# )
# print("Seq 10%")
# print(
#     1
#     - metrics.auc(seq_10["index"].values, seq_10["value"].values)
# )
#
# # Compare totals
# print(16.063286895622916 + 30.666541673483657)
# print(15.922631436041161 + 30.764463436386382)
# =============================================================================
