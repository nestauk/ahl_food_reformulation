# Read in libraries
from ahl_food_reformulation import PROJECT_DIR
import pandas as pd
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
    f"{PROJECT_DIR}/outputs/data/decision_table/distribution_reduction_top_10.csv"
)


# Separate based on reformulation level
avg_5 = diff_avg[diff_avg["Reformulation Level"] == "5%"].dropna()
avg_10 = diff_avg[diff_avg["Reformulation Level"] == "10%"].dropna()


# calculate total number of calories removed from population
print("TOP 10 CATEGORIES")
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
