# Read in libraries
from ahl_food_reformulation import PROJECT_DIR
from sklearn import metrics
import pandas as pd

from scipy.integrate import simpson
from numpy import trapz
import numpy as np

# Read in tables
diff_avg = pd.read_csv(
    f"{PROJECT_DIR}/outputs/data/decision_table/distribution_reduction_avg.csv"
)
diff_seq = pd.read_csv(
    f"{PROJECT_DIR}/outputs/data/decision_table/distribution_reduction_seq.csv"
)

# Create inputs for AUC
avg_5 = diff_avg[diff_avg["Reformulation Level"] == "5%"]
avg_10 = diff_avg[diff_avg["Reformulation Level"] == "10%"]
seq_5 = diff_seq[diff_seq["Reformulation Level"] == "5%"]
seq_10 = diff_seq[diff_seq["Reformulation Level"] == "10%"]


square_x = avg_5["index"].values
square_y = [x for x in range(-70, 1, int(70 / 10))]

# Testing different methods
# Sklearn
print("Avg 5% - Sklearn AUC")
print(1 - metrics.auc(avg_5["index"].values, avg_5["value"].values))
# Compute the area using the composite trapezoidal rule.
area = trapz(avg_5["value"].values, avg_5["index"].values)
print("Avg 5% - trapezoidal rule")
print(1 - area)
# Compute the area using the composite Simpson's rule.
area = simpson(avg_5["value"].values, avg_5["index"].values)
print("Avg 5% - Simpson's rule")
print(1 - area)

# Using AUC
print("Average method AUC")
print("--------")
print("Avg 5%")
print(
    metrics.auc(square_x, square_y)
    - metrics.auc(avg_5["index"].values, avg_5["value"].values)
)
print("Avg 10%")
print(
    metrics.auc(square_x, square_y)
    - metrics.auc(avg_10["index"].values, avg_10["value"].values)
)
print("")
print("Sequential method AUC")
print("--------")
print("Seq 5%")
print(
    metrics.auc(square_x, square_y)
    - metrics.auc(seq_5["index"].values, seq_5["value"].values)
)
print("Seq 10%")
print(
    metrics.auc(square_x, square_y)
    - metrics.auc(seq_10["index"].values, seq_10["value"].values)
)

# Compare totals
print(16.063286895622916 + 30.666541673483657)
print(15.922631436041161 + 30.764463436386382)
