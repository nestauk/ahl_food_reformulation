# Import modules
import pandas as pd
import numpy as np
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar as get_k
from ahl_food_reformulation.utils import lookups as lps
from ahl_food_reformulation.pipeline import transform_data as td

# Read in data
pan_mast = get_k.household_master()
pan_ind_mast = get_k.household_ind()
demog_coding = get_k.demog_coding()
demog_val = get_k.demog_val()
prod_mast = get_k.product_master()
val_fields = get_k.val_fields()
uom = get_k.uom()
prod_codes = get_k.product_codes()
prod_vals = get_k.product_values()
prod_att = get_k.product_attribute()

# Get tables
bmi_table = td.bmi_households(pan_ind_mast)
hh_demographics = lps.hh_demographic_table(demog_coding, demog_val, pan_mast)
product_info = lps.product_table(
    val_fields, prod_mast, uom, prod_codes, prod_vals, prod_att
)

# Check the first few rows
hh_demographics.head(5)

# Check the first few rows
bmi_table.head(5)

# Check the first few rows
product_info.head(5)

# Should be 29170 households + 127572
print(bmi_table.shape, hh_demographics.shape)
print(prod_mast.shape, product_info.shape)
