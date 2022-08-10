### Representations to test
# Testing new ways to represent the households purchasing.

# Representations
# - [x] Household kcal contribution to volume
# - [x] Household kcal purchased
# - [x] 1 year of household purchases
# - [ ] Absolute kcal / household size (with some weighting for children)

# Import libraries and directory
from sklearn.preprocessing import MinMaxScaler
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.getters import kantar as get_k
from ahl_food_reformulation.pipeline import transform_data as td
from ahl_food_reformulation.pipeline import create_clusters as cc

# Get data
purch_recs = get_k.purchase_subsets(
    202110
)  # Creating one month subset or reads file if exists - 202108
nut_subset = get_k.nutrition_subsets(
    202110
)  # Creating one month subset or reads file if exists - 202108
prod_mast = get_k.product_master()
val_fields = get_k.val_fields()
uom = get_k.uom()
prod_codes = get_k.product_codes()
prod_vals = get_k.product_values()
nutrition = get_k.nutrition()
purch_recs_year = get_k.purchase_records()

## Get time periods
# October 2021
purch_recs = td.combine_files(
    val_fields, purch_recs, prod_mast, uom, prod_codes, prod_vals, 2907
)
purch_recs_subset = td.nutrition_merge(nut_subset, purch_recs, ["Energy KCal"])
# Whole time (1 year)
purch_recs_year = td.combine_files(
    val_fields, purch_recs_year, prod_mast, uom, prod_codes, prod_vals, 2907
)
purch_recs_year = td.nutrition_merge(nutrition, purch_recs_year, ["Energy KCal"])

# Get total purchases per product and household for dataset
purch_recs_subset = td.total_product_hh_purchase(purch_recs_subset)
purch_recs_year = td.total_product_hh_purchase(purch_recs_year)

# Testing also removing units as looks like its not a consistent form of measurement (need to confirm this with Kantar).
# It shouldn't affect the household kcal representation as it doesn't use measurements so only testing it on the kcal / volume contribution.
# Removing units
purch_recs_year_units_rem = purch_recs_year[
    purch_recs_year["Reported Volume"] != "Units"
].copy()

# Get kcal contribution to volume for each time period (and units removed)
kcal_vol_subset = td.kcal_contribution(purch_recs_subset)
kcal_vol_year = td.kcal_contribution(purch_recs_year)
kcal_vol_year_nu = td.kcal_contribution(purch_recs_year_units_rem)

# Get household kcal for each time period
hh_kcal_subset = td.hh_kcal(purch_recs_subset)
hh_kcal_year = td.hh_kcal(purch_recs_year)

### Test clustering representations
k_numbers = [2, 4, 6, 10, 15, 20, 25, 30, 35, 40, 50]
# Household kcal purchased - Oct 2021
silh_hh_k_subset = cc.test_fit_clusters(hh_kcal_subset, "hh_kcal_sub", k_numbers)
# Household kcal purchased - Year
silh_hh_k_year = cc.test_fit_clusters(hh_kcal_year, "hh_kcal_year", k_numbers)
# Household kcal contribution to volume - Oct 2021
silh_kv_subset = cc.test_fit_clusters(kcal_vol_subset, "kcal_vol_sub", k_numbers)
# Household kcal contribution to volume - year
silh_kv_year = cc.test_fit_clusters(kcal_vol_year, "kcal_vol_year", k_numbers)
# Household kcal contribution to volume - year (no units)
silh_kv_nu = cc.test_fit_clusters(kcal_vol_year_nu, "kcal_vol_year_nu", k_numbers)
