from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.utils import lookups as lps
from ahl_food_reformulation import PROJECT_DIR
import logging

logging.info("loading data")
# Reading in data
pur_recs = kantar.purchase_records()
nut_recs = kantar.nutrition()
prod_mast = kantar.product_master()
val_fields = kantar.val_fields()
uom = kantar.uom()

logging.info("Slice by Kilos")
# Adding volume measurement and slicing by only kilos
pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)
pur_rec_kilos = pur_rec_vol[pur_rec_vol["Reported Volume"] == "Kilos"].copy()

logging.info("Getting unique product list")
# Applying function to get unique product list with selected per 100g nutritional info
unique_prods_nut = lps.products_per_100g(
    ["Energy KJ", "Protein KG"], pur_rec_kilos, nut_recs
)

# Checking a few examples
unique_prods_nut.head(5)

# Check for duplicates
unique_prods_nut["Product Code"].is_unique
