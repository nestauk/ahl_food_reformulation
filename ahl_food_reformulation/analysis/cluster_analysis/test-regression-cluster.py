from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import cluster_analysis as cl

# read data
pur_recs = kantar.purchase_records()
nut_rec = kantar.nutrition()
pan_ind = kantar.household_ind()
prod_mast = kantar.product_master()
val_fields = kantar.val_fields()
uom = kantar.uom()
prod_codes = kantar.product_codes()
prod_vals = kantar.product_values()
panel_weight = kantar.panel_weights_year()
cl_kcal_share = kantar.cluster_kcal_share()
cl_adj_size = kantar.cluster_adj_size()


purch_recs_wide_share = cl.mk_reg_df_share(
    val_fields, pur_recs, prod_mast, uom, prod_codes, prod_vals, nut_rec, 2907
)

share_table = cl.reg_share(cl_kcal_share, panel_weight, purch_recs_wide_share, 0.05)

purch_recs_wide_abs = cl.mk_reg_df_adj(
    pan_ind, val_fields, pur_recs, prod_mast, uom, prod_codes, prod_vals, nut_rec, 2907
)

adj_table = cl.reg_adj(cl_adj_size, panel_weight, purch_recs_wide_abs, 0.05)
