from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import cluster_analysis as cl
from ahl_food_reformulation import PROJECT_DIR

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

rst_4_extended = 2907
rst_4_market_sector = 2828

# rst_4_extended


purch_recs_wide_share = cl.mk_reg_df_share(
    val_fields, pur_recs, prod_mast, uom, prod_codes, prod_vals, nut_rec, rst_4_extended
)

share_table = cl.reg_share(cl_kcal_share, panel_weight, purch_recs_wide_share, 0.05)

share_table.to_csv(
    f"{PROJECT_DIR}/outputs/data/decision_table/rst_4_extended_cluster_share.csv",
    float_format="%.3f",
)

purch_recs_wide_abs = cl.mk_reg_df_adj(
    pan_ind,
    val_fields,
    pur_recs,
    prod_mast,
    uom,
    prod_codes,
    prod_vals,
    nut_rec,
    rst_4_extended,
)

adj_table = cl.reg_adj(cl_adj_size, panel_weight, purch_recs_wide_abs, 0.05)

adj_table.to_csv(
    f"{PROJECT_DIR}/outputs/data/decision_table/rst_4_extended_cluster_adj.csv",
    float_format="%.3f",
)

# rst 4 market sector

purch_recs_wide_share = cl.mk_reg_df_share(
    val_fields,
    pur_recs,
    prod_mast,
    uom,
    prod_codes,
    prod_vals,
    nut_rec,
    rst_4_market_sector,
)

share_table = cl.reg_share(cl_kcal_share, panel_weight, purch_recs_wide_share, 0.05)

share_table.to_csv(
    f"{PROJECT_DIR}/outputs/data/decision_table/rst_4_market_sector_cluster_share.csv",
    float_format="%.3f",
)

purch_recs_wide_abs = cl.mk_reg_df_adj(
    pan_ind,
    val_fields,
    pur_recs,
    prod_mast,
    uom,
    prod_codes,
    prod_vals,
    nut_rec,
    rst_4_market_sector,
)

adj_table = cl.reg_adj(cl_adj_size, panel_weight, purch_recs_wide_abs, 0.05)

adj_table.to_csv(
    f"{PROJECT_DIR}/outputs/data/decision_table/rst_4_market_sector_cluster_adj.csv",
    float_format="%.3f",
)
