from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import cluster_analysis as cl
from ahl_food_reformulation import PROJECT_DIR
import logging


if __name__ == "__main__":

    logging.info("Reading Data")

    # read data
    pur_recs = kantar.purchase_records_updated()
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
    prod_meta = kantar.prod_meta_update()

    rst_4_market_sector = 2828

    out_mkt = cl.cluster_table(
        val_fields,
        pur_recs,
        prod_codes,
        prod_vals,
        nut_rec,
        prod_meta,
        panel_weight,
        cl_kcal_share,
        cl_adj_size,
        pan_ind,
        att_num=rst_4_market_sector,
        sig_level=0.05,
        top=0.25,
    )

    logging.info("Saving table - market sector")

    out_mkt.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/table_cluster_rst_4_market_sector.csv",
        float_format="%.3f",
    )

    rst_4_extended = 2907

    out_ext = cl.cluster_table(
        val_fields,
        pur_recs,
        prod_codes,
        prod_vals,
        nut_rec,
        prod_meta,
        panel_weight,
        cl_kcal_share,
        cl_adj_size,
        pan_ind,
        att_num=rst_4_extended,
        sig_level=0.05,
        top=0.25,
    )

    logging.info("Saving table - extended")

    out_ext.to_csv(
        f"{PROJECT_DIR}/outputs/data/decision_table/table_cluster_rst_4_extended.csv",
        float_format="%.3f",
    )
