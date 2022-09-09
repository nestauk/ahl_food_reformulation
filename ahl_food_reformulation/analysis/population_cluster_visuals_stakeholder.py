# Read in libraries
from ahl_food_reformulation.getters import kantar
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.utils import lookups as lps
from ahl_food_reformulation import PROJECT_DIR
import altair as alt
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from matplotlib.pyplot import figure
from pylab import rcParams
import numpy as np
from cmath import nan

logging.info("loading data")
# Read in data
pur_recs = kantar.purchase_records()
nut_recs = kantar.nutrition()
prod_mast = kantar.product_master()
val_fields = kantar.val_fields()
uom = kantar.uom()
prod_meta = kantar.product_metadata()
clusters = kantar.panel_clusters().drop("Unnamed: 0", axis=1)
prod_codes = kantar.product_codes()
prod_vals = kantar.product_values()
prod_att = kantar.product_attribute()


def top_purchased_energy_density(high_den, size):
    """Most bought products categories in the high density category"""
    high_den_pop = (
        pd.pivot_table(
            high_den,
            values=["scaled_gross_up_factor"],
            index=["rst_4_extended"],
            aggfunc="sum",
        )
        .reset_index()
        .sort_values(by=["scaled_gross_up_factor"], ascending=False)
    )
    return list(high_den_pop.head(size)["rst_4_extended"])


def monthly_share_purchases(prod_all):
    """Group products by month, category and energy group."""
    month_cat_share_exc = (
        (
            prod_all.groupby(["month", "energy_density_cat", "rst_4_market_sector"])[
                ["scaled_gross_up_factor"]
            ].sum()
        )
        / (prod_all.groupby(["month"])[["scaled_gross_up_factor"]].sum())
        * 100
    )
    return month_cat_share_exc.reset_index()


def high_energy_per_cat(prod_all_exc):
    """Catgegorical purchase volume per cluster for high energy density."""
    cat_density_clusters_exl = (
        prod_all_exc.groupby(
            ["clusters", "rst_4_market_sector", "rst_4_extended", "energy_density_cat"]
        )[["scaled_gross_up_factor"]]
        .sum()
        .reset_index()
    )

    cat_density_clusters_exl = cat_density_clusters_exl.assign(
        prop_purchases=cat_density_clusters_exl.groupby(["clusters"])[
            ["scaled_gross_up_factor"]
        ].apply(transform.perc_variable)
    )
    cat_density_clusters_all_high = cat_density_clusters_exl[
        cat_density_clusters_exl["rst_4_extended"].isin(high_density_list)
    ].copy()
    return cat_density_clusters_all_high[
        cat_density_clusters_all_high.energy_density_cat == "high"
    ].copy()


logging.info("Transforming data")
# add standardised volume measurement
pur_rec_vol = transform.vol_for_purch(pur_recs, val_fields, prod_mast, uom)
# Conversion table
conv_meas = lps.measure_table(kantar.product_measurement())
# Measurements to convert
measures = ["Units", "Litres", "Servings"]
# Convert selected measures and combine with existing kilos
pur_rec_kilos = lps.conv_kilos(pur_rec_vol, conv_meas, measures)
# Merge cluster and purchase records
pur_rec_kilos_clusters = clusters.merge(pur_rec_kilos, how="left", on="Panel Id")
# Adding energy density measure
pur_recs_energy = transform.add_energy_density(pur_rec_kilos_clusters, nut_recs)
# Only looking at products with reported volume of kilos
pur_recs_energy["product_weight"] = np.where(
    pur_recs_energy["Reported Volume"] == "Kilos", pur_recs_energy["Volume"], nan
)
# subset to products with non-missing energy density
pur_recs_energy.dropna(inplace=True, subset=["energy_density_cat"])
# add scaled gross-up weight
pur_recs_energy["scaled_gross_up_factor"] = (
    pur_recs_energy["Gross Up Weight"] * pur_recs_energy["product_weight"]
)
# Month field
pur_recs_energy["month"] = pur_recs_energy["Purchase Date"].dt.month
# get product categories merge in product info
prod_all = prod_meta.merge(
    pur_recs_energy, left_on=["product_code"], right_on=["Product Code"], how="inner"
)
# Categories to exclude (less than 90% kilo purchases)
cats_exclude = [
    "Alcohol",
    "Ambient Bakery Products",
    "Chilled Bakery Products",
    "Chilled Drinks",
    "Dairy Products",
    "Frozen Confectionery",
    "Take Home Soft Drinks",
]

prod_all_exc = prod_all.loc[~prod_all["rst_4_market_sector"].isin(cats_exclude)].copy()

# Get monthly share of purchases
month_cat_share_exc = monthly_share_purchases(prod_all_exc)

logging.info("Plotting monthly share of high energy density purchases")
### Plot of monthly share of purhases as high energy density
source = month_cat_share_exc[month_cat_share_exc["energy_density_cat"] == "high"][
    ["month", "scaled_gross_up_factor", "rst_4_market_sector"]
]
source.columns = ["Month", "Scaled volume", "Category"]

source = source[source["Scaled volume"] > 0.5].copy()

# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection(
    type="single", nearest=True, on="mouseover", fields=["Month"], empty="none"
)

# The basic line
line = (
    alt.Chart(source)
    .mark_line(interpolate="basis")
    .encode(x="Month:Q", y="Scaled volume:Q", color="Category:N")
)

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = (
    alt.Chart(source)
    .mark_point()
    .encode(
        x="Month:Q",
        opacity=alt.value(0),
    )
    .add_selection(nearest)
)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = line.mark_text(align="left", dx=5, dy=-5).encode(
    text=alt.condition(nearest, "Category:N", alt.value(" "))
)

# Draw a rule at the location of the selection
rules = (
    alt.Chart(source)
    .mark_rule(color="gray")
    .encode(
        x="Month:Q",
    )
    .transform_filter(nearest)
)

# Put the five layers into a chart and bind the data
alt.layer(line, selectors, points, rules, text).properties(
    width=600, height=550
).configure_axis(grid=False, domain=False)

logging.info("Plotting high energy density across clusters and categories")
### High energy density purchases across clusters and categories
# Only high energy density
high_den = prod_all[prod_all.energy_density_cat == "high"]
# Top purchased high energy density products
high_density_list = top_purchased_energy_density(high_den, 39)
# Recode nut products into one category
prod_all_exc["rst_4_market_sector"] = np.where(
    prod_all_exc["rst_4_extended"] == "Nuts Standard Peanuts",
    "Take Home Savouries",
    prod_all_exc["rst_4_market_sector"],
)
cat_density_clusters_all_high = high_energy_per_cat(prod_all_exc)

source = cat_density_clusters_all_high[
    cat_density_clusters_all_high["prop_purchases"] != 0
].copy()
order = list(source["rst_4_market_sector"].value_counts().index)
alt.Chart(source).mark_circle().encode(
    x="clusters:O",
    y=alt.Y("rst_4_extended:O", sort=order, axis=alt.Axis(labelLimit=300, title=None)),
    size="prop_purchases:Q",
    color=alt.Color(
        "rst_4_market_sector",
        scale=alt.Scale(
            domain=order,
            range=[
                "#ff6e47ff",
                "#18a48cff",
                "#9a1bbeff",
                "#eb003bff",
                "#0000ffff",
                "#97d9e3ff",
            ],
        ),
    ),
).properties(height=800, width=650).configure_axis(labelFontSize=15)

logging.info("Plotting monthly share of high energy density purchases across clusters")
### Monthly share of high energy density purchase volume across clusters (select clusters highlighted)
month_density_clusters = (
    prod_all.groupby(["month", "clusters", "energy_density_cat"])[
        ["scaled_gross_up_factor"]
    ]
    .sum()
    .reset_index()
)

month_density_clusters = month_density_clusters.assign(
    prop_purchases=month_density_clusters.groupby(["clusters", "month"])[
        ["scaled_gross_up_factor"]
    ].apply(transform.perc_variable)
)

c = [0, 14, 15, 7, 3]
select_clusters = month_density_clusters[month_density_clusters.clusters.isin(c)]
source = select_clusters[select_clusters["energy_density_cat"] == "high"].copy()

select_clusters2 = month_density_clusters.loc[~month_density_clusters.clusters.isin(c)]
source2 = select_clusters2[select_clusters2["energy_density_cat"] == "high"].copy()

range_ = [
    "#ff6e47ff",
    "#D3D3D3",
    "#D3D3D3",
    "#9a1bbeff",
    "#D3D3D3",
    "#D3D3D3",
    "#D3D3D3",
    "#eb003bff",
    "#D3D3D3",
    "#D3D3D3",
    "#D3D3D3",
    "#D3D3D3",
    "#D3D3D3",
    "#D3D3D3",
    "#0000ffff",
    "#18a48cff",
]

highlight = alt.selection(
    type="single", on="mouseover", fields=["clusters"], nearest=True, empty="none"
)

foreground = (
    alt.Chart(source)
    .mark_line(strokeWidth=3)
    .encode(
        x="month:Q",
        y=alt.Y("prop_purchases:Q", scale=alt.Scale(domain=[9, 17])),
        color=alt.Color("clusters:N", scale=alt.Scale(range=range_)),
    )
    .properties(height=500, width=550)
)

background = (
    alt.Chart(source2)
    .mark_line(strokeWidth=3)
    .encode(
        x="month:Q",
        y=alt.Y("prop_purchases:Q", scale=alt.Scale(domain=[9, 18])),
        color=alt.condition(highlight, "clusters:N", alt.value("lightgray")),
        tooltip=["clusters:N"],
    )
    .add_selection(highlight)
    .properties(height=500, width=550)
)

main = background + foreground

main.configure_axis(grid=False, domain=False)

logging.info(
    "Plotting variation in purhcases of high energy density across clusters and categories"
)
### Variation in purhcases of high energy density across clusters and categories
month_cat_density_clusters = (
    prod_all_exc.groupby(["clusters", "rst_4_market_sector", "energy_density_cat"])[
        ["scaled_gross_up_factor"]
    ]
    .sum()
    .reset_index()
)
month_cat_density_clusters = month_cat_density_clusters.assign(
    prop_purchases=month_cat_density_clusters.groupby(["clusters"])[
        ["scaled_gross_up_factor"]
    ].apply(transform.perc_variable)
)

energy_density_source = month_cat_density_clusters[
    month_cat_density_clusters["energy_density_cat"] == "high"
].copy()
pur_recs_kcal = (
    pur_recs[["Panel Id", "Product Code", "PurchaseId", "Period", "Gross Up Weight"]]
    .merge(
        nut_recs[["Purchase Number", "Purchase Period", "Energy KCal"]],
        how="left",
        left_on=["PurchaseId", "Period"],
        right_on=["Purchase Number", "Purchase Period"],
    )
    .merge(clusters, how="left", on="Panel Id")
)


product_list = lps.product_table(
    val_fields, prod_mast, uom, prod_codes, prod_vals, prod_att
)[
    [
        "Product Code",
        "Reported Volume",
        "RST 4 Market Sector",
        "RST 4 Extended",
        "RST 4 Market",
    ]
]
pur_recs_kcal = pur_recs_kcal.merge(product_list, how="left", on="Product Code")
pur_recs_kcal["Gross Up kcal"] = (
    pur_recs_kcal["Energy KCal"] * pur_recs_kcal["Gross Up Weight"]
)

cluster_kcal_broad = (
    pur_recs_kcal.groupby(["clusters", "RST 4 Market Sector"])["Gross Up kcal"].sum()
    / (pur_recs_kcal.groupby(["clusters"])["Gross Up kcal"].sum())
) * 100  # .reset_index()
cluster_kcal_broad = cluster_kcal_broad.reset_index()
energy_kcal_clusters = energy_density_source.merge(
    cluster_kcal_broad,
    how="left",
    left_on=["clusters", "rst_4_market_sector"],
    right_on=["clusters", "RST 4 Market Sector"],
)
source = energy_kcal_clusters[energy_kcal_clusters["prop_purchases"] != 0].copy()

top_cats = [
    "Biscuits",
    "Take Home Confectionery",
    "Take Home Savouries",
    "Packet Breakfast",
    "Savoury Home Cooking",
]
top_five = source[source.rst_4_market_sector.isin(top_cats)].copy()


sns.set(style="white", font_scale=1.5)
source = top_five[
    ["clusters", "Gross Up kcal", "prop_purchases", "rst_4_market_sector"]
].copy()
source.clusters = source["clusters"].astype(str)
source = source[source.clusters.isin(["0", "2", "3", "7", "5", "15"])]
source.columns = ["Cluster", "Kcal share", "High density share", "Category"]

g = sns.relplot(
    data=source,
    x="High density share",
    y="Kcal share",
    hue="Category",
    col="Cluster",
    col_wrap=3,
    palette=["#ff6e47ff", "#18a48cff", "#9a1bbeff", "#eb003bff", "#0000ffff"],
    s=150,
)


g._legend.remove()
g.fig.set_size_inches(14, 9)

plt.show()

source = energy_kcal_clusters[energy_kcal_clusters["prop_purchases"] != 0].copy()
c = [0, 14, 2, 7, 3]
source = source[source.clusters.isin(c)].copy()

g = sns.relplot(
    data=source,
    x="rst_4_market_sector",
    y="prop_purchases",
    hue="clusters",
    size="Gross Up kcal",
    sizes=(10, 400),
    palette=["#ff6e47ff", "#18a48cff", "#9a1bbeff", "#eb003bff", "#0000ffff"],
)

g.fig.set_size_inches(17, 5)
plt.xticks(rotation=90)
plt.ylabel("percent of purchases", fontsize=14)
plt.xlabel("")
plt.show()
g.savefig("test.png", dpi=500)

cluster_kcal = (
    pur_recs_kcal.groupby(["clusters", "RST 4 Market Sector", "RST 4 Market"])[
        "Gross Up kcal"
    ].sum()
    / (pur_recs_kcal.groupby(["clusters"])["Gross Up kcal"].sum())
) * 100
c = [0, 14, 2, 7, 3]
cluster_kcal = cluster_kcal.reset_index()
cluster_kcal = cluster_kcal[cluster_kcal.clusters.isin(c)].copy()

rcParams["figure.figsize"] = 5, 5

data = cluster_kcal[
    cluster_kcal["RST 4 Market Sector"] == "Savoury Home Cooking"
].copy()
ax = sns.pointplot(
    x="Gross Up kcal", y="RST 4 Market", data=data, join=False, color="#9a1bbeff"
)
plt.xlabel("Percent of kcal purchased")
plt.ylabel("Savoury Home Cooking")

cluster_kcal_granular = (
    pur_recs_kcal.groupby(["clusters", "RST 4 Market Sector", "RST 4 Extended"])[
        "Gross Up kcal"
    ].sum()
    / (pur_recs_kcal.groupby(["clusters"])["Gross Up kcal"].sum())
) * 100
cluster_kcal_granular = cluster_kcal_granular.reset_index()
cluster_kcal_granular = cluster_kcal_granular[
    cluster_kcal_granular.clusters.isin(c)
].copy()

rcParams["figure.figsize"] = 5, 15


data = cluster_kcal_granular[
    cluster_kcal_granular["RST 4 Market Sector"] == "Ambient Bakery Products"
].copy()
ax = sns.pointplot(
    x="Gross Up kcal", y="RST 4 Extended", data=data, join=False, color="#9a1bbeff"
)
plt.xlabel("Percent of kcal purchased")
plt.ylabel("Ambient Bakery Products")

rcParams["figure.figsize"] = 5, 30


data = cluster_kcal_granular[
    cluster_kcal_granular["RST 4 Market Sector"] == "Dairy Products"
].copy()
ax = sns.pointplot(
    x="Gross Up kcal", y="RST 4 Extended", data=data, join=False, color="#ff6e47ff"
)
plt.xlabel("Percent of kcal purchased")
plt.ylabel("Dairy Products")

logging.info("Plotting top 20 categories on total kcal purchased")
### Total kcal purchased
total_kcal = (
    pur_recs_kcal.groupby(["RST 4 Market Sector", "RST 4 Market"])["Gross Up kcal"]
    .sum()
    .reset_index()
)
top_20 = total_kcal.sort_values(by="Gross Up kcal", ascending=False).head(20)

sns.set_theme(style="white")

g = sns.catplot(
    x="RST 4 Market",
    y="Gross Up kcal",
    hue="RST 4 Market Sector",  # col="time",
    palette=sns.color_palette("Spectral", 13),
    data=top_20,
    kind="bar",
    aspect=15 / 8.27,
    dodge=False,
)


plt.xticks(rotation=90)
plt.xlabel("Food category", fontsize=14)
plt.ylabel("Total kcal", fontsize=14)

plt.show()

logging.info("Plotting kcal share vs high energy density share across catgeories")
### Kcal share vs High density share across categories
kcal_share = (
    pur_recs_kcal.groupby(["RST 4 Market Sector"])["Gross Up kcal"].sum()
    / (pur_recs_kcal["Gross Up kcal"].sum())
) * 100
kcal_share = kcal_share.reset_index()

total_vol_cat = (
    prod_all_exc.groupby(["rst_4_market_sector"])["scaled_gross_up_factor"]
    .sum()
    .reset_index()
)
total_vol_cat.rename(
    {"scaled_gross_up_factor": "volume purchased"}, axis=1, inplace=True
)
perc_cat_sect = (
    prod_all_exc.groupby(["rst_4_market_sector", "energy_density_cat"])[
        "scaled_gross_up_factor"
    ].sum()
    / (prod_all["scaled_gross_up_factor"].sum())
) * 100
perc_cat_sect = perc_cat_sect.reset_index()
perc_cat_high = perc_cat_sect[perc_cat_sect.energy_density_cat == "high"].copy()

perc_cat_high = perc_cat_high.merge(
    kcal_share,
    how="left",
    right_on="RST 4 Market Sector",
    left_on="rst_4_market_sector",
).merge(total_vol_cat, how="left", on="rst_4_market_sector")
sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 20})
sns.axes_style("whitegrid")

source = perc_cat_high[
    [
        "rst_4_market_sector",
        "scaled_gross_up_factor",
        "Gross Up kcal",
        "volume purchased",
    ]
].copy()
source.columns = ["Category", "high density share", "kcal share", "volume"]
source = source[source["kcal share"] > 1].copy()

g = sns.relplot(
    data=source,
    x="high density share",
    y="kcal share",
    size="volume",
    sizes=(100, 900),
    color="#0000ffff",
)


def label_point(x, y, val, ax):
    a = pd.concat({"x": x, "y": y, "val": val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point["x"] + 0.1, point["y"], str(point["val"]), fontsize=20)


label_point(
    source["high density share"], source["kcal share"], source["Category"], plt.gca()
)
g._legend.remove()
plt.legend(
    fontsize="16", title_fontsize="20", bbox_to_anchor=(1.1, 1.05), frameon=False
)

g.fig.set_size_inches(15, 12)
plt.xticks(rotation=90)
plt.ylabel("")
plt.xlabel("")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
