# Import libraries
from ahl_food_reformulation import PROJECT_DIR
from ahl_food_reformulation.pipeline import transform_data as transform
from ahl_food_reformulation.utils.plotting import configure_plots
import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler


def rank_col(df_col):
    return df_col.rank(ascending=False).astype(int)


def cat_entropy(df: pd.DataFrame, cat: str):
    """
    Calculates the entropy value per category of products
    Args:
        df (pd.DataFrame): Pandas dataframe kcal per 100g/ml per product
        cat (str): Product category
    Returns:
        pd.Series: Series with entropy per category
    """
    df["deciles"] = pd.qcut(df["kcal_100g_ml"], 10, labels=False)
    return (
        df.replace([np.inf, -np.inf], 0)
        .groupby(cat)["deciles"]
        .apply(lambda x: entropy(x.value_counts(), base=2))
    )


def cat_variance(df: pd.DataFrame, cat: str, metric: str):
    """
    Calculates the variance per category of products
    Args:
        df (pd.DataFrame): Pandas dataframe kcal per 100g/ml per product
        cat (str): Product category
        metric (str): column name to calculate variance on
    Returns:
        pd.Series: Series with variance per category
    """
    return df.replace([np.inf, -np.inf], 0).fillna(0).groupby(cat)[metric].var()


def avg_metric_samples(df: pd.DataFrame, num_runs: int, cat: str, size: int):
    """
    Calculates average entropy and variance per category of products per selected sample size and number of runs
    Args:
        df (pd.DataFrame): Pandas dataframe kcal per 100g/ml per product
        num_runs (int): Number of times to sample the data and collect results
        cat (str): Product category
        size (int): Size of sample
    Returns:
        pd.DataFrame: Dataframe of avg entropy and variance per category of products
    """
    ent_list = []
    var_list = []
    for i in range(num_runs):
        sample = df.groupby(cat).sample(random_state=1, n=size)
        ent_list.append(cat_entropy(sample, cat).values)
        var_list.append(cat_variance(sample, cat, "kcal_100g_ml").values)
    return pd.DataFrame(
        {
            "entropy_size_adj": np.mean(ent_list, axis=0),
            "variance_size_adj": np.mean(var_list, axis=0),
            cat: cat_entropy(sample, cat).index,
        }
    )


def create_diversity_df(df: pd.DataFrame, cat: str, n_runs: int, sample_size: int):
    """
    Applies the entropy and variance functions to get total results for df and avg per samples
    Args:
        df (pd.DataFrame): Pandas dataframe kcal per 100g/ml per product
        cat (str): Product category
        num_runs (int): Number of times to sample the data and collect results
        sample_size (int): Size of sample
    Returns:
        pd.DataFrame: Dataframe of total and avg per samples entropy and variance per category of products
    """
    scaler = MinMaxScaler()  # Add scaler (minmax)
    df_diversity = pd.concat(
        [
            df.groupby(cat).size(),
            cat_entropy(df, cat),
            cat_variance(df, cat, "kcal_100g_ml"),
        ],
        axis=1,
    )

    df_diversity.columns = ["count", "entropy", "variance"]
    avg_metrics = avg_metric_samples(df, n_runs, cat, sample_size)
    df_merged = df_diversity.merge(avg_metrics, on=cat)
    df_merged[["variance_scaled", "variance_adj_scaled"]] = scaler.fit_transform(
        df_merged[["variance", "variance_size_adj"]]
    )
    df_merged[["var_rank", "var_adj_rank"]] = df_merged[
        ["variance", "variance_size_adj"]
    ].apply(rank_col)
    return df_merged


def diversity_heatmaps(
    df_diversity: pd.DataFrame,
    cat: str,
    entropy_values: str,
    variance_values: str,
    filename: str,
):
    """
    Plots heatmaps of count, entropy and variance metrics and saves file
    Args:
        df_diversity (pd.DataFrame): Pandas dataframe nutrient diversity metrics
        cat (str): Product category
        entropy_values (str): entropy column name
        variance_values (str): variance column name
        filename (str): String to append to filename
    Returns:
        Heatmap plots
    """
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 8))

    cmap = sns.cm.rocket_r

    sns.heatmap(
        df_diversity.set_index(cat)[[entropy_values]].sort_values(
            by=entropy_values, ascending=False
        ),
        ax=ax1,
        cmap=cmap,
    )
    sns.heatmap(
        df_diversity.set_index(cat).sort_values(by=entropy_values, ascending=False)[
            [variance_values]
        ],
        ax=ax2,
        cmap=cmap,
    )
    sns.heatmap(
        df_diversity.set_index(cat).sort_values(by=entropy_values, ascending=False)[
            ["count"]
        ],
        ax=ax3,
        cmap=cmap,
    )

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Method")
        ax.set_ylabel("Category")

    plt.tight_layout()
    # Add folder if not already created
    Path(f"{PROJECT_DIR}/outputs/figures/nutrient_diversity/").mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(
        f"{PROJECT_DIR}/outputs/figures/nutrient_diversity/diversity_heatmaps_"
        + filename
        + ".png",
        bbox_inches="tight",
    )
    plt.show(block=False)


def get_nut_proportions(nut_props_cat: pd.DataFrame, cat: str):
    """
    Creates new columns to show the proportion of Carbohydrate's protein and fat out of the Energy kcal
    Args:
        nut_props_cat (pd.DataFrame): Pandas dataframe including kcal and macro nutrients
        cat (str): Name of category level
    Returns:
        pd.DataFrame: Dataframe with added columns
    """
    # Calculate the proportions
    nut_props_cat["Carb_prop"] = (
        (nut_props_cat["carb_gross"] * 4000) / nut_props_cat["kcal_gross"]
    ) * 100
    nut_props_cat["Prot_prop"] = (
        (nut_props_cat["prot_gross"] * 4000) / nut_props_cat["kcal_gross"]
    ) * 100
    nut_props_cat["Fat_prop"] = (
        (nut_props_cat["fat_gross"] * 9000) / nut_props_cat["kcal_gross"]
    ) * 100
    nut_props_cat["Sum_props"] = (
        nut_props_cat["Carb_prop"]
        + nut_props_cat["Prot_prop"]
        + nut_props_cat["Fat_prop"]
    )
    return nut_props_cat


def macro_diversity(pur_nut_info: pd.DataFrame, cat: str):
    """
    Produces df with variance calculated for each each category for each macro nutrient
    Args:
        pur_nut_info (pd.DataFrame): Pandas dataframe including kcal, macro nutrients and categories
        cat (str): Name of category level
    Returns:
        pd.DataFrame: Dataframe macro nutrient diversity
    """
    # Create nutrition proportions
    nut_props_prod = pur_nut_info.groupby([cat, "Product Code"])[
        ["kcal_gross", "carb_gross", "prot_gross", "fat_gross"]
    ].sum()
    prod_nut_props = get_nut_proportions(nut_props_prod, cat).reset_index()
    # Create diversity table
    df_diversity = pd.concat(
        [
            cat_variance(prod_nut_props, cat, "Carb_prop"),
            cat_variance(prod_nut_props, cat, "Prot_prop"),
            cat_variance(prod_nut_props, cat, "Fat_prop"),
        ],
        axis=1,
    )
    df_diversity.columns = ["Carb_variance", "Prot_variance", "Fat_variance"]
    return df_diversity


def macro_nutrient_table(
    pur_recs: pd.DataFrame, prod_meta: pd.DataFrame, nut_recs: pd.DataFrame, cat: str
):
    """
    Creates macro nutrient table
    Args:
        pur_recs (pd.DataFrame): Pandas dataframe purchase records
        prod_meta (pd.DataFrame): Pandas dataframe product metadata
        nut_recs (pd.DataFrame): Pandas dataframe nutritional info for purchases
        cat (str): Name of category level
    Returns:
        pd.DataFrame: Dataframe macro nutrient diversity
    """
    # Combine purchase, product and nutrition info
    comb_files = pur_recs[
        ["PurchaseId", "Period", "Product Code", "Gross Up Weight"]
    ].merge(
        prod_meta[["product_code", cat]],
        left_on=["Product Code"],
        right_on="product_code",
        how="left",
    )
    pur_nut_info = transform.nutrition_merge(
        nut_recs, comb_files, ["Energy KCal", "Carbohydrate KG", "Protein KG", "Fat KG"]
    )

    pur_nut_info["kcal_gross"] = (
        pur_nut_info["Energy KCal"] * pur_nut_info["Gross Up Weight"]
    )
    pur_nut_info["carb_gross"] = (
        pur_nut_info["Carbohydrate KG"] * pur_nut_info["Gross Up Weight"]
    )
    pur_nut_info["prot_gross"] = (
        pur_nut_info["Protein KG"] * pur_nut_info["Gross Up Weight"]
    )
    pur_nut_info["fat_gross"] = pur_nut_info["Fat KG"] * pur_nut_info["Gross Up Weight"]

    # Create proportions table
    # Group by category and sum the macro nutrients
    nut_props_cat = pur_nut_info.groupby([cat])[
        ["kcal_gross", "carb_gross", "prot_gross", "fat_gross"]
    ].sum()
    cat_nut_props = get_nut_proportions(nut_props_cat, cat).reset_index()

    # Create diversity table
    df_diversity = macro_diversity(pur_nut_info, cat).reset_index()

    return cat_nut_props.merge(df_diversity, on=cat)


def plot_macro_proportions(
    broad_plot_df: pd.DataFrame, gran_plot_df: pd.DataFrame, driver, broad_cats
):
    """
    Plots proportions of macro nutrients for both categories as stacked bar charts
    Args:
        broad_plot_df (pd.DataFrame): Pandas dataframe of broad category proportions
        gran_plot_df (pd.DataFrame): Pandas dataframe of granular category proportions
        driver = google_chrome_driver_setup,
        broad_cats (str): List of broad categories to use for title
    Returns:
        Saved plots
    """
    # Create broad cats plot
    fig_broad = (
        alt.Chart(broad_plot_df)
        .mark_bar()
        .encode(
            y=alt.Y("Categories", title="Categories", axis=alt.Axis(titlePadding=20)),
            x=alt.X("sum(proportions):Q", axis=alt.Axis(format="%")),
            color="Macro nutrients",
        )
    )
    configure_plots(
        fig_broad,
        "Macronutrient proportions for top candidates",
        "",
        16,
        20,
        16,
    )
    # Create granular cats plot
    fig_gran = (
        alt.Chart(gran_plot_df)
        .mark_bar()
        .encode(
            y=alt.Y("Categories", title="Categories", axis=alt.Axis(titlePadding=20)),
            x=alt.X("sum(proportions):Q", axis=alt.Axis(format="%")),
            color="Macro nutrients",
            facet=alt.Facet(
                "Market sector:N", columns=2, header=alt.Header(labelFontSize=16)
            ),
        )
    )
    configure_plots(
        fig_gran,
        "Macronutrient proportions for " + broad_cats[0],
        "",
        16,
        20,
        16,
    )
    return fig_broad, fig_gran
