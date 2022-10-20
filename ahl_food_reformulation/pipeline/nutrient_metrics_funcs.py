# Import libraries
from ahl_food_reformulation import PROJECT_DIR
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
        sample = df.groupby(cat).sample(n=size)
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
