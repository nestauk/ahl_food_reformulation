# Notes on EDA and prototyping for food reformulation analysis

## Feb 17th

Datasets used: [Open Food Facts](https://world.openfoodfacts.org/) (most recent version accessed on Feb 1, 2022)
Notebooks: Prototyping.py

Quick notes on the dataset:

- Contains over 2mln products with a range of variables on nutrient composition,
  brand, packaging.
- A subset of products sold in the US, UK, Ireland and Australia is about 500K rows.
- Has some kind of a hierarchy, captured in `categories` field.

### Data processing:

- After reviewing data fields description, kept a subset of columns. These
  include info on core nutrients + those in Nutrient Rich Foods 15 (not all are
  available though), `brands` looks useful for manufacturer proxy, `categories`
  for product category info, `ingredients` for potential NLP analysis, `serving size`
  to compare to values per 100g (though it's a string), `nutrition-score-uk` for
  NPS scores.
- Used csvkit in command line to only keep certain fields.
- Some products look like duplicates (e.g. same `product_name` and nutrient content).
- Many missing values, but still a good coverage for core nutrients.

### Started looking at soups:

- Subset using categories.str.contains("Soup").
- Had to iteratively refine categories - drop broths, stock, bouillon.
- Used `categories` to only keep 2 largest.
- Lot's of outliers. Explored 2 options for removing them (1.5\* IQR and [1,99] percentile).
- Missing values had to be removed as well.
- Plotted histograms and boxplots for some of the core nutrients.
- Looked at correlations:
  - Energy density positively correlated with fat, proteins and carbs
  - Salt and sodium (obvs)
  - Cholesterol negative to fiber
  - Carbs positive with protein
  - Fiber positive with proteins
  - Sugar moderately positive with carbs, energy and salt

#### Dimensionality reduction and plotting with UMAP

After curating the dataset for soups, prepared numeric columns for UMAP:
normalised with `sklearn.preprocessing.normalize`.

Mapped to 2-D, then mapped to 9 dimensions (same as num of variables). Only
doing this so that HDBSCAN clustering makes sense on UMAP projection. When
using straight normalised observations, the clusters make less sense visually.

Most recent run of HDBSCAN:

- params `min_cluster_size=25, min_samples=2, cluster_selection_method="leaf"`
- 39 clusters
- not that many outliers (~500 out of 3K) and the number is fairly stable even if I reduce min_cluster_size.
- useful patterns emerging - miso soups, condensed tomato soups, cream of soups.

What can be done next:

- explore clusters - which are high salt/ sugar? which are high energy density?
  what about fiber and protein?
- consider refining for more narrow categories (e.g. tomato soups)
- consider further pruning (e.g. removing miso soups)
- do PCA to understand how many dimensions capture variance
- consider switching to simpler category like Yogurts

## Feb 21

After talking to George:

- Define schema and then reshape datasets (so variable names remain consistent)
- Create a simplified Open Food Facts dataset with 3-4 categories and a few subcategories.
  - Bread
  - Yogurts
  - Sauces
  - Pizza
  - Sausages
- Separate data curation from EDA (different notebooks)

## March 8

Started building utils for subsequent analyses.

- General utilities:

  - `get_segments`: Dividing a pandas Series into segments. This will be used
    to derive price segments.

- Nutrient profiling.

  - Wrote functions for calculating A and C points using 2004-2005 NPS. All except
    one function works by partitioning a pandas Series into custom bins.
  - Where possible used daily recommended intake as upper limit for binning variables.
  - For `assign_nps_fiber_score` used thresholds for AOAC. Will need to ask Kantar
    what fiber definition they use.
  - For fruit, vegetable and nut content, define proportions as fractions (e.g. 0.4 instead of 40%)

- Entropy:
  - For continuous variables should use Kozachenko-Leonenko instead of Shanon see [here](https://stats.stackexchange.com/questions/347390/interpretation-of-entropy-for-continuous-distribution)
    and [here](https://stackoverflow.com/questions/43265770/entropy-python-implementation)
  - Installed `entropy_estimators` package
  - Reverted to just Gini coefficient for continuous variables (e.g. nutrients)
  - How about the Gini coefficient for the sum of NPS scores as a holistic measure of dispersion.
  - For entropy just used `scipy.stats.entropy`. Check out relative entropy in the future `Kullback–Leibler divergence`.
  - For entropy greater than 1 see [here](https://stats.stackexchange.com/questions/95261/why-am-i-getting-information-entropy-greater-than-1)
  - And on Gini, Entropy and Information gain [here](https://www.bogotobogo.com/python/scikit-learn/scikt_machine_learning_Decision_Tree_Learning_Informatioin_Gain_IG_Impurity_Entropy_Gini_Classification_Error.php)
  - Should I look at `skbio.diversity.alpha as gini_index`?
  - Looks interesting [too](https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation)

`
