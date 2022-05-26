# AHL Food Reformulation Project

## Methodological notes

A place to plan and track progress on analysis of product categories.

## Research questions

Data request with sample outputs can be accessed [here](https://docs.google.com/document/d/1kfFELv48ABfNIrcfo5Ze7VaS_nzD-bWDEjdXpRLu_cs/edit?usp=sharing)

1. Identify food categories that contribute the most to overall consumption of calories and core nutrients.
   To answer this question, we will analyse total volumes and nutritional value of products purchased by a representative sample of UK households. To account for seasonality and any short-term fluctuations we will use recent data for a period of at least 12 months. While we are primarily interested in population intake of calories and core nutrients, because of data availability
   constraints we will treat data on purchasing as a proxy for consumption.

An illustrative output is shown in Table 2 in the Appendix to this document.

2. Examine the composition of purchased foods by product price segment, manufacturer, type and location of households and product promotion.
   To evaluate the potential impact of food reformulation, we will analyse how total sales volume of a given category is disaggregated by:

- Price segment (by quantiles using data on purchasing spend)
- Manufacturer
- - Type of a household (using characteristics like social grade and life stage)
- Household location (as granular as possible starting from regions of the UK)

Several indicative outputs are provided in Table 3 and Figures 1 and 2 in the Appendix to this document.

3. Measure nutrient profile diversity of food categories as a proxy for feasibility of reformulation.
   We start with an assumption that the distribution of nutrient profiles within a given product category may signal feasibility of reformulation. Specifically, if there are substantial technical barriers to food reformulation this may impose limitations on nutrient content. This means we are more likely to see a lower level of variation in nutrient profiles (i.e. low nutrient diversity).
   To assess nutrient diversity of product categories we will start with data on energy and nutrient content of a) 100g/100ml of a product b) recommended serving size. We will then explore various options for measuring distribution of energy density and nutrient profiles within a given category. These options include:

- Entropy of energy density
- Pairwise and average cosine similarity of nutrient profile vectors
- Number of density based clusters (using DBSCAN or Gaussian Mixture Models)

A sample output is shown in Figure 3 in the Appendix section.

4. Assess changes in purchasing behaviour following product reformulation.
   Based on the availability of information on changes in content of core nutrients of products purchased by the household panel over time, we will carry out an initial exploration of changes in purchasing volumes following reformulation. We will start with calculating percentage change in sales volumes before and after the reformulation, but will explore technical feasibility of performing a more detailed time series analysis.
   A tentative output from this analysis will be a tracker of food reformulation activity combined with information on percentage change in sales. An early

## Outstanding questions

Some questions that we'll need to answer in order to address research questions listed above:

- Categories to drop? Previously discussed dropping fresh fruit and veg, fresh meat.
  In other studies people also removed Oils and Alcoholic drinks.
- Do we care about added sugars?
- Separate food from drink from the start?
- Classification of products into HFSS. What nutrient profile model to use? How
  to get estimates for fruit, veg and nut content.
- Will using nutrient diversity help capture instances when more than 1 nutrient
  needs changing - NNPS non-compensatory system sounds relevant.
- Serving size vs 100g
- Measuring nutrient diversity.
  - What metrics do people use for similar analyses? So far seen entropy of
    %energy within a given product. Also mean and SD of individual ingredients.
    Nestle Nutritional Profile System pass rate for individual products.
  - How would we evaluate them?
  - How would we compare different product categories (will comparisons be valid)?
  - Should we compare variation for very similar types of product (e.g. same type of soup)?
- Are we interested in nutrient %energy (i.e. calorific value)?
  Is using simple Atwater conversion sufficient?
- Any additional datasets we might need?

## Exploratory data analysis

- Quick check of what each variable is - link to data dict if available, typical
  range, methodology if derived.
- Missing values. Rank vars by % missing values.
- Any duplicates?
- Distributions and outliers (for each category). This will probably be both
  histograms, boxplots (or violin plots?) and countplots (or waffle charts) for categorical vars.
  Rank vars by number of outliers (below 1st and above 99 percentile).
  _ Energy density
  _ Each of the core nutrients (fat, sugar, salt, saturated fat, carbs, fibre, protein)
  _ Price
  _ Average monthly purchasing volumes \* Manufacturer
- Correlations (e.g. price vs energy density). Start with a correlation matrix.
- Dimensionality reduction. The idea here is to check whether foods fall into any
  broad nutrition types. UMAP and PCA. See info on how these can be combined
  [here](https://umap-learn.readthedocs.io/en/latest/faq.html#what-is-the-difference-between-pca-umap-vaes).
- Heatmaps for proportion of HFSS products?

## Points to ponder

- Would we want to look at level of food processing? E.g. ultra processed foods?
- Single nutrient policies are flawed (see paper on ultra-processed food)
- Supermarkets contribution to HFSS might be an interesting thing to look at
  (own brand).
- We are not likely to analyse nutrient density.
- Ultra processed foods may not be affected by simple food reformulation action.

## Early thoughts

- On diversity, in Tesco paper, authors looked at within product diversity, using
  entropy of %energy. Entropy was normalised [0,1]. It was obtained by dividing the entropy values
  by the maximum entropy, calculated as: log2 (number of distinct nutrients).

## Links

- Using entropy to measure concentration/dispersion of products across categories
  [stack exchange post](https://stats.stackexchange.com/questions/311329/what-is-the-best-way-of-measuring-the-dispersion-or-concentration-of-categorical)
