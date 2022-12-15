<!-- #region -->

# Clustering pipeline

`ahl_food_reformulation/pipeline/clusteting/save_optimum_clusters.py` contains code to:

1. Create two household representations based on the October purchasing activity:
   - Share of kcal
   - Adjusted share of kcal:
2. For each representation, runs k-means clustering on different numbers of k and plots the results
3. Using the optimum number of k based on the average silhoutte score for each representation:
   - Refits k-means
   - Clusters the centroids using k = 6
   - Saves the results into two csv files

Run it like this:

```
python ahl_food_reformulation/pipeline/clusteting/save_optimum_clusters.py

```

The resulting cluster assignments are used as part of the reformulation analysis to help choose the top categories to prioritise for reformulation. Low income clusters are identified with categories scored on the level of impact to these low income clusters.

<!-- #endregion -->
