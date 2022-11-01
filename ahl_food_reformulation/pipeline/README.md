<!-- #region -->

# Processing pipeline

The scripts in this folder can be used to process the kantar data from cleaning and transformation to clustering and the production of indicators for the reformulation report. The headers below give details on how to run certain scripts.

## Cluster methods

`ahl_food_reformulation/pipeline/cluster_methods.py` contains the functions to perform hierarcical and k-means clustering including transforming the data, fitting the models and reporting results.

## Robust clustering

`ahl_food_reformulation/pipeline/robust_clustering.py` contains the functions to perform robust / ensemble clustering.

Run it like this:

```
from ahl_food_reformulation.pipeline.robust_clustering import extract_clusters, clustering_params

cluster_df, cluster_lookup = extract_clusters(feature_df, 20, 5, 0.8)
```

In this case, `feature_df` is a pandas dataframe where the observations (e.g. households) are rows and the features e.g. items in shopping basket are columns. The other parameters are number of PCA components to extract (20), number of iterations to run each clustering algorithm (5) and resolution for community detection in the cluster graph (0.8).

The parameters to tune with multiple runs based on silhouette scores are pca components and community detection (the latter can also help to determine the granularity of the clustering solution that we extract),

<!-- #endregion -->
