# Prototyping

The scripts in this folder contain functions from analysis we used during the project but that did not end up in the final report. This includes a script to perform a 'robust clusterng' method on the household representations. Due to the speed of running we chose not to use this method for the project.

## Robust clustering

`ahl_food_reformulation/pipeline/prototyping/robust_clustering.py` contains the functions to perform robust / ensemble clustering. The steps below give detail on how to run the script but has not been updated since we decided not to use it in our main pipeline.

Run it like this:

```
from ahl_food_reformulation.pipeline.prototyping.robust_clustering import extract_clusters, clustering_params

cluster_df, cluster_lookup = extract_clusters(feature_df, 20, 5, 0.8)
```

In this case, `feature_df` is a pandas dataframe where the observations (e.g. households) are rows and the features e.g. items in shopping basket are columns. The other parameters are number of PCA components to extract (20), number of iterations to run each clustering algorithm (5) and resolution for community detection in the cluster graph (0.8).

The parameters to tune with multiple runs based on silhouette scores are pca components and community detection (the latter can also help to determine the granularity of the clustering solution that we extract),
