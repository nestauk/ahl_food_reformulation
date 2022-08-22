# Clustering households using Kantar data

The code in the `test_representations` folder produces a basic workflow which clusters households based on purchasing behaviour, testing different household represenations and parameters and saving the results. It then looks at the top purchases by k-calorie intake for each cluster as well as the demographic differences.

## Installation and running the code

### Clone and set up the repo

1. Follow the below steps to setup the project.

```shell
$ git clone https://github.com/nestauk/ahl_food_reformulation
$ cd ahl_food_reformulation
```

2. Run the command `make install` to create the virtual environment and install dependencies

### Download and save the data

Download the data from the [ahl-private-data](https://s3.console.aws.amazon.com/s3/buckets/ahl-private-data?region=eu-west-2&prefix=kantar/latest_data/&showversions=false) s3 bucket and save to the `inputs/data` folder.

<br>
-----------------------

### Test cluster pipeline


Download the latest data from the [ahl-private-data](https://s3.console.aws.amazon.com/s3/buckets/ahl-private-data?region=eu-west-2&prefix=kantar/latest_data/&showversions=false) s3 bucket and save to the `inputs/data` folder. This part is important for the code to run correctly.

1. [Run robust clustering](https://github.com/nestauk/ahl_food_reformulation/blob/31_consolidate_pipeline/ahl_food_reformulation/pipeline/robust_clustering.py) (with reduced parameters), save the panel_clusters.csv file. Also checks the number of households per cluster (weighted and un-weighted) incase you need to remove a cluster for step 2.
  `python ahl_food_reformulation/analysis/run_cluster_pipeline.py`
2. Run the below file to create a save demographic and product analysis plots.
  `python ahl_food_reformulation/analysis/clustering_outputs_analysis_update.py`

#### Outputs

- panel_clusters.csv file with household ID's and their assigned clusters from robust clusterng
- Demographic and product analysis plots saved in outputs/figures


<br>
-----------------------

### Analysis of cluster contents

Running the script `ahl_food_reformulation/analysis/clustering_outputs_analysis.py` reproduces the charts and analysis presented internally in July 2022 (charts are saved in `outputs/figures/png`).

You can modify the script to adapt it to new clustering outputs. Perhaps the cleanest strategy to do that is to change the `kantar.panel_clusters` table with new outputs so that households are automatically tagged with them when you read demographic information with `kantar.demog_clean()`. Another option would be to assign clusters in the script itself.
