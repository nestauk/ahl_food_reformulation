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

Download the data from the [ahl-private-data](https://s3.console.aws.amazon.com/s3/buckets/ahl-private-data?prefix=kantar%2Fdata_v3%2F&region=eu-west-2&showversions=false#) s3 bucket and save to following folders:

- `inputs/data`
  - purchase_records.csv
  - product_master.csv
  - validation_field.csv
  - uom.csv
  - product_attribute_coding.csv
  - product_attribute_values.csv
  - nutrition_data.csv
- `outputs/data`
  - panel_demographic_table_202110.csv

### Run the code

1. `cd ahl_food_reformulation/analysis/test_representations`
2. Run `python cluster_households.py`
3. Run `python review_clusters.py`

### Outputs

Running the script `cluster_households.py` produces figures depicting the clusters saved in [outputs/figures](https://github.com/nestauk/ahl_food_reformulation/outputs/figures). It also produces the file `panel_clusters.csv` which gives the household id and the cluster labels for each household representation.

### Analysis of cluster contents

Running the script `ahl_food_reformulation/analysis/clustering_1st_iteration/clustering_outputs_analysis.py` reproduces the charts and analysis presented internally in July 2022 (charts are saved in `outputs/figures/png`).

You can modify the script to adapt it to new clustering outputs. Perhaps the cleanest strategy to do that is to change the `kantar.panel_clusters` table with new outputs so that households are automatically tagged with them when you read demographic information with `kantar.demog_clean()`. Another option would be to assign clusters in the script itself.
