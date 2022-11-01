<!-- #region -->

# Analysis of the Kantar data

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

## Folders

### Analysis of decision table

Run the script `ahl_food_reformulation/analysis/decision_table_analysis/make_table_analysis.py` to reproduce the figures and tables used in the final report, which will be saved in the `outputs/figures/png` and `outputs/reports` folders.

<!-- #endregion -->
