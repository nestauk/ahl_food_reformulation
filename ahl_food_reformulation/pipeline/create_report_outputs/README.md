# Create report outputs

This folder contains the code to reproduce the analysis and visualisations used in the reformulation report.

Run the files in this folder in the following order:

1. `create_report_table.py`: For each category level creates all indicators and combines them into one dataframe and saves to csv. To note, file csv files are saved locally to outputs/data whereas the files read in step two are from
2. `make_table_analysis.py`: Produces ranked list of categories based on aggregate indicators and saves json files and visualisations (created from ranked list).
3. Calulates the reduction in kcal at population level from reformulating the chosen categories and produces visualisations from the results: `impact-kcal-plots.py` and `impact-kcal-auc.py`.
4. `visualise_targets.py`: Produces additional visualisations used in the report.
5. `macro_nutrient_plots.py`: Visualises the macro nutrient levels in each of the chosen categories.
