import pandas as pd

print(gridsearch_csv["error"].unique())gridsearch_csv = pd.read_csv("docs/model_gridsearch_first_run_google_colab_fraction_100.csv")
successful = gridsearch_csv[gridsearch_csv["error"] == "None"]
print(len(successful), "successful experiments")
print(successful.head())
