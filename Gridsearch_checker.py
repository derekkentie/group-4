import pandas as pd

gridsearch_csv = pd.read_csv("docs/model_gridsearch.csv")
print(gridsearch_csv["error"].unique())