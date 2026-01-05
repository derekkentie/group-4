import pandas as pd

last_calculation_model_trainer = pd.read_csv(r"data/submission2.csv")
kaggle_submission = pd.read_csv(r"data/submission1_kaggle.csv")

res = last_calculation_model_trainer.compare(kaggle_submission)
print(res)