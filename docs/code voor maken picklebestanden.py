import pickle
import pandas as pd
frame = pickle.load(open('aafeauture_dataframe.pkl','rb'))
# print(f"feature{' '*30}min{' '*7}max")
# for feature in frame.columns:
#     print(f"{feature:30} {min(frame[feature]):.12f} {max(frame[feature])}")
print(frame)