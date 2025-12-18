import pickle
import pandas as pd
frame = pickle.load(open('docs\mol representatie picklebestanden\molecule_descriptor_representation.pkl','rb'))
# print(f"feature{' '*30}min{' '*7}max")
# for feature in frame.columns:
#     print(f"{feature:30} {min(frame[feature]):.12f} {max(frame[feature])}")

train_df = pd.read_csv(r"C:\Users\20234364\group-4\data\train.csv")


#feature concatenation to combining each ligand-protein pair
for _, row in train_df.iterrows():
    smiles = row["molecule_SMILES"]
    print(frame[smiles][1])
print(len(frame), type(frame))
print(len(train_df))
print(train_df["molecule_SMILES"].nunique())

missing = set(train_df["molecule_SMILES"]) - set(frame.keys())
print(len(missing))
print(train_df["molecule_SMILES"].value_counts().head())