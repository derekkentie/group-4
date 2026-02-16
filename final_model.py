import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#selecting the preferred dicitionaries
molecule_features_dict_train = pickle.load(open(r"docs\mol representatie picklebestanden\train_molecule_combined_representation.pkl",'rb'))
molecule_features_dict_test = pickle.load(open(r"docs\mol representatie picklebestanden\test_molecule_combined_representation.pkl",'rb'))
protein_features_dict = pickle.load(open(r"docs\Sep's picklebestanden\protein dicts to use in gridsearch\dict ID to feature vector 2 in 2 pieces", 'rb'))

#loading the training set
train_df = pd.read_csv(r"data\train.csv")
test_df = pd.read_csv(r"data\test.csv")

X = []
y = []

#feature concatenation to combine each ligand-protein pair
for _, row in train_df.iterrows():
    smiles = row["molecule_SMILES"]
    protein = row["UniProt_ID"]
    affinity_score = row["affinity_score"]

    #quick check if all elements are available
    if smiles not in molecule_features_dict_train: 
        raise FileNotFoundError(
            f"The following SMILES exists in the trainingset but not in the molecule-features dictionary: {smiles}"
        )
    if protein not in protein_features_dict: 
        raise FileNotFoundError(
            f"The following Uniprot_ID exists in the trainingset but not in the protein-features dictionary: {protein}"
        )

    #feature concatenation
    if isinstance(molecule_features_dict_train[smiles], np.ndarray):
        molecule_features_dict_train[smiles] = molecule_features_dict_train[smiles].tolist()
    if isinstance(protein_features_dict[protein], np.ndarray):
        protein_features_dict[protein] = protein_features_dict[protein].tolist()
    combined = molecule_features_dict_train[smiles] + protein_features_dict[protein]

    #data seperation
    X.append(combined)
    y.append(affinity_score)
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

#creating input list for final submission
X_predict = []

for _, row in test_df.iterrows():
    smiles = row["molecule_SMILES"]
    protein = row["UniProt_ID"]

    #quick check if all elements are available
    if smiles not in molecule_features_dict_test: 
        raise FileNotFoundError(
            f"The following SMILES exists in the testset but not in the molecule-features dictionary: {smiles}"
        )
    if protein not in protein_features_dict: 
        raise FileNotFoundError(
            f"The following Uniprot_ID exists in the testset but not in the protein-features dictionary: {protein}"
        )

    #feature concatenation
    if isinstance(molecule_features_dict_test[smiles], np.ndarray):
        molecule_features_dict_test[smiles] = molecule_features_dict_test[smiles].tolist()
    if isinstance(protein_features_dict[protein], np.ndarray):
        protein_features_dict[protein] = protein_features_dict[protein].tolist()
    combined = molecule_features_dict_test[smiles] + protein_features_dict[protein]

    #data seperation
    X_predict.append(combined)
X_predict = np.array(X_predict, dtype=float)


#splitting the data in training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    #ONLY CHANGE THE TEST_SIZE BY PREFERENCE
    X, y, test_size=0.33, random_state=42 
)

#scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print('len X',len(X))
X = scaler.fit_transform(X)
X_predict = scaler.transform(X_predict)
print("minmax scaling complete")

#setting randomforest parameters
model = RandomForestRegressor(n_estimators=500,
                              max_features='sqrt',
                              max_depth = 400
)

#making the predictions
model.fit(X, y)
y_predict = model.predict(X_predict)

#creating the submission
submission = pd.DataFrame({
    "ID": test_df["ID"],
    "affinity_score": y_predict
})
submission.to_csv("data/final_submission.csv", index=False)
