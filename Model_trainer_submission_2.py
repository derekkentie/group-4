import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

#selecting the preferred dicitionaries
molecule_features_dict_train = pickle.load(open(r"docs\mol representatie picklebestanden\train_molecule_descriptor_representation.pkl",'rb'))
molecule_features_dict_test = pickle.load(open(r"docs\mol representatie picklebestanden\test_molecule_descriptor_representation.pkl",'rb'))
protein_features_dict = pickle.load(open(r"docs\Sep's picklebestanden\dict ID to featurevector in one-hot padded with nans", 'rb'))
print("dictionaries loaded")
#loading the training set
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X = []
y = []

#feature concatenation to combining each ligand-protein pair

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
print("feature concatenation complete")

#splitting the data in training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    #ONLY CHANGE THE TEST_SIZE BY PREFERENCE
    X, y, test_size=0.33, random_state=42 
)
print("data splitting complete")

#BELOW ARE OPTIONS FOR SCALING AND PCA, REMOVE DOCSTRINGS FOR
#THE PREFERRED OPTION(S)

#choose one of the following scaling option, or leave them out if preferred
"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_predict = scaler.transform(X_predict)
print("standard scaling complete")
"""

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.fit_transform(X)
#X_predict = scaler.transform(X_predict)
print("minmax scaling complete")

#apply PCA if preferred
r"""
ValueError: Input X contains NaN.
PCA does not accept missing values encoded as NaN natively. 
For supervised learning, you might want to consider 
sklearn.ensemble.HistGradientBoostingClassifier and Regressor 
which accept missing values encoded as NaNs natively. 
Alternatively, it is possible to preprocess the data, 
for instance by using an imputer transformer in a pipeline 
or drop samples with missing values. 
See https://scikit-learn.org/stable/modules/impute.html 
You can find a list of all estimators that handle NaN values 
at the following page: 
https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
"""
"""
pca = PCA(n_components=8)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)
print("PCA application complete")
"""
#NOW APPLY YOUR PREFERRED MODEL TYPE
r"""
ValueError: Input X contains NaN.
MLPRegressor does not accept missing values encoded as NaN natively. 
For supervised learning, you might want to consider 
sklearn.ensemble.HistGradientBoostingClassifier and Regressor 
which accept missing values encoded as NaNs natively. 
Alternatively, it is possible to preprocess the data, 
for instance by using an imputer transformer in a pipeline 
or drop samples with missing values. 
See https://scikit-learn.org/stable/modules/impute.html 
You can find a list of all estimators that handle NaN values 
at the following page: 
https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
"""

"""
model = MLPRegressor(
    hidden_layer_sizes=( 16, 8),
    activation='logistic',
    learning_rate='adaptive',
    max_iter=400,
    random_state=42
)
"""

# model = RandomForestRegressor(
#     criterion= 'absolute_error'
# )


model = HistGradientBoostingRegressor(
    loss= "absolute_error",
    learning_rate= 0.1,
    max_iter= 100
)

model.fit(X_train, y_train)
print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))

#FOR MAKING THE ACTUAL PREDICTIONS

model.fit(X, y)
y_predict = model.predict(X_predict)

submission = pd.DataFrame({
    "ID": test_df["ID"],
    "affinity_score": y_predict
})
submission.to_csv("data/submission2.csv", index=False)