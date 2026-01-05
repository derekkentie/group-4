import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

#added by Sep:
"""This code is added to make it easier to select specific physicochemical properties to use to make the model. 
It does so by first creating a dictionary coupling every aminoacid to a subsequent featurevector, after which new dictionaries are made coupling IDs to featurevectors or -matrices,
make sure you keep the dictionaries you need for the model. If you want to use a picklefile directly for your model, feel free to use ctrl + / to silence the added code."""


# ID_to_protein = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict ID to sequence",'rb'))
# aa_to_featurevec_unfilter = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict aminoacid to property vector 2",'rb')) 

# dict_to_concatenate =None
# #Above PAM250 is selected, can be exchanged with BLOSUM62, make sure it is a dictionary coupling aminoacids to vectors, not ID's to vectors, can be set to None if you only want to use physicochemical properties


# features_used = [0, 1, 2, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] #contains indices of physicochemical properties kept, feel free to experiment by changing what it contains, do not exceed 24 as there are no properties with a higher index
# #FYI: these are the highly correlating pairs (abs(R) > 0.9): [(3, 2), (5, 2), (5, 3), (17, 15), (21, 2), (21, 3)]
# aa_to_featurevec_filtered = dict()

# for aa in aa_to_featurevec_unfilter.keys(): #creating dictionary coupling aminoacids to featurevectors
#     value = aa_to_featurevec_unfilter[aa]
#     features_kept = []
#     for feature in features_used:
#         features_kept.append(value[feature])
#     if dict_to_concatenate is not None:
#         features_kept = np.concatenate([features_kept,dict_to_concatenate[aa]])
#     aa_to_featurevec_filtered[aa] = features_kept

# #creating dictionaries coupling IDs to featurevectors or -matrices   
# ID_to_featurevec = dict()
# ID_to_feature_mat = dict()
# ID_codes = ID_to_protein.keys()

# for ID in ID_codes:
#     protein = ID_to_protein[ID]
#     vec = np.array([])
#     for aa in protein:
#         vec = np.concatenate((vec,aa_to_featurevec_filtered[aa]))

#     ID_to_featurevec[ID] = vec
#     ID_to_feature_mat[ID] = vec.reshape((len(vec)//len(aa_to_featurevec_filtered['A']),len(aa_to_featurevec_filtered['A'])))

# #creating dictionaries coupling IDs to featurevectors or -matrices padded with nans
# lengths = [len(value) for value in ID_to_featurevec.values()]
# max_length = max(lengths)
# dict_ID_to_propertyvector_padded_with_nans = dict()
# dict_ID_to_propertymatrix_padded_with_nans = dict()


# for ID in ID_to_featurevec.keys():
#     value = ID_to_featurevec[ID]
#     nr_nans = max_length - len(value)
#     nan_array = (np.array([float('nan')]*nr_nans))
#     vec = np.concatenate((value,nan_array))
#     dict_ID_to_propertyvector_padded_with_nans[ID] = vec
#     dict_ID_to_propertymatrix_padded_with_nans[ID] = vec.reshape((len(vec)//len(aa_to_featurevec_filtered['A']),len(aa_to_featurevec_filtered['A'])))

# print(dict_ID_to_propertymatrix_padded_with_nans['P31749'][20])
# #End of: added by Sep

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
X_predict = np.array(X_predict, dtype=np.float32)
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
X = scaler.transform(X)
X_predict = scaler.transform(X_predict)
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

y_predict = model.predict(X_predict)
if len(y_predict) != 34626:
    raise IndexError(
        f"y_predict does not contain the same amount of samples as the test.csv file, but has {len(y_predict)} samles."
    )

submission = pd.DataFrame({
    "ID": test_df["ID"],
    "affinity_score": y_predict
})
submission.to_csv("data/submission2.csv", index=False)