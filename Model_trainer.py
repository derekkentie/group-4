import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

#selecting the preferred dicitionaries
molecule_features_dict = pickle.load(open("docs\mol representatie picklebestanden\molecule_combined_representation.pkl",'rb'))
protein_features_dict = pickle.load(open("docs\Sep's picklebestanden\dict ID to physicochemical 2 + PAM250 vector padded with nans", 'rb'))

#loading the training set
train_df = pd.read_csv("data/train.csv")

X = []
y = []

#feature concatenation to combining each ligand-protein pair
for _, row in train_df.iterrows():
    smiles = row["molecule_SMILES"]
    protein = row["UniProt_ID"]

    #quick check if all elements are available
    if smiles not in molecule_features_dict: 
        raise FileNotFoundError(
            f"The following SMILES exists in the trainingset but not in the molecule-features dictionary: {smiles}"
        )
    if protein not in protein_features_dict: 
        raise FileNotFoundError(
            f"The following Uniprot_ID exists in the trainingset but not in the protein-features dictionary: {protein}"
        )

    #feature concatenation
    if isinstance(molecule_features_dict[smiles], np.ndarray):
        molecule_features_dict[smiles] = molecule_features_dict[smiles].tolist()
    if isinstance(protein_features_dict[protein], np.ndarray):
        protein_features_dict[protein] = protein_features_dict[protein].tolist()
    combined = molecule_features_dict[smiles] + protein_features_dict[protein]

    #data seperation
    X.append(combined)
    y.append(row["affinity_score"])

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

#splitting the data in training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    #ONLY CHANGE THE TEST_SIZE BY PREFERENCE
    X, y, test_size=0.33, random_state=42 
)

#BELOW ARE OPTIONS FOR SCALING AND PCA, REMOVE DOCSTRINGS FOR
#THE PREFERRED OPTION(S)

#choose one of the following scaling option, or leave them out if preferred
"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""
"""
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""

#apply PCA if preferred
"""
pca = PCA(n_components=8)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)
"""

#NOW APPLY YOUR PREFERRED MODEL TYPE
model = MLPRegressor(
    hidden_layer_sizes=( 16, 16, 8),
    activation='tanh',
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))
