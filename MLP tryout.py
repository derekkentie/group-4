import numpy as np
import pandas as pd
import pickle
import random
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray


molecule_features = pickle.load(open(r"C:\Users\20234364\group-4\data\molecule_combined_representation_with_pca_8.pkl", 'rb'))
protein_features = pickle.load(open("docs\Sep's picklebestanden\dict ID to feature matrix",'rb'))
#chat gegeven code hieronder
df = pd.read_csv("data/train.csv")

X = []
y = []


# --- Convert protein feature matrices → fixed size vectors (mean pooling) ---
protein_vectors = {}

for protein_id, mat in protein_features.items():
    mat = np.array(mat)
    pooled = mat.mean(axis=0)     # shape: (feature_dim,)
    protein_vectors[protein_id] = pooled.tolist()

# --- Build raw feature matrix before scaling/PCA ---
for _, row in df.iterrows():
    smiles = row["molecule_SMILES"]
    protein = row["UniProt_ID"]

    if smiles not in molecule_features: 
        continue
    if protein not in protein_vectors: 
        continue

    combined = molecule_features[smiles].tolist() + protein_vectors[protein]
    X.append(combined)
    y.append(row["affinity_score"])

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)


#Train een MLP (sklearn)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=16)   # of 8, wat jij wilt

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

model = MLPRegressor(
    hidden_layer_sizes=( 16, 16, 8),
    activation='logistic',
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42
)

model.fit(X_train_pca, y_train)

print("Train score:", model.score(X_train_pca, y_train))
print("Test score:", model.score(X_test_pca, y_test))
