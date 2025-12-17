import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

#selecting the preferred dicitionaries
molecule_features_dict = pickle.load(open("docs\mol representatie picklebestanden\molecule_descriptor_representation.pkl",'rb'))
protein_features_dict = pickle.load(open("docs\Sep's picklebestanden\dict ID to featurevector in one-hot padded with nans", 'rb'))
print("dictionaries loaded")
#loading the training set
train_df = pd.read_csv("data/train.csv")

X = []
y = []

#feature concatenation to combining each ligand-protein pair
for _, row in train_df.iterrows():
    smiles = row["molecule_SMILES"]
    protein = row["UniProt_ID"]
    affinity_score = row["affinity_score"]
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
    y.append(affinity_score)

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

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
print("standard scaling complete")
"""
"""
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("minmax scaling complete)
"""
r"""
Traceback (most recent call last):
  File "c:\Users\20234364\group-4\Model_trainer.py", line 64, in <module>
    X_train = scaler.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\utils\_set_output.py", line 316, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\base.py", line 894, in fit_transform
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\utils\_set_output.py", line 316, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\base.py", line 894, in fit_transform
    data_to_wrap = f(self, X, *args, **kwargs)
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\base.py", line 894, in fit_transform
    return self.fit(X, **fit_params).transform(X)
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\preprocessing\_data.py", line 907, in fit
    return self.partial_fit(X, y, sample_weight)
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\base.py", line 1365, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\preprocessing\_data.py", line 1029, in partial_fit
    self.mean_, self.var_, self.n_samples_seen_ = _incremental_mean_and_var(
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\utils\extmath.py", line 1138, in _incremental_mean_and_var
    new_sum = _safe_accumulator_op(sum_op, X, axis=0)
  File "C:\ProgramData\anaconda3\lib\site-packages\sklearn\utils\extmath.py", line 1060, in _safe_accumulator_op
    result = op(x, *args, **kwargs)
  File "C:\ProgramData\anaconda3\lib\site-packages\numpy\lib\_nanfunctions_impl.py", line 727, in nansum
    a, mask = _replace_nan(a, 0)
  File "C:\ProgramData\anaconda3\lib\site-packages\numpy\lib\_nanfunctions_impl.py", line 109, in _replace_nan
    a = np.array(a, subok=True, copy=True)
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 4.69 GiB for an array with shape (9942, 63359) and data type float64
PS C:\Users\20234364\group-4>
"""

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

model.fit(X_train, y_train)
print("model fitting complete")


"""
model = RandomForestRegressor(
    criterion= 'absolute_error'
)
model.fit(X_train, y_train)
print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))