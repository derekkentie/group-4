import numpy as np
import pandas as pd
import pickle #to un-encode the dictionaries
from pathlib import Path #used for looping through representation dictionaries
from itertools import product #used for gridsearch on hyperparameters

#data processing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#model libraries
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


class Modeltrainer:
    """
    This program is made to perform a gigantic gridsearch, not ontly on model-
    specific hyperparameters, but on all the different possible combinations
    between: molecule and protein representation, representation-combination
    method, scaling method, PCA options, ML model, and ML model specific
    hyperparameters. The program will then store all its findings in an excel
    with columns: train_score, test_score, combination.

    IT IS STRONGLY ADVISED TO NOT RUN THIS PROGRAM ON YOUR PERSONAL LAPTOP!
    
    Rather, make use of the available gpu's on Google Colab by importing the
    complete folder, named 'group-4', on Colab and then running the current
    program, named 'Model_trainer.py', while making use of the gpu's.
    """
    def __init__(self):
        self.dict_of_representation_combination_methods = {
            "feature concatenation": self.feature_concatenation,
            "Late Fusion": self.late_fusion,
            "Mid-level Fusion": self.mid_level_fusion,
            "Cross-attention": self.cross_attention,
            "End-to-end Multimodel Learning": self.multimodel_learning
            }
        
        self.dict_of_model_builders = {
            "Ridge": self.build_ridge,
            "Lasso": self.build_lasso,
            "Random Forest": self.build_random_forest,
            "Gradient Boosting Regressor": self.build_gbr,
            "Hist Gradient Boosting Regressor": self.build_hgbr,
            "Support Vector Regression": self.build_svr,
            "Multi-Layer Perceptron": self.build_mlp,
            "K Nearest Neighbors": self.build_knn,
        }

        self.dict_of_model_hyperparams = {
            "Ridge": {
                "alpha": [0.1, 1.0, 10.0, 100.0]
            },
            "Lasso": {
                "alpha": [1e-4, 1e-3, 1e-2]
            },
            "Random Forest": {
                "n_estimators": [200, 500],
                "max_depth": [None, 20],
                "min_samples_split": [2, 5]
            },
            "Gradient Boosting Regressor": {
                "n_estimators": [200, 500],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            },
            "Hist Gradient Boosting Regressor": {
                "max_iter": [200, 500],
                "learning_rate": [0.05, 0.1],
                "max_depth": [6, 8, 10],
                "max_bins": [128, 255]
            },
            "Support Vector Regression": {
                "kernel": ["rbf"],
                "C": [1.0, 10.0, 100.0],
                "gamma": ["scale"]
            },
            "Multi-Layer Perceptron": {
                "hidden_layer_sizes": [(256, 128), (512, 256)],
                "learning_rate_init": [1e-3, 1e-4],
                "alpha": [1e-4],
                "max_iter": [500]
            },
            "K Nearest Neighbors": {
                "n_neighbors": [5, 10, 20],
                "weights": ["distance"]
            }
        }

        self.dict_of_data_processors = {
            "scaler": [self.standard_scaling, self.minmax_scaling, self.no_scaling],
            "pca": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
    
    def data_loader(self, data_location):
        data = pd.read_csv(rf"{data_location}")
        return data
 
    def representation_gridsearch(self, folder_location):
        """
        Returns a list of filepaths from that representations that are
        included in the representation-folder.
        """
        representation_folder = Path(folder_location)
        return list(representation_folder)

    def pickle_converter(self, pickle_file_path):
        """
        Returns the unencoded/unpickled document of pickle_file_path.
        """
        with pickle_file_path.open("rb") as pickle_file_handle:
            return pickle.load(pickle_file_handle)

    def dicts_collector(self, map_location):
        """
        Returns a dictionary of dictionaries from both the molecule and
        protein representations, with key equal to file_path_name and item
        equal to the respective representation dicitionary which is 
        retreived using the pickle_converter function.
        """
        pickle_file_paths = self.representation_gridsearch(rf"{map_location}")
        dict_of_rep_dicts = {}
        for pickle_file_path in pickle_file_paths:
            file_path_name = pickle_file_path.stem #collects only the file name from the Path object
            dict_of_rep_dicts[file_path_name] = self.pickle_converter(pickle_file_path)
        if len(pickle_file_paths) == len(dict_of_rep_dicts):
            return dict_of_rep_dicts
        else:
            raise AttributeError(
                "something went wrong with extracting the dictionaries in the dict_collector function"
            )
    
    def hyperparam_generator(self, model_name):
        grid = self.model_hyperparams[model_name]
        keys = grid.keys()
        values = grid.values()

        for combination in product(*values):
            yield dict(zip(keys, combination))   

    def model_builder(self, model_name, hyperparams):
        """
        Returns the application of a combinations of hyperparameters
        to a model.
        """
        model_builder = self.dict_of_model_builders[model_name]
        return model_builder(hyperparams)
 
    def experiment_maker(self):
        """
        Returns every possible combination between de representations,
        representation-combination methods, models and hyperparameters.
        """
        combinations = {}
        mol_rep_dict_of_dicts = self.dicts_collector("docs\mol representatie picklebestanden")
        protein_rep_dict_of_dicts = self.dicts_collector("docs/Sep's picklebestanden")
        
###############    Model builders   ###############
    def build_ridge(self, hyperparams):
        return Ridge(**hyperparams)
    
    def build_lasso(self, hyperparams):
        return Lasso(**hyperparams)    
    
    def build_random_forest(self, hyperparams):
        return RandomForestRegressor(**hyperparams)
    
    def build_gbr(self, hyperparams):
        return GradientBoostingRegressor(**hyperparams)
    
    def build_hgbr(self, hyperparams):
        return HistGradientBoostingRegressor(**hyperparams)
    
    def build_svr(self, hyperparams):
        return SVR(**hyperparams)
    
    def build_mlp(self, hyperparams):
        return MLPRegressor(**hyperparams)
    
    def build_knn(self, hyperparams):
        return KNeighborsRegressor(**hyperparams)
###################################################


############### Combination methods ###############
       
    def feature_concatenation(self, molecule_feature_dict, protein_feature_dict, data):
        """
        Returns a numpy array of the concetenated features form the molecule
        and protein feature dictionaries. 

        This function checks if every item from the datafile is exists in both
        feature dictionaries, and will otherwise raise an error.

        If the datafile contains affinity scores (which means the data file
        is the train.csv file) it will collect those in the numpy array y and 
        also return it.
        """
        X = []

        #feature concatenation to combining each ligand-protein pair
        for _, row in data.iterrows():
            smiles = row["molecule_SMILES"]
            protein = row["UniProt_ID"]

            #quick check if all elements are available
            if smiles not in molecule_feature_dict: 
                raise FileNotFoundError(
                    f"The following SMILES exists in the trainingset but not in the molecule-features dictionary: {smiles}"
                )
            if protein not in protein_feature_dict: 
                raise FileNotFoundError(
                    f"The following Uniprot_ID exists in the trainingset but not in the protein-features dictionary: {protein}"
                )
        

            #feature concatenation
            if isinstance(molecule_feature_dict[smiles], np.ndarray):
                molecule_feature_dict[smiles] = protein_feature_dict[smiles].tolist()
            if isinstance(protein_feature_dict[protein], np.ndarray):
                molecule_feature_dict[protein] = protein_feature_dict[protein].tolist()
            combined = molecule_feature_dict[smiles] + protein_feature_dict[protein]

            #data seperation
            X.append(combined)

        X = np.array(X, dtype=float)
        if "affinity_score" in data.columns.values.tolist():
            y = self.affinity_score_extraction(self, data)
            return X, y
        else:
            return X
    
    def affinity_score_extraction(self, data):
        y = []
        for _, row in data.iterrows():
            affinity_score = row["affinity_score"]
            y.append(affinity_score)
        return np.array(y, dtype=float)

    def late_fusion(self):
        pass
    
    def mid_level_fusion(self):
        pass

    def cross_attention(self):
        pass

    def multimodel_learning(self):
        pass
###################################################


###############   Data processing   ###############
    def standard_scaling(self):
        pass

    def minmax_scaling(self):
        pass

    def pca(self):
        pass

    def train_test_split(self):
        pass
###################################################

#splitting the data in training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     #ONLY CHANGE THE TEST_SIZE BY PREFERENCE
#     X, y, test_size=0.33, random_state=42 
# )


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

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X = scaler.fit_transform(X)
# #X_predict = scaler.transform(X_predict)
# print("minmax scaling complete")

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

# model.fit(X_train, y_train)
# print("Train score:", model.score(X_train, y_train))
# print("Test score:", model.score(X_test, y_test))
print(model.__init__)

#FOR MAKING THE ACTUAL PREDICTIONS
"""
model.fit(X, y)
y_predict = model.predict(X_predict)
if len(y_predict) != 34626:
    raise IndexError(
        f"y_predict does not contain the same amount of samples as the test.csv file, but has {len(y_predict)} samles."
    )

submission = pd.DataFrame({
    "ID": test_df["ID"],
    "affinity_score": y_predict
})
submission.to_csv("data/submission1.csv", index=False)
"""