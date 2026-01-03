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

# #added by Sep:
# """This code is added to make it easier to select specific physicochemical properties to use to make the model. 
# It does so by first creating a dictionary coupling every aminoacid to a subsequent featurevector, after which new dictionaries are made coupling IDs to featurevectors or -matrices,
# make sure you keep the dictionaries you need for the model. If you want to use a picklefile directly for your model, feel free to use ctrl + / to silence the added code."""


# ID_to_protein = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict ID to sequence",'rb'))
# aa_to_featurevec_unfilter = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict aminoacid to property vector 2",'rb')) 

# dict_to_concatenate =None
# #Above PAM250 is selected, can be exchanged with BLOSUM62, make sure it is a dictionary coupling aminoacids to vectors, not ID's to vectors, can be set to None if you only want to use physicochemical properties

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

        # 1. Load representations
        mol_rep_dict_of_dicts = self.dicts_collector(
            map_location= "docs\mol representatie picklebestanden"
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
molecule_features_dict_train = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\mol representatie picklebestanden\train_molecule_descriptor_representation.pkl",'rb'))
molecule_features_dict_test = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\mol representatie picklebestanden\test_molecule_descriptor_representation.pkl",'rb'))
# protein_features_dict = pickle.load(open(r"docs\Sep's picklebestanden\dict ID to featurevector in one-hot padded with nans", 'rb'))

#generating dictionary for 'local averages' of protein
#these two lines can be changed to experiment with different features and hyperparameters
ID_to_mat_old = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict ID to BLOSUM62 matrix",'rb')) #must be a dictionary coupling ID's to matrices
nr_pieces = 16 #will probably run into semantic errors if it exceeds 146, as the code is not equiped for that, let me know if you want to try 
add_length = True

ID_to_protein = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict ID to sequence",'rb')) 
nr_features = len(ID_to_mat_old['P24941'][0]) #any featurevector for 1 aa would suffice
protein_features_dict = dict()
# ID_to_mat_new = dict()


for ID in ID_to_protein.keys():
    average_length= len(ID_to_protein[ID])/nr_pieces
    highest_index_processed = -1 #this does not mean the last index
    residue = 0
    vec_new = np.array([])
    for i in range(nr_pieces-1): #-1 because the last iteration is an edge case due to possible floating-point-inaccuracies            
        length_piece = int(average_length + residue)
        residue = average_length + residue - int(average_length + residue)
        aas_in_piece = ID_to_mat_old[ID][range(highest_index_processed+1, highest_index_processed+1+length_piece)]
        highest_index_processed = highest_index_processed+length_piece
        
        average_in_piece = np.average(aas_in_piece,axis=0)
        vec_new = np.concatenate((vec_new,average_in_piece))

    #last iteration
    aas_in_piece = ID_to_mat_old[ID][range(highest_index_processed+1,len(ID_to_protein[ID]))]
    average_in_piece = np.average(aas_in_piece,axis=0)
    vec_new = np.concatenate((vec_new,average_in_piece))

    #final processing
    if add_length:
        protein_features_dict[ID] = np.concatenate((vec_new,np.array([len(ID_to_protein[ID])])))
        # vec_extended = np.concatenate((vec_new,np.array([len(ID_to_protein[ID])]*nr_features))) #makes sure an entire extra row is added in the matrix containing just the lenghts
        # ID_to_mat_new[ID] = vec_extended.reshape(len(vec_extended)//nr_features,nr_features)
        
    else:
        protein_features_dict[ID] = vec_new
        # ID_to_mat_new[ID] = vec_new.reshape(len(vec_new)//nr_features,nr_features)
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
        protein_rep_dict_of_dicts = self.dicts_collector( #in the same indented block as the error?
            map_location= "docs/Sep's picklebestanden"
        )

        # 2. Loop over molecule & protein representations
        for (mol_rep_name, mol_rep_dict), (prot_rep_name, prot_rep_dict) in product(
            mol_rep_dict_of_dicts.items(),
            protein_rep_dict_of_dicts.items()
        ):
            # 3. Loop over representation-combination methods
            for comb_name, comb_fn in self.dict_of_representation_combination_methods.items():

                # 4. Loop over models
                for model_name in self.dict_of_model_builders.keys():

                    # 5. Loop over hyperparameter combinations (generator!)
                    for hyperparams in self.hyperparam_generator(model_name):

                    
                        experiments.append({
                            "molecule_representation_name": mol_rep_name,
                            "molecule_representation_dict": mol_rep_dict,

                            "protein_representation_name": prot_rep_name,
                            "protein_representation_dict": prot_rep_dict,

                            "combination_method_name": comb_name,
                            "combination_method_fn": comb_fn,

                            "model_name": model_name,
                            "hyperparameters": hyperparams
                        })

        return experiments
        
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
# X_train, X_test, y_train, y_test = train_test_split(
#     #ONLY CHANGE THE TEST_SIZE BY PREFERENCE
#     X, y, test_size=0.33, random_state=42 
# )


#BELOW ARE OPTIONS FOR SCALING AND PCA, REMOVE DOCSTRINGS FOR
#THE PREFERRED OPTION(S)

#choose one of the following scaling option, or leave them out if preferred

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_predict = scaler.transform(X_predict)
# print("standard scaling complete")


# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X = scaler.fit_transform(X)
# #X_predict = scaler.transform(X_predict)
# print("minmax scaling complete")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) #will this make sure the same minimum and maximum is used as in X_train?
X = scaler.fit_transform(X) #idem
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

# pca = PCA(n_components=8)
# X_train = pca.fit_transform(X_train)
# X_test  = pca.transform(X_test)
# print("PCA application complete")

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

# model = MLPRegressor(
#     hidden_layer_sizes=( 16, 8),
#     activation='logistic',
#     learning_rate='adaptive',
#     max_iter=400,
#     random_state=42
# )


model = RandomForestRegressor(
    criterion= 'absolute_error', #this is the criterion kaggle uses, but it is slow
    min_samples_split=2, #considering our large dataset, it might be useful to test bigger values, though based on my runs, it lowers the score significantly.
    min_samples_leaf=1, #idem
    max_leaf_nodes=None, #maybe this is helpful too
    n_estimators=10, #this value is very low and adds a handicap of sorts to the model. 
    # A nescesssity in my opinion to reduce computational cost for the gridsearch, please increase for the actual predictions.
    max_features='sqrt' #don't make this too high, or it will take too long.
)

# model.fit(X_train, y_train)
# print("Train score:", model.score(X_train, y_train))
# print("Test score:", model.score(X_test, y_test))
print(model.__init__)

# model = HistGradientBoostingRegressor(
#     loss= "absolute_error",
#     learning_rate= 0.1,
#     max_iter= 100
# )

model.fit(X_train, y_train)
print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))

#FOR MAKING THE ACTUAL PREDICTIONS

# model.fit(X, y)
# y_predict = model.predict(X_predict)
# if len(y_predict) != 34626:
#     raise IndexError(
#         f"y_predict does not contain the same amount of samples as the test.csv file, but has {len(y_predict)} samles."
#     )

# submission = pd.DataFrame({
#     "ID": test_df["ID"],
#     "affinity_score": y_predict
# })
# submission.to_csv("data/submission2.csv", index=False)