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
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


class gridsearch:
    """
    This program is made to perform a gigantic gridsearch, not ontly on model-
    specific hyperparameters, but on all the different possible combinations
    between: molecule and protein representation, representation-combination
    method, scaling method, PCA options, ML model, and ML model specific
    hyperparameters. The program will then store all its findings in an excel
    with columns: train_score, test_score, combination.

    IT IS STRONGLY ADVISED TO NOT RUN THIS PROGRAM ON YOUR 
    PERSONAL LAPTOP DUE TO EXTREME AMOUNT OF COMPUTER CALCULATION!
    
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
            "Random Forest": self.build_random_forest,
            "Gradient Boosting Regressor": self.build_gbr,
            "Hist Gradient Boosting Regressor": self.build_hgbr,
            "Support Vector Regression": self.build_svr,
            "Multi-Layer Perceptron": self.build_mlp,
            "K Nearest Neighbors": self.build_knn,
        }

        self.dict_of_model_hyperparams = {
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
            "scaler": [None, self.standard_scaling, self.minmax_scaling],
            "pca": [None, 1, 2, 3, 4]
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
        return list(representation_folder.iterdir())

    def pickle_converter(self, pickle_file_path):
        """
        Returns the unencoded/unpickled document of pickle_file_path.
        """
        with pickle_file_path.open("rb") as pickle_file_handle:
            return pickle.load(pickle_file_handle)

    def dicts_collector(self, folder_location):
        """
        Returns a dictionary of dictionaries from both the molecule and
        protein representations, with key equal to file_path_name and item
        equal to the respective representation dicitionary which is 
        retreived using the pickle_converter function.
        """
        pickle_file_paths = self.representation_gridsearch(rf"{folder_location}")
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
        grid = self.dict_of_model_hyperparams[model_name]
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
 
    def experiment_maker(self, one_or_all_combination_method = "feature concatenation"):
        """
        Returns every possible combination between de representations,
        representation-combination methods, models and hyperparameters,
        in a list of dictionaries.
        """
        experiments = []

        #selection for one or all combination methods, feature concatenation is now used 
        if one_or_all_combination_method == "all":
            comb_method = self.dict_of_representation_combination_methods.items()
        else:
            comb_method = [(one_or_all_combination_method, self.dict_of_representation_combination_methods[one_or_all_combination_method])]

        # 1. Load representations
        mol_rep_dict_of_dicts = self.dicts_collector(
            folder_location= "docs/mol representatie picklebestanden"
        )
        protein_rep_dict_of_dicts = self.dicts_collector(
            folder_location= "docs/Sep's picklebestanden/protein dicts to use in gridsearch"
        )

        # 2. Loop over molecule & protein representations
        for (mol_rep_name, mol_rep_dict), (prot_rep_name, prot_rep_dict) in product(
            mol_rep_dict_of_dicts.items(),
            protein_rep_dict_of_dicts.items()
        ):
            # 3. Loop over representation-combination methods
            for comb_name, comb_function in comb_method:

                # 4. Loop over models
                for model_name in self.dict_of_model_builders.keys():

                    # 5. Loop over hyperparameter combinations
                    for hyperparams in self.hyperparam_generator(model_name):
                        
                        # 6. Loop over scaling types
                        for scale_type in self.dict_of_data_processors["scaler"]:

                            # 7. Loop over pca range from None to 4 PC's
                            for pca in self.dict_of_data_processors["pca"]:

                                # 7. Appending every possible combination as a dictionary in the list experiments
                                experiments.append({
                                    "molecule_representation_name": mol_rep_name,
                                    "molecule_representation_dict": mol_rep_dict,

                                    "protein_representation_name": prot_rep_name,
                                    "protein_representation_dict": prot_rep_dict,

                                    "combination_method_name": comb_name,
                                    "combination_method_function": comb_function,

                                    "model_name": model_name,
                                    "hyperparameters": hyperparams,

                                    "scale_type": scale_type,
                                    "pca": pca
                                })

        return experiments
    
    def init_results_csv(self, output_csv_path):
        output_csv_path = Path(output_csv_path)

        if not output_csv_path.exists():
            print(f"creating {output_csv_path} as new path")
            df = pd.DataFrame(columns=[
                "mol_representation",
                "combination_method",
                "model",
                "hyperparams",
                "seed",
                "train_score",
                "train_accuracy",
                "test_score",
                "test_accuracy",
                "error"
            ])
            df.to_csv(output_csv_path, index=False)
        else:
            print(f"{output_csv_path} already exists")
    
    def experiment_tester(self, output_csv_path, data, debug_fraction_selector = 1):
        experiments = self.experiment_maker()
        self.init_results_csv(output_csv_path)
        results = []
        for i, experiment in enumerate(experiments, start=1):

            row = {
                "mol_representation":       experiment["molecule_representation_name"],
                "protein_representation":   experiment["protein_representation_name"],
                "combination_method":       experiment["combination_method_name"],
                "model":                    experiment["model_name"],
                "hyperparams":              str(experiment["hyperparameters"]),
                "seed":                     experiment.get("seed", None),
                "train_score":              "None",
                "train_accuracy":           "None",
                "test_score":               "None",
                "test_accuracy":            "None",
                "error":                    "None"
            }
            if i/debug_fraction_selector == int(i/debug_fraction_selector):
                try:
                    model_type = self.dict_of_model_builders[experiment["model_name"]]
                    model = model_type(experiment["hyperparameters"])
                    
                    combination_method = self.dict_of_representation_combination_methods[experiment["combination_method_name"]]
                    X, y = combination_method(experiment["molecule_representation_dict"], experiment["protein_representation_dict"], data)

                    X_train, y_train, X_test, y_test = self.train_test_split(X, y)

                    if self.dict_of_data_processors["scaler"] != None:
                        X_train, X_test = self.dict_of_data_processors["scaler"](X_train, X_test)

                    if self.dict_of_data_processors["pca"] != None:
                        X_train, X_test = self.pca(X_train, X_test)

                    model.fit(X_train, y_train)

                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    row["train_score"] = model.score(X_train, y_train)
                    row["train_accuracy"] = self.accuracy(y_train, y_train_pred)

                    row["test_score"] = model.score(X_test, y_test)
                    row["test_accuracy"] = self.accuracy(y_test, y_test_pred)



                except Exception as e:
                    row["error"] = str(e)

                if i % 100 == 0:
                    print(f"{i}/{len(experiments)} experiments calculated")
            
                pd.DataFrame([row]).to_csv(
                output_csv_path,
                mode="a",
                header=False,
                index=False
                )

                results.append(row)

        return results

        
###############    Model builders   ###############
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
        y = []

        #feature concatenation to combining each ligand-protein pair
        for _, row in data.iterrows():
            smiles = row["molecule_SMILES"]
            protein = row["UniProt_ID"]
            affinity_score = row["affinity_score"]

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

            #making sure that both representations are in list form
            

            if isinstance(molecule_feature_dict[smiles], np.ndarray):
                molecule_feature_dict[smiles] = molecule_feature_dict[smiles].tolist()
            if isinstance(protein_feature_dict[protein], np.ndarray):
                protein_feature_dict[protein] = protein_feature_dict[protein].tolist()

            combined = molecule_feature_dict[smiles] + protein_feature_dict[protein]

            #data seperation
            X.append(combined)
            y.append(affinity_score)
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        return X, y

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
    def standard_scaling(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def minmax_scaling(self, X_train, X_test):
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def pca(self, X_train, X_test):
        pca = PCA(self.dict_of_data_processors["pca"])
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
        return X_train, X_test

    def train_test_split(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)
###################################################

    def accuracy(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(y_true == y_pred)


gridsearch = gridsearch()
data = gridsearch.data_loader("data/train.csv")
results = gridsearch.experiment_tester("docs/model_gridsearch.csv", data, debug_fraction_selector= 10000)

print(len(results))
print(results[6])