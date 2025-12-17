import numpy as np
import pandas as pd
import pickle
import random
import statistics
from typing import Literal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

class MoleculeRepresentationGenerator:
    def __init__(self, 
                 rep_type: Literal['combined', 'descriptor', 'fingerprint'] = "combined", 
                 scale_type: Literal['standard', 'minmax', 'None'] = 'standard', 
                 pca = 0, 
                ):
        self.rep_type = rep_type
        self.scale_type = scale_type
        self.pca = pca
        self.last_progress = -1

        self.descriptor_names = [d[0] for d in Descriptors._descList]
        self.descriptor_functions = [d[1] for d in Descriptors._descList]

 
    # MAIN FUNCTIONS

    def data_loader(self, file):
        """
        returns data from the input file as pandas DataFrame
        and the index_list based on the datasize.
        """
        print("data loader function activated")
        #check file extension
        if not file.lower().endswith(".csv"):
            raise ValueError(
                f"Invalid file type: '{file}'. This function only accepts .csv files."
        )

        #checking file availability
        try:
            data = pd.read_csv(f"{file}", sep=',') #searches for local file location and reads it into data
        except FileNotFoundError:
            (f"{file} not found, try again.")

        return data
    
    def smiles_to_mols(self, unique_smiles):
        print("mol extraction function activated")
        mols = []
        for smiles in unique_smiles:
            mol = Chem.MolFromSmiles(smiles)
            mols.append(mol)
            self.progress_tracker(mols, "mols")
        return mols

    def get_rep_dict(self, file):
        print("representation dictionary function activated")
        data = self.data_loader(file)
        unique_smiles = self.unique_smiles(data)
        self.datasize = len(unique_smiles)
        mols = self.smiles_to_mols(unique_smiles)
        rep_dict = {}

        #applying the chosen representation function
        rep = self.rep_type_selector(mols)
        #applying PCA if prefered
        if self.pca > 0:
            rep = self.PCA(rep)

        for i in range(len(unique_smiles)):
            smile = unique_smiles[i]
            rep_dict[smile] = rep[i]

        return rep_dict
    
    def pickle_export(self, rep_dict):
        print("pickle export function activated")
        scale_title=''
        pca_title = ''
        if self.scale_type.lower() != 'none':
            scale_title = f'_{self.scale_type}_scaling'
        if self.pca >0:
            pca_title = f'_{self.pca}_principal_components'
        with open(f"docs/mol representatie picklebestanden/molecule_{self.rep_type}_representation{scale_title}{pca_title}.pkl", "wb") as export_file:
                pickle.dump(rep_dict, export_file)
                export_file.close()      


    # REPRESENTATION FUNCTIONS
    def rep_type_selector(self, mols):
        if self.rep_type == 'descriptor':
            rep = self.descriptor_rep(mols)
        elif self.rep_type == 'fingerprint':
            rep = self.fingerpint_rep(mols)
        elif self.rep_type == 'combined':
            rep = self.combined_rep(mols)
        else:
            TypeError(
            f"invalid representation type: {self.rep_type}. Choose between 'descriptor', 'fingerprint' or 'combined'."                
            )
        return rep

    def descriptor_rep(self, mols):
        print("descriptor function activated")
        descriptors_per_mol = []
        rep = []

        #combining every mol with each descriptor calculation function
        for mol in mols:
            for func in self.descriptor_functions:
                value = func(mol)
                descriptors_per_mol.append(value) #using each descriptor function on the current mol code to create a list of descriptor values
            rep.append(descriptors_per_mol) #appending the descriptors per molecule as row into X
            descriptors_per_mol = [] #clearing the descriptor list for it to be filled for the next molecule
            self.progress_tracker(rep, "descriptor")
        
        descriptors_to_remove = np.unique(self.find_low_std_descriptors(rep) + self.find_highly_correlated_descriptors(rep) + self.find_missing_values(rep))

        #temporary conversion to DataFrame for easier deletion of descriptors
        rep = pd.DataFrame(rep)
        for descriptor in sorted(descriptors_to_remove, reverse=True):
            del rep[descriptor]
        
        rep_reduced = []
        for i in range(len(rep)):
            rep_reduced.append(rep.iloc[i].tolist())

        #applying the chosen scale type
        if self.scale_type.lower() == 'Standard':
            rep_reduced = self.standard_scaling(rep_reduced)
        elif self.scale_type.lower() == 'minmax':
            rep_reduced = self.minmax_scaling(rep_reduced)
        elif self.scale_type.lower() == 'none':
            rep_reduced = rep_reduced
        else:
            TypeError(
            f"invalid scale type: {self.scale_type}. Choose between 'standard', 'minmax' or 'None'."                
            )
        

        print(  "Descriptor reduction complete:", '\n',
                "amount of descriptors:", len(self.descriptor_functions), '\n', 
                "amount of descriptors to removed:", len(descriptors_to_remove), '\n',
                "amount of descriptors left over:", (len(self.descriptor_functions)-len(descriptors_to_remove))
             )

        print("Descriptor representation shape:", np.array(rep_reduced).shape)

        return rep_reduced
    
    def fingerpint_rep(self, mols):
        print("fingerprint function activated")
        fingerprints_per_mol = []
        rep = []
        
        generator =  GetMorganGenerator(radius=2,
                                        countSimulation=True,     # count-based fingerprints
                                        includeChirality=False,
                                        useBondTypes=True,
                                        onlyNonzeroInvariants=False,
                                        includeRingMembership=True,
                                        fpSize=1024
                                        )
        
        for mol in mols:
            fingerprints_per_mol = generator.GetFingerprint(mol)
            rep.append(fingerprints_per_mol) #appending the descriptors per molecule as row into X
            amount_of_fingerprints = len(fingerprints_per_mol)
            fingerprints_per_mol = [] #clearing the descriptor list for to be filled for the next molecule
            self.progress_tracker(rep, "fingerprint")

        #changing the array of fingerprints to a propper numpy array
    
        rep = [self.fp_to_list(fp) for fp in rep]

        #removing the fingerprints that contain constant values
        constant_fingerprints = self.find_constant_values(rep)

        rep = pd.DataFrame(rep)
        for fingerprint in sorted(constant_fingerprints, reverse=True):
            del rep[fingerprint]

        rep_reduced = []
        for i in range(len(rep)):
            rep_reduced.append(rep.iloc[i].tolist())

        print("Fingerprint reduction complete:", '\n',
              "amount of fingerprints:", amount_of_fingerprints, '\n',
              "amount of fingerprints removed:", len(constant_fingerprints), '\n',
              "amount of fingerprints left over:", amount_of_fingerprints - len(constant_fingerprints)
              )
        
        print("Fingerprint representation type:", type(rep_reduced), '\n',
            "Fingerprint representation shape:", np.array(rep_reduced).shape)

        return rep_reduced
    
    def combined_rep(self, mols):
        print("combined function activated")
        descriptors = self.descriptor_rep(mols)
        fingerprints = self.fingerpint_rep(mols)
        rep = []
        if len(descriptors) == len(fingerprints):
            for row in range(len(descriptors)):
                rep.append(descriptors[row] + fingerprints[row])
                self.progress_tracker(rep, "combined")
        print("Feature combination complete:", '\n',
              "Total amount of features left over:", (np.array(rep).shape)
              )
        return rep
    

    # HELPER FUNCTIONS

    def find_highly_correlated_descriptors(self, X, threshold=0.9):
        """
        X = list of lists (rows x columns)
        threshold = absolute correlation threshold for removing columns
        """
        print("correlation-filter function activated")
        arr = np.array(X, dtype=float)
        n_cols = arr.shape[1]
        cols_to_remove = []

        for i in range(n_cols):
            for j in range(i):
                # bereken correlatie tussen kolom i en kolom j
                corr = np.corrcoef(arr[:, i], arr[:, j])[0, 1]

                if abs(corr) > threshold:
                    cols_to_remove.append(i)
                    #print(f"Removing {self.descriptor_names[i]} due to corr > {threshold} with {self.descriptor_names[j]}")
                    break   # stop met vergelijken zodra 1 hoge correlatie gevonden is

        return cols_to_remove

    def find_low_std_descriptors(self, X, threshold=0.001):
        print("low-stdev function activated")
        arr = np.array(X, dtype=float)
        n_rows, n_cols = arr.shape
        #filtering the descriptors based on functionality
        values_per_descriptor = []
        descriptors_to_remove = []

        for descriptor in range(n_cols):
            for value in range(n_rows):
                values_per_descriptor.append(X[value][descriptor])

            #filtering out the descriptors that are constant or have a stdev < threshold
            if statistics.stdev(values_per_descriptor) < threshold: 
                descriptors_to_remove.append(descriptor)
                #print(f"Removing {self.descriptor_names[descriptor]} due to too low standard deviation")
            values_per_descriptor=[]
        return descriptors_to_remove

    def find_missing_values(self, X):
        print("missing value function activated")
        arr = np.array(X, dtype=float)
        n_rows, n_cols = arr.shape
        values_per_descriptor = []
        descriptors_to_remove = []

        for descriptor in range(n_cols):
            for value in range(n_rows):
                values_per_descriptor.append(X[value][descriptor])

            uniques = np.unique(values_per_descriptor)
            if 0 in uniques: 
                # 1. Boolean check: exact set {0,1}
                if set(uniques.tolist()) == {0, 1}:
                    continue  # boolean → toegestaan
                # 2. Integer-only check
                elif all(float(u).is_integer() for u in uniques):
                    continue  # integer descriptor → toegestaan
                else:
                    descriptors_to_remove.append(descriptor)
                    #print(f"Removing {self.descriptor_names[descriptor]} due to missing value(s)")
            values_per_descriptor=[]
        return descriptors_to_remove

    def standard_scaling(self, X):
        print("standard scaling function activated")
        arr = np.array(X, dtype=float)
        n_rows, n_cols = arr.shape
        scaler = StandardScaler()

        for descriptor in range(n_cols):
            # haal kolom op
            column = arr[:, descriptor].reshape(-1, 1)

            # scale kolom
            scaled_column = scaler.fit_transform(column).flatten()

            # schrijf terug naar matrix
            for i in range(n_rows):
                arr[i, descriptor] = scaled_column[i]
        print("Standard scaling complete.")
        return arr.tolist()
                
    def minmax_scaling(self, X):
        print("minmax scaling function activated")
        arr = np.array(X, dtype=float)
        n_rows, n_cols = arr.shape

        for descriptor in range(n_cols):
            # haal kolom
            column = arr[:, descriptor]

            col_min = column.min()
            col_max = column.max()

            scaled_column = (column - col_min) / (col_max - col_min)

            # schrijf schaalwaarden terug naar matrix
            for i in range(n_rows):
                arr[i, descriptor] = scaled_column[i]
        print("min-max scaling complete.")
        return arr.tolist()
    
    def find_constant_values(self, X):
        print("constant value function activated")
        arr = np.array(X, dtype= float)
        n_col = arr.shape[1]
        constant_fingerprints = []

        for col in range(n_col):
            if len(np.unique(arr[:,col])) == 1:
                constant_fingerprints.append(col)
        return constant_fingerprints

    def fp_to_list(self, fp):
        n_bits = fp.GetNumBits()
        arr = np.zeros((n_bits,), dtype=int)
        ConvertToNumpyArray(fp, arr)
        return arr.tolist()

    def PCA(self, X):
        print("PCA function activated")
        pca = PCA(n_components = self.pca)
        X_pca = pca.fit_transform(X)
        print(f"Principal Component Analysis complete with {self.pca} components.")
        return X_pca

    def progress_tracker(self, X, function: str):
        #keeping track of the descriptor calculation progress 
        progress_calculation = int(100*len(X)/self.datasize) 
        if progress_calculation > self.last_progress:
            #print(f"progress of {function} calculation: {progress_calculation}%")
            self.last_progress = progress_calculation
        if len(X) == self.datasize:
            self.last_progress = -1

    def unique_smiles(self, data):
        return data["molecule_SMILES"].unique()
    
mol_feature_model = MoleculeRepresentationGenerator(rep_type= 'descriptor',scale_type= 'None')
molecule_features = mol_feature_model.get_rep_dict('data/train.csv')
mol_feature_model.pickle_export(molecule_features)