import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import Descriptors


"""
EXPLANATION OF THE CODE

This function extracts the data from the test- and train.csv files that contain the molecules in SMILES representation,
after which it uses the rdkit library to calculate 217 molecular descriptors for each molecule.

The descriptors will be filtered down to reduce file size.

At last, the most useful descriptors will be used to train a ML model to predict protein-ligand affinity strength.
"""

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

This code generates two excel files in the data file map of the repository with extremely large file sizes, 
which results in the repository being too large to push to the remote.

The file sizes will be reduced later on when we reduce the descriptors to only the most valuable ones.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

"""
PLANNED IMPROVEMENTS ON THE CODE

    1. The reduction of descriptors based on the follwing factors:
        (i) removal of descriptors with constant values
        (ii) removal of descriptors with constant and near-constant variables
        (iii) removal of descriptors with a standard deviation of less  than 0.001
        (iv) removal of descriptors with at least one missing value
        (v) removal of descriptors with a pair correlation larger than or equal to 0.90

    2. The addition of a calculator for the fingerprint presentations of each molecule

    3. Reforming the code into a single function

    4. Applying data controls to make sure the model will not crash

    5. Universalizing the code for more data types

    6. Finding out how to append the descriptor names as header in X

    7. Applying feature scaling
"""


def descriptor_extractor_csv(file, batchpercentage = 1):
    """
    This function creates a csv file with all descriptors for 
    each SMILES represented molecule in the input file.
    """

    #making the list of functions for every descriptor calculation
    descriptor_functions = [d[1] for d in Descriptors._descList]

    #creating starting conditions for variables
    descriptors_per_mol = [] #used to store descriptor values per molecule
    last_progress = -1 #used to keep track on SMILES extraction and descriptor calcuation
    mols = [] #used to store the mol codes per SMILES represented molecule
    X =[] #used to store all descriptor values of every molecule to export to .xlsx file

    #checking if the sought after file is available
    try:
        data = pd.read_csv(f"{file}", sep=',') #searches for local file location and reads it into data
    except FileNotFoundError:
        (f"{file} not found, try again.")
        data = None

    #extracting the SMILES from the file specific "molecule_SMILES" column into mol code

    #the following if-else statement is made to allow for quicker calculations when debugging
    datasize = int(len(data)*batchpercentage)
    print(datasize)
    print(len(data))
    if batchpercentage == 1:
        index_list = range(1, datasize)
    else:
        index_list = random.sample(range(1, len(data)), datasize)

    #this function goes through every molecule (= row index), and proceeds to calculate the mol from the SMILES code that is in the "molecule_SMILES" column
    for molecule in index_list:       
        mol = Chem.MolFromSmiles(np.array(data)[molecule][list(data).index("molecule_SMILES")])
        mols.append(mol) 
        #keeping track of the descriptor calculation progress 
        progress_calculation = int(100*len(mols)/datasize) 
        if progress_calculation > last_progress:
                print(f"progress of SMILES extraction: {progress_calculation}%")
                last_progress = progress_calculation
    last_progress = -1

    #combining every mol with each descriptor calculation function
    for mol in mols:
        for func in descriptor_functions:
            value = func(mol)
            descriptors_per_mol.append(value) #using each descriptor function on the current mol code to create a list of descriptor values
        X.append(descriptors_per_mol) #appending the descriptors per molecule as row into X
        descriptors_per_mol = [] #clearing the descriptor list for to be filled for the next molecule

        #keeping track of the descriptor calculation progress 
        progress_calculation = int(100*len(X)/datasize) 
        if progress_calculation > last_progress:
            print(f"progress of descriptor calculation: {progress_calculation}%")
            last_progress = progress_calculation

    X = np.array(X, dtype=float)
    return np.savetxt("data/descriptors_extraction.xlsx", X, delimiter=",")

descriptor_extractor_csv("data/train.csv", batchpercentage=0.01 )