import numpy as np
import pandas as pd
import random
import statistics
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler


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


def descriptor_extractor_csv_to_xlsx(file: str, batchpercentage: float = 1):
    """
    Returns an excel file with descriptors calculated for molecules.

    This function looks at the SMILES representation of a molecule in a csv file
    and uses the RDkit library to convert these SMILES to mol representations,
    after which it uses the same library to calculate every discriptor for
    every molecule.
    """
    # check file extension
    if not file.lower().endswith(".csv"):
        raise ValueError(
            f"Invalid file type: '{file}'. This function only accepts .csv files."
        )
    
    #making the list of functions for every descriptor calculation
    descriptor_names = [d[0] for d in Descriptors._descList]
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

    #this function goes through every molecule (= row index), 
    #and proceeds to calculate the mol from the SMILES code 
    #that is in the "molecule_SMILES" column
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

    #applying correlation check to further filter out descriptors
    descriptors_to_remove = np.unique(find_low_std_descriptors(X) + find_highly_correlated_descriptors(X) + find_missing_values(X))

    print(descriptors_to_remove, '\n',
        "amount of descriptors:", len(descriptor_functions), '\n', 
        "amount of descriptors to remove:", len(descriptors_to_remove), '\n',
        "sum of descriptors left over:", (len(descriptor_functions)-len(descriptors_to_remove))
        )


    for descriptor in sorted(descriptors_to_remove, reverse=True):
        X = [row[:descriptor] + row[descriptor+1:] for row in X]
    print(np.array(X).shape)
    X = standard_scaling(X)
    X = np.array(X, dtype=float)
    print(X.shape)

    return np.savetxt("data/descriptors_extraction.xlsx", X, delimiter=",") #returning the calculated matrix in an excel file

def find_highly_correlated_descriptors(X, threshold=0.9):
    """
    X = list of lists (rows x columns)
    threshold = absolute correlation threshold for removing columns
    """
    descriptor_names = [d[0] for d in Descriptors._descList]
    arr = np.array(X, dtype=float)
    n_cols = arr.shape[1]
    cols_to_remove = []

    for i in range(n_cols):
        for j in range(i):
            # bereken correlatie tussen kolom i en kolom j
            corr = np.corrcoef(arr[:, i], arr[:, j])[0, 1]

            if abs(corr) > threshold:
                cols_to_remove.append(i)
                print(f"Removing {descriptor_names[i]} due to corr > {threshold} with {descriptor_names[j]}")
                break   # stop met vergelijken zodra 1 hoge correlatie gevonden is

    return cols_to_remove

def find_low_std_descriptors(X, threshold=0.001):
    descriptor_names = [d[0] for d in Descriptors._descList]
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
            print(f"Removing {descriptor_names[descriptor]} due to too low standard deviation")
        values_per_descriptor=[]
    return descriptors_to_remove

def find_missing_values(X):
    descriptor_names = [d[0] for d in Descriptors._descList]
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
                print(f"Removing {descriptor_names[descriptor]} due to missing value(s)")
        values_per_descriptor=[]
    return descriptors_to_remove

def standard_scaling(X):
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

    return arr.tolist()

def minmax_scaling(X):
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

    return arr.tolist()

descriptor_extractor_csv_to_xlsx("data/train.csv", batchpercentage=0.01)