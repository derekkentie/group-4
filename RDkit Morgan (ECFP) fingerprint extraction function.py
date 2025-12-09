import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

def fingerprint_extractor_csv_to_xlsx(file: str, batchpercentage: float = 1):
    """
    Returns an excel file with fingerpints calculated for molecules.

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
    
    #creating starting conditions for variables
    fingerprints_per_mol = [] #used to store descriptor values per molecule
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
        generator = GetMorganGenerator(radius=2,
                                        countSimulation=True,     # count-based fingerprints
                                        includeChirality=False,
                                        useBondTypes=True,
                                        onlyNonzeroInvariants=False,
                                        includeRingMembership=True,
                                        fpSize=1024
                                        )
        fingerprints_per_mol = generator.GetFingerprint(mol)
        X.append(fingerprints_per_mol) #appending the descriptors per molecule as row into X
        fingerprints_per_mol = [] #clearing the descriptor list for to be filled for the next molecule

        #keeping track of the descriptor calculation progress 
        progress_calculation = int(100*len(X)/datasize) 
        if progress_calculation > last_progress:
            print(f"progress of fingerprint calculation: {progress_calculation}%")
            last_progress = progress_calculation

    #changing the array of fingerprints to a propper numpy array
    
    X = [fp_to_list(fp) for fp in X]

    #removing the fingerprints that contain constant values
    constant_fingerprints = find_constant_values(X)
    for fingerprint in sorted(constant_fingerprints, reverse=True):
        X = [row[:fingerprint] + row[fingerprint+1:] for row in X]

    X = np.array(X, dtype=int)
    n_mol, n_fp = X.shape
    print("number of molecules in the document:", n_mol)
    print("number of fingerprints in the document:", n_fp)
    return np.savetxt("data/Morgan_fingerprint_extraction.xlsx", X, delimiter=",", fmt= "%d") #returning the calculated matrix in an excel file

def find_constant_values(X):
    arr = np.array(X, dtype= float)
    n_col = arr.shape[1]
    constant_fingerprints = []

    for col in range(n_col):
        if len(np.unique(arr[:,col])) == 1:
            constant_fingerprints.append(col)
    return constant_fingerprints

def fp_to_list(fp):
    n_bits = fp.GetNumBits()
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr.tolist()

fingerprint_extractor_csv_to_xlsx("data/train.csv", batchpercentage=0.01)