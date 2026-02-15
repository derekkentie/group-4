import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem

#showing data info
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
protein_info = pd.read_csv("data/protein_info.csv")
print("train:", '\n', train[:10], '\n', train.describe(), '\n')
print("test:", '\n',test[:10], '\n')
print("protein_info", '\n', protein_info[:10], '\n')

#extracting data into numpy arrays
train_array = np.array(pd.read_csv("data/train.csv", sep=',', header=None))
test_array = np.array(pd.read_csv("data/test.csv", sep=',', header=None))
protein_array = np.array(pd.read_csv("data/protein_info.csv", sep=',', header=None))

#creating the different lists needed to store the data in
affinity_score = []
index = []
error_sample = []
train_molecule_length = []
test_molecule_length = []
protein_length = []

train_molecule_uniques = []
test_molecule_uniques = []
unique_atoms_train = []
unique_atoms_test = []

#extracting the specific data and filtering out erroneous measurements
for sample in range(1, len(train_array)):
    if float(train_array[sample][-1]) < 40:
        print("erroneous samples:", '\n', 
              "index:", sample, '\n', 
              "SMILES: ", train_array[sample][0], '\n', 
              "Uniprot_ID: ", train_array[sample][1], '\n',
              "Affinity score (lower than 40): ", train_array[sample][2], '\n',
              "molecule length: ", len(train_array[sample][0]))
        
        #storing the erroneous measurements in a seperate list
        error_sample.append([sample, train_array[sample], len(train_array[sample][0])]) 
    else:
        #appending the right measurements to the index and affinity scores
        index.append(sample)
        affinity_score.append(float(train_array[sample][-1]))

        #obtaining train molecule length using RDkit
        mol = Chem.MolFromSmiles(train_array[sample][0])
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]    
        train_molecule_length.append(len(atom_symbols))
        train_molecule_uniques.append(np.unique(atom_symbols, return_index=True))

#obtaining test molecule length using RDkit
for sample in range(1, len(test_array)):
    mol = Chem.MolFromSmiles(test_array[sample][1])
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]    
    test_molecule_length.append(len(atom_symbols))
    test_molecule_uniques.append(np.unique(atom_symbols, return_index=True))

#obtaining protein length using RDkit
for protein in range(1, len(protein_array)):
    protein_length.append(len(protein_array[protein][2]))

#analyzing the unique atoms in each molecule for both the training and test set
for unique in range(len(train_molecule_uniques)):
    for atom in train_molecule_uniques[unique][0]:
        if atom not in unique_atoms_train:
            unique_atoms_train.append(str(atom))

for unique in range(len(test_molecule_uniques)):
    for atom in test_molecule_uniques[unique][0]:
        if atom not in unique_atoms_test:
            unique_atoms_test.append(str(atom))

print("unique atoms in training set", np.sort(unique_atoms_train))
print("unique atoms in test set", np.sort(unique_atoms_test))
    
#converting to numpy arrays
affinity_score = np.array(affinity_score)
index = np.array(index)
train_molecule_length = np.array(train_molecule_length)
test_molecule_length = np.array(test_molecule_length)
protein_length = np.array(protein_length)

#creating the subplots for every plot in the figure
fig, ([ax11train, ax12train, ax13train], 
      [ax21trainmollenviol, ax22testmollenviol, ax23protlenviol], 
      [ax31trainmollenbox, ax32testmollenbox, ax33protlenbox]) =    plt.subplots(nrows = 3, 
                                                                                ncols = 3,
                                                                                figsize =(12, 8),
                                                                                sharex = False,
                                                                                sharey = False)


#plotting the affinity scores in violinplot, boxplot and scatterplot
ax11train.set_ylabel('affinity scores of the test set')
ax11train.violinplot(affinity_score)
ax12train.boxplot(affinity_score)
sc = ax13train.scatter(index, affinity_score, c= affinity_score, cmap='viridis')
plt.colorbar(sc, ax = ax13train) #adding the colorbar for easier interpretation

#plotting the molecule and protein lengths in violin- and boxplot
ax21trainmollenviol.set_ylabel("violinplot of sample length")
ax31trainmollenbox.set_ylabel("boxplot of sample length")
ax21trainmollenviol.set_xlabel("trainingset")
ax22testmollenviol.set_xlabel("testset")
ax23protlenviol.set_xlabel("proteins")

#assigning the different plots in row 2 and 3
ax21trainmollenviol.violinplot(train_molecule_length)
ax31trainmollenbox.boxplot(train_molecule_length)

ax22testmollenviol.violinplot(test_molecule_length)
ax32testmollenbox.boxplot(test_molecule_length)

ax23protlenviol.violinplot(protein_length)
ax33protlenbox.boxplot(protein_length)

#properly separating the different plots
fig.subplots_adjust(hspace=0.2)
plt.tight_layout(pad=1)
plt.show()