import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#showing data info
train = pd.read_csv(r"C:\Users\20234364\group-4\data\train.csv")
test = pd.read_csv(r"C:\Users\20234364\group-4\data\test.csv")
protein_info = pd.read_csv(r"C:\Users\20234364\group-4\data\protein_info.csv")
print("train:", '\n', train[:10], '\n', train.describe(), '\n')
print("test:", '\n',test[:10], '\n')
print("protein_info", '\n', protein_info[:10], '\n')

# Extracting data into numpy arrays
train_array = np.array(pd.read_csv(r"C:\Users\20234364\group-4\data\train.csv", sep=',', header=None))
test_array = np.array(pd.read_csv(r"C:\Users\20234364\group-4\data\test.csv", sep=',', header=None))
protein_array = np.array(pd.read_csv(r"C:\Users\20234364\group-4\data\protein_info.csv", sep=',', header=None))

#creating the different lists needed to store the data in
affinity_score = []
index = []
error_sample = []
train_molecule_length = []
test_molecule_length = []
protein_length = []

#extracting the specific data and filtering out erroneous measurements
for sample in range(1, len(train_array)):
    if float(train_array[sample][-1]) < 40:
        print("erroneous sample:", '\n', 
              "index:", sample, '\n', 
              "SMILES: ", train_array[sample][0], '\n', 
              "Uniprot_ID: ", train_array[sample][1], '\n',
              "Affinity score (lower than 40): ", train_array[sample][2], '\n',
              "molecule length: ", len(train_array[sample][0]))
        
        # Storing the erroneous measurements in a seperate list
        error_sample.append([sample, train_array[sample], len(train_array[sample][0])]) 
    else:
        # Appending the right measurements to the index and affinity scores
        index.append(sample)
        affinity_score.append(float(train_array[sample][-1]))

        # Train molecule length storage
        train_molecule_length.append(len(train_array[sample][0]))

# Test molecule length storage
for sample in range(1, len(test_array)):
    test_molecule_length.append(len(test_array[sample][1]))

# Protein length storage
for protein in range(1, len(protein_array)):
    protein_length.append(len(protein_array[protein][2]))

#converting to numpy arrays
affinity_score = np.array(affinity_score)
index = np.array(index)
train_molecule_length = np.array(train_molecule_length)
test_molecule_length = np.array(test_molecule_length)
protein_length = np.array(protein_length)

# plot the image
fig, ([ax11train, ax12train, ax13train], 
      [ax21trainmollenviol, ax22testmollenviol, ax23protlenviol], 
      [ax31trainmollenbox, ax32testmollenbox, ax33protlenbox]) =    plt.subplots(nrows = 3, 
                                                                                ncols = 3,
                                                                                figsize =(12, 8),
                                                                                sharex = False,
                                                                                sharey = False)


# plotting the affinity scores in violinplot, boxplot and scatterplot
ax11train.set_ylabel('affinity scores of the test set')
ax11train.violinplot(affinity_score)
ax12train.boxplot(affinity_score)
sc = ax13train.scatter(index, affinity_score, c= affinity_score, cmap='viridis')
plt.colorbar(sc, ax = ax13train)

# plotting the molecule and protein lengths in violin- and boxplot
ax21trainmollenviol.set_ylabel("violinplot of sample length")
ax31trainmollenbox.set_ylabel("boxplot of sample length")
ax21trainmollenviol.set_xlabel("trainingset")
ax22testmollenviol.set_xlabel("testset")
ax23protlenviol.set_xlabel("proteins")
ax21trainmollenviol.violinplot(train_molecule_length)
ax31trainmollenbox.boxplot(train_molecule_length)

ax22testmollenviol.violinplot(test_molecule_length)
ax32testmollenbox.boxplot(test_molecule_length)

ax23protlenviol.violinplot(protein_length)
ax33protlenbox.boxplot(protein_length)

fig.subplots_adjust(hspace=0.2)
plt.tight_layout(pad=1)
plt.show()