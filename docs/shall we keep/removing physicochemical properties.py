import numpy as np
import statistics
import pickle
# import scipy

dict_aa_to_featurevec = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict aminoacid to property vector 2",'rb'))
X = []
for vec in dict_aa_to_featurevec.values():
    list = vec.tolist()
    # del list[3]
    # del list[4]
    # del list[15]
    # del list[18]
    
    X.append(list)
# print(X)

def find_highly_correlated_descriptors(X, threshold=0.9):
        """
        X = list of lists (rows x columns)
        threshold = absolute correlation threshold for removing columns
        """
        print("correlation-filter function activated")
        arr = np.array(X, dtype=float)
        n_cols = arr.shape[1]
        print('n_cols=',n_cols)
        cols_to_remove = []

        for i in range(n_cols):
            # print('i=',i)
            for j in range(i):
                # print('j=',j)
                # bereken correlatie tussen kolom i en kolom j
                # print('arr[:, i]=',arr[:, i])
                # print('arr[:, j]=',arr[:, j])
                corr = np.corrcoef(arr[:, i], arr[:, j],rowvar=False)[0, 1]
                # print('corr=',corr)

                if abs(corr) > threshold:
                    cols_to_remove.append((i,j))
                    # print(i,'added')
                    #print(f"Removing {self.descriptor_names[i]} due to corr > {threshold} with {self.descriptor_names[j]}")
                    # break   # stop met vergelijken zodra 1 hoge correlatie gevonden is

        print("correlation-filter function ended")
        return cols_to_remove

def find_low_std_descriptors(X, threshold=0.001):
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

def find_missing_values(X):
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
print(find_highly_correlated_descriptors(X))