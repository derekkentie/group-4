#combining featurevectors from multiple sources.
import numpy as np
import pickle

ID_to_protein = pickle.load(open('dict ID to sequence','rb'))

aa_to_featurevec_unfilter = pickle.load(open('dict aminoacid to property vector 2','rb')) 
features_used = [0, 1, 2, 3, 4, 5, 6, 12, 13, 17, 18, 22, 23, 24]
aa_to_featurevec_filtered = dict()

for aa in aa_to_featurevec_unfilter.keys():
    value = aa_to_featurevec_unfilter[aa]
    features_kept = []
    for feature in features_used:
        features_kept.append(value[feature])
    aa_to_featurevec_filtered[aa] = features_kept
   
print(aa_to_featurevec_filtered['C'])
ID_to_featurevec = dict()
ID_to_feature_mat = dict()
ID_codes = ID_to_protein.keys()

for ID in ID_codes:
    protein = ID_to_protein[ID]
    vec = np.array([])
    for aa in protein:
        vec = np.concatenate((vec,aa_to_featurevec_filtered[aa]))

    ID_to_featurevec[ID] = vec
    ID_to_feature_mat[ID] = vec.reshape((len(vec)//len(features_used),len(features_used)))

#for padding with nans
lengths = [len(value) for value in ID_to_featurevec.values()]
max_length = max(lengths)
dict_ID_to_propertyvector_padded_with_nans = dict()
dict_ID_to_propertymatrix_padded_with_nans = dict()


for ID in ID_to_featurevec.keys():
    value = ID_to_featurevec[ID]
    nr_nans = max_length - len(value)
    nan_array = (np.array([float('nan')]*nr_nans))
    vec = np.concatenate((value,nan_array))
    dict_ID_to_propertyvector_padded_with_nans[ID] = vec
    dict_ID_to_propertymatrix_padded_with_nans[ID] = vec.reshape((len(vec)//len(features_used),len(features_used)))
