#combining featurevectors from multiple sources.
import numpy as np
import pickle

nr_features = 45 #!!!NEEDS TO BE ADJUSTED MANUALLY, IS THE TOTAL AMOUNT OF FEATURES FOR EACH AMINO ACID!!!
ID_to_protein = pickle.load(open('dict ID to sequence','rb'))

featurevec1 = pickle.load(open('dict aminoacid to property vector 2','rb')) #<--- first dictionary from aa to featurevector goes here
featurevec2 = pickle.load(open('dict aa to PAM250 substitution vector (alfabetically)','rb')) #<--- second goes here

aa_to_featurevec = dict()

for aa in featurevec1.keys():
    aa_to_featurevec[aa] = np.concatenate([np.array(featurevec1[aa]), np.array(featurevec2[aa])])
   

ID_to_featurevec = dict()
ID_to_feature_mat = dict()
ID_codes = ID_to_protein.keys()

for ID in ID_codes:
    protein = ID_to_protein[ID]
    vec = np.array([])
    for aa in protein:
        featurevec = aa_to_featurevec[aa]
        vec = np.concatenate((vec,featurevec))

    ID_to_featurevec[ID] = vec
    ID_to_feature_mat[ID] = vec.reshape((len(vec)//nr_features,nr_features))

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
    dict_ID_to_propertymatrix_padded_with_nans[ID] = vec.reshape((len(vec)//nr_features,nr_features))

#outputfiles go here !!!RENAME TO NOT OVERWRITE THEM!!!
pickle.dump(ID_to_featurevec,open('dict ID to physicochemical 2 + PAM250 vector','wb')) 
pickle.dump(ID_to_feature_mat,open('dict ID to physicochemical 2 + PAM250 matrix','wb'))
pickle.dump(dict_ID_to_propertyvector_padded_with_nans,open('dict ID to physicochemical 2 + PAM250 vector padded with nans','wb'))
pickle.dump(dict_ID_to_propertymatrix_padded_with_nans,open('dict ID to physicochemical 2 + PAM250 matrix padded with nans','wb'))