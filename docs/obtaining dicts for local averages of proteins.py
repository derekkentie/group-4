import numpy as np
import pickle

#these two lines can be changed to experiment with different features and hyperparameters
ID_to_mat_old = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict ID to feature matrix 2",'rb')) #must be a dictionary coupling ID's to matrices
nr_pieces = 16 #will probably run into semantic errors if it exceeds 146, as the code is not equiped for that, let me know if you want to try 
add_length = True

ID_to_protein = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict ID to sequence",'rb')) 
nr_features = len(ID_to_mat_old['P24941'][0]) #any featurevector for 1 aa would suffice
ID_to_vec_new = dict()
ID_to_mat_new = dict()


for ID in ID_to_protein.keys():
    average_length= len(ID_to_protein[ID])/nr_pieces
    highest_index_processed = -1 #this does not mean the last index
    residue = 0
    vec_new = np.array([])
    for i in range(nr_pieces-1): #-1 because the last iteration is an edge case due to possible floating-point-inaccuracies            
        length_piece = int(average_length + residue)
        residue = average_length + residue - int(average_length + residue)
        aas_in_piece = ID_to_mat_old[ID][range(highest_index_processed+1, highest_index_processed+1+length_piece)]
        highest_index_processed = highest_index_processed+length_piece
        
        average_in_piece = np.average(aas_in_piece,axis=0)
        vec_new = np.concatenate((vec_new,average_in_piece))

    #last iteration
    aas_in_piece = ID_to_mat_old[ID][range(highest_index_processed+1,len(ID_to_protein[ID]))]
    average_in_piece = np.average(aas_in_piece,axis=0)
    vec_new = np.concatenate((vec_new,average_in_piece))

    #final processing
    if add_length:
        ID_to_vec_new[ID] = np.concatenate((vec_new,np.array([len(ID_to_protein[ID])])))
        vec_extended = np.concatenate((vec_new,np.array([len(ID_to_protein[ID])]*nr_features))) #makes sure an entire extra row is added in the matrix containing just the lenghts
        ID_to_mat_new[ID] = vec_extended.reshape(len(vec_extended)//nr_features,nr_features)
        
    else:
        ID_to_vec_new[ID] = vec_new
        ID_to_mat_new[ID] = vec_new.reshape(len(vec_new)//nr_features,nr_features)
print(len(ID_to_mat_new["O60674"].flatten()))

# #back-up
# ID_to_protein = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict ID to sequence",'rb'))
# aa_to_featurevec_unfilter = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict aminoacid to property vector 2",'rb')) 

# dict_to_concatenate =None
# #Above PAM250 is selected, can be exchanged with BLOSUM62, make sure it is a dictionary coupling aminoacids to vectors, not ID's to vectors, can be set to None if you only want to use physicochemical properties


# features_used = [0, 1, 2, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] #contains indices of physicochemical properties kept, feel free to experiment by changing what it contains, do not exceed 24 as there are no properties with a higher index
# #FYI: these are the highly correlating pairs (abs(R) > 0.9): [(3, 2), (5, 2), (5, 3), (17, 15), (21, 2), (21, 3)]
# aa_to_featurevec_filtered = dict()

# for aa in aa_to_featurevec_unfilter.keys(): #creating dictionary coupling aminoacids to featurevectors
#     value = aa_to_featurevec_unfilter[aa]
#     features_kept = []
#     for feature in features_used:
#         features_kept.append(value[feature])
#     if dict_to_concatenate is not None:
#         features_kept = np.concatenate([features_kept,dict_to_concatenate[aa]])
#     aa_to_featurevec_filtered[aa] = features_kept

# #creating dictionaries coupling IDs to featurevectors or -matrices   
# ID_to_featurevec = dict()
# ID_to_feature_mat = dict()
# ID_codes = ID_to_protein.keys()

# for ID in ID_codes:
#     protein = ID_to_protein[ID]
#     vec = np.array([])
#     for aa in protein:
#         vec = np.concatenate((vec,aa_to_featurevec_filtered[aa]))

#     ID_to_featurevec[ID] = vec
#     ID_to_feature_mat[ID] = vec.reshape((len(vec)//len(aa_to_featurevec_filtered['A']),len(aa_to_featurevec_filtered['A'])))

# #creating dictionaries coupling IDs to featurevectors or -matrices padded with nans
# lengths = [len(value) for value in ID_to_featurevec.values()]
# max_length = max(lengths)
# dict_ID_to_propertyvector_padded_with_nans = dict()
# dict_ID_to_propertymatrix_padded_with_nans = dict()


# for ID in ID_to_featurevec.keys():
#     value = ID_to_featurevec[ID]
#     nr_nans = max_length - len(value)
#     nan_array = (np.array([float('nan')]*nr_nans))
#     vec = np.concatenate((value,nan_array))
#     dict_ID_to_propertyvector_padded_with_nans[ID] = vec
#     dict_ID_to_propertymatrix_padded_with_nans[ID] = vec.reshape((len(vec)//len(aa_to_featurevec_filtered['A']),len(aa_to_featurevec_filtered['A'])))

# print(dict_ID_to_propertymatrix_padded_with_nans['P31749'][20])
# #End of: added by Sep