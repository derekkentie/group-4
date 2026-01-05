from pathlib import Path
import pickle
import numpy as np
pathlist = Path(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\sources for dicts to use in gridsearch").glob('*')

#these two lines can be changed to experiment with different features and hyperparameters
ID_to_protein = pickle.load(open(r"C:\Users\20243625\OneDrive - TU Eindhoven\Desktop\group-4\docs\Sep's picklebestanden\dict ID to sequence",'rb')) 
add_length = True
nr_pieces_to_test = [1]
for path in pathlist:
    ID_to_mat_old = pickle.load(open(path,'rb')) #must be a dictionary coupling ID's to matrices
    for nr_pieces  in nr_pieces_to_test: #will probably run into semantic errors if it exceeds 146, as the code is not equiped for that, let me know if you want to try 
        # print('array I hope',ID_to_mat_old['P24941'].tolist()[0])
        nr_features = len((ID_to_mat_old['P24941'].tolist()[0])) #any featurevector for 1 aa would suffice
        protein_features_dict = dict()
        # ID_to_mat_new = dict()


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
                protein_features_dict[ID] = np.concatenate((vec_new,np.array([len(ID_to_protein[ID])])))
                # vec_extended = np.concatenate((vec_new,np.array([len(ID_to_protein[ID])]*nr_features))) #makes sure an entire extra row is added in the matrix containing just the lenghts
                # ID_to_mat_new[ID] = vec_extended.reshape(len(vec_extended)//nr_features,nr_features)
                
            else:
                protein_features_dict[ID] = vec_new
                # ID_to_mat_new[ID] = vec_new.reshape(len(vec_new)//nr_features,nr_features)

        pickle.dump(protein_features_dict, open(rf"{path} in {nr_pieces} pieces".replace('matrix','vector').replace('sources for ','protein '),'wb'))