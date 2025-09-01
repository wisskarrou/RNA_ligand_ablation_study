import json
import pandas as pd
import pickle
from itertools import chain
from collections import defaultdict

for dataset in ["Robin","Biosensor"]:

    dataset_path = f"data/{dataset}/{dataset}_all_data_3_coors_C4.pkl"

    with open(dataset_path,'rb') as f:
        data = pickle.load(f)

    RNA_repre,Mol_graph,RNA_Graph,RNA_feats,RNA_C4_coors,RNA_coors,Mol_feats,Mol_coors,LAS_input, y_true = data
    interaction_data = pd.DataFrame(index=range(len(data[0])), columns=["rna","mol","label"])

    already_visited_RNA = []
    rna_index = 0
    rna_data = [[] for _ in range(5)]

    for index, reference_RNA_rep in enumerate(data[0]):

        # iterate over all distinct RNAs
        if index not in already_visited_RNA:

            #store once each single RNA in rna_data list (in the order they arise in the pkl file)
            rna_data[0].append(data[0][index])
            rna_data[1].append(data[2][index])
            rna_data[2].append(data[3][index])
            rna_data[3].append(data[4][index])
            rna_data[4].append(data[5][index])
            
            for i in range(index, len(data[0])):

                if data[0][i].shape==reference_RNA_rep.shape:

                    if (data[0][i]==reference_RNA_rep).all():
                        interaction_data.loc[i]["rna"] = rna_index
                        interaction_data.loc[i]["label"] = data[9][i]
                        already_visited_RNA.append(i)

            rna_index += 1

    already_visited_SM = []
    mol_index = 0
    mol_data = [[] for _ in range(4)]

    for index, reference_mol_rep in enumerate(data[1]):

        # iterate oiver all distinct small molecules
        if index not in already_visited_SM:

            #store once each single small molecule in mol_data list (in the order they arise in the pkl file)
            mol_data[0].append(data[1][index])
            mol_data[1].append(data[6][index])
            mol_data[2].append(data[7][index])
            mol_data[3].append(data[8][index])

            for i in range(index, len(data[1])):
                if data[1][i]==reference_mol_rep:
                    interaction_data.loc[i]["mol"] = mol_index
                    already_visited_SM.append(i)
            mol_index += 1

    # Store RNA data in a pkl file
    with open(f'data/{dataset}/{dataset}_RNA.pkl', 'wb') as file:
        pickle.dump(rna_data, file)

    # Store small molecule data in a pkl file
    with open(f'data/{dataset}/{dataset}_Mol.pkl', 'wb') as file:
        pickle.dump(mol_data, file)

    # Store interaction data in a CSV file
    interaction_data.to_csv(f'data/{dataset}/{dataset}_interaction.csv')
    