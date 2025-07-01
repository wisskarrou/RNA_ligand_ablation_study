import pickle
import torch
from torch.utils import data
from torch_geometric.data import Batch
import numpy as np
import json

import numpy as np
import sys
import os

from torch_geometric.data import Data
from torch.autograd import Variable
import torch.optim as optim
import random
from utils.net_utils import *
from utils.metrics import *

torch.multiprocessing.set_sharing_strategy('file_system')


class GerNA_dataset(data.Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        self.RNA_repre, self.Mol_graph, self.RNA_Graph, self.RNA_feats, self.RNA_C4_coors, self.RNA_coors, self.Mol_feats, self.Mol_coors, self.LAS_input, self.Label_list = self.data
        
    def __len__(self):
        return len(self.RNA_repre)

    def __getitem__(self, index):
        return self.RNA_repre[index], self.Mol_graph[index], self.RNA_Graph[index], self.RNA_feats[index], self.RNA_C4_coors[index], self.RNA_coors[index], self.Mol_feats[index], self.Mol_coors[index], self.LAS_input[index], self.Label_list[index]

class GerNA_dataset2(data.Dataset):
    def __init__(self, rna_dataset_path, mol_dataset_path, interaction_dataset_path):
        with open(rna_dataset_path, 'rb') as f:
            self.rna_data = pickle.load(f)
        with open(mol_dataset_path, 'rb') as f:
            self.mol_data = pickle.load(f)
        with open(interaction_dataset_path, 'rb') as f:
            self.interaction_data = json.load(f)
        self.rna_id, self.RNA_repre, self.RNA_Graph, self.RNA_feats, self.RNA_C4_coors, self.RNA_coors = self.rna_data
        self.mol_id, self.Mol_graph, self.Mol_feats, self.Mol_coors, self.LAS_input = self.mol_data

    def __len__(self):
        return len(self.RNA_repre)

    def __getitem__(self, index):
        rna_index = self.interaction_data[index]["rna"]
        mol_index = self.interaction_data[index]["mol"]
        return self.RNA_repre[rna_index], self.Mol_graph[mol_index], self.RNA_Graph[rna_index], self.RNA_feats[rna_index], self.RNA_C4_coors[rna_index], self.RNA_coors[rna_index], self.Mol_feats[mol_index], self.Mol_coors[mol_index], self.LAS_input[mol_index], self.interaction_data[index]["label"]
        

def custom_collate_fn(batch):
    batch_RNA_repre = []
    batch_seq_mask = []
    batch_Mol_Graph = []
    batch_RNA_Graph = []
    batch_RNA_feats = []
    batch_RNA_C4_coors = []
    batch_RNA_coors = []
    batch_RNA_mask = []
    batch_Mol_feats = []
    batch_Mol_coors = []
    batch_Mol_mask = []
    batch_Mol_LAS = []
    batch_label = []
    
    for i,item in enumerate(batch):
        rna_repre,mol_graph,rna_graph,rna_feats,rna_coors_C4,rna_coors,mol_feats,mol_coors,mol_las,label = item
        
        batch_RNA_repre.append(rna_repre)
        batch_Mol_Graph.append(mol_graph)
        batch_RNA_Graph.append(rna_graph)
        
        batch_RNA_feats.append(rna_feats)
        batch_RNA_C4_coors.append(rna_coors_C4)
        batch_RNA_coors.append(rna_coors)

        batch_Mol_feats.append(mol_feats)
        batch_Mol_coors.append(mol_coors)

        batch_Mol_LAS.append(mol_las)
        batch_label.append(label)
    
    batch_seq_mask = torch.tensor( np.array(get_mask(batch_RNA_repre))).float()#.to(device)
    batch_RNA_repre = torch.tensor( np.array(pack2D(batch_RNA_repre))).float()#.to(device)
    
    batch_Mol_Graph = Batch.from_data_list(batch_Mol_Graph)#.to(device)
    batch_RNA_Graph = Batch.from_data_list(batch_RNA_Graph)#.to(device)
    
    batch_RNA_mask = torch.tensor( np.array(get_mask(batch_RNA_feats))).bool()#.to(device)
    batch_Mol_mask = torch.tensor( np.array(get_mask(batch_Mol_feats))).bool()#.to(device)
    
    batch_RNA_feats = torch.tensor( np.array(pack1D(batch_RNA_feats))).long()#.to(device)
    batch_RNA_C4_coors = torch.tensor( np.array(pack2D(batch_RNA_C4_coors)) ).float()
    batch_RNA_coors = torch.tensor( np.array(pack2D(batch_RNA_coors))).float()#.to(device)
    
    batch_Mol_feats = torch.tensor( np.array(pack1D(batch_Mol_feats))).long()#.to(device)
    batch_Mol_coors = torch.tensor( np.array(pack2D(batch_Mol_coors))).float()#.to(device)
    
    batch_Mol_LAS = torch.tensor( np.array(pack2D(batch_Mol_LAS))).float()#.to(device)
    batch_label = torch.FloatTensor( np.array(batch_label) )#.to(device)

    return batch_RNA_repre, batch_seq_mask, batch_Mol_Graph, batch_RNA_Graph, batch_RNA_feats, batch_RNA_C4_coors, batch_RNA_coors, batch_RNA_mask, batch_Mol_feats, batch_Mol_coors, batch_Mol_mask, batch_Mol_LAS, batch_label
