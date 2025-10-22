import random
import copy
from torch.utils.data import Dataset
from torch_geometric.data import Data

class CustomDualDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        assert len(self.dataset1) == len(self.dataset2)

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        return len(self.dataset1)  

def identity(rna_dataset, mol_dataset, seed=0):
    return [rna for rna in rna_dataset], [mol for mol in mol_dataset]

def target_swap(rna_dataset, mol_dataset, seed=0):
    rna_initial_mapping = [None for rna in rna_dataset]
    rna_index = 0
    reference_rnas = []
    for i, rna in enumerate(rna_dataset):
        if rna_initial_mapping[i] is None:
            reference_x = rna.x
            reference_rnas.append(i)
            for j in range(i, len(rna_dataset)):
                if rna_dataset[j].x.shape == reference_x.shape:
                    if (rna_dataset[j].x == reference_x).all():
                        rna_initial_mapping[j] = rna_index
            rna_index += 1
        
    random.seed(seed)
    inds = random.sample(reference_rnas, rna_index)
    RNA_swap = [inds[rna_ind] for rna_ind in rna_initial_mapping]
    dset_new = []
    for i, d in enumerate(rna_dataset):
        new_RNA = rna_dataset[RNA_swap[i]]
        dset_new.append(
            Data(x=new_RNA.x, edge_index=new_RNA.edge_index, y=d.y, t_id=new_RNA.t_id, e_id=new_RNA.e_id, emb=new_RNA.emb, rna_len = new_RNA.rna_len)
        )
    return dset_new, [mol for mol in mol_dataset]



def ligand_swap(rna_dataset, mol_dataset, seed=0):
    mol_initial_mapping = [None for mol in mol_dataset]
    mol_index = 0
    reference_mols = []
    for i, mol in enumerate(mol_dataset):
        if mol_initial_mapping[i] is None:
            reference_smiles = mol.smiles_ori
            reference_mols.append(i)
            for j in range(i, len(mol_dataset)):
                if mol_dataset[j].smiles_ori == reference_smiles:
                    mol_initial_mapping[j] = mol_index
            mol_index += 1
        
    random.seed(seed)
    inds = random.sample(reference_mols, mol_index)
    mol_swap = [inds[mol_ind] for mol_ind in mol_initial_mapping]
    dset_new = []
    for i, d in enumerate(mol_dataset):
        dset_new.append(mol_dataset[mol_swap[i]])
    return [rna for rna in rna_dataset], dset_new