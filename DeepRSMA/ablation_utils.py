from typing import List, Any
import random
import copy
from torch.utils.data import Dataset
from torch_geometric.data import Data

def guaranteed_derangement(items: List[Any], seed: int=0) -> List[Any]:
    """
    Generates a permutation of 'items' that is guaranteed to be a derangement.
    A derangement ensures that no element remains in its original position.
    
    This function uses a shuffle-and-fix approach to resolve fixed points.
    
    Args:
        items: The list of elements to be deranged (e.g., [1, 5, 9, 12]).

    Returns:
        A list representing the derangement of the input list.
    """
    random.seed(seed)
    n = len(items)
    if n <= 1:
        # Cannot derange a list of 0 or 1 elements
        return items[:]

    shuffled = items[:]
    
    # We loop until we find a derangement or manually resolve all fixed points
    attempts = 0
    while attempts < 100:
        random.shuffle(shuffled)
        
        # 1. Identify fixed points: where shuffled[i] == items[i]
        fixed_indices = [i for i in range(n) if shuffled[i] == items[i]]
        
        if not fixed_indices:
            # Success! Found a derangement.
            return shuffled
        
        # If fixed points exist, manually resolve them with simple swaps
        # This loop should ensure we resolve the fixed points in the current permutation
        if len(fixed_indices) == 1:
            i = fixed_indices[0]
            # If only one fixed point, swap it with its neighbor (j)
            j = (i + 1) % n
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            # Since n > 1, this swap guarantees the original fixed point 'i' is resolved.
            
        elif len(fixed_indices) > 1:
            # If multiple, swap pairs of fixed points (i, j)
            for k in range(0, len(fixed_indices) - 1, 2):
                i = fixed_indices[k]
                j = fixed_indices[k+1]
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            
            # If there's an odd one out, swap it with a random non-fixed point
            if len(fixed_indices) % 2 != 0:
                i = fixed_indices[-1]
                
                # Find a non-fixed point index j to swap with
                non_fixed_indices = [k for k in range(n) if k not in fixed_indices]
                if non_fixed_indices:
                    j = random.choice(non_fixed_indices)
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
                else:
                    # Fallback: swap with the next element (only happens if all elements were fixed, which is impossible for N>1)
                    j = (i + 1) % n
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        # Check again if the resolution worked (it should for all but the rarest N=2 case)
        if all(shuffled[i] != items[i] for i in range(n)):
             return shuffled

        attempts += 1 # Only happens if the resolution created new fixed points
        
    # As a last resort, return the best effort, though the while loop above should always succeed
    print("Warning: Derangement construction failed to fully resolve after many attempts.")
    return shuffled

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
        
    inds = guaranteed_derangement(reference_rnas, seed) 
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
        
    inds = guaranteed_derangement(reference_mols, seed)
    mol_swap = [inds[mol_ind] for mol_ind in mol_initial_mapping]
    dset_new = []
    for i, d in enumerate(mol_dataset):
        dset_new.append(mol_dataset[mol_swap[i]])
    return [rna for rna in rna_dataset], dset_new