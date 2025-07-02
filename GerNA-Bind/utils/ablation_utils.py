import random
import copy


def target_swap(dataset, seed=0):
    """dataset.target -> RNA sequence
    dataset.edge_attr, dataset.x, dataset.edge_index --> ligand
    """
    dset_new = copy.deepcopy(dataset)
    random.seed(seed)
    inds = random.sample(list(range(dataset.RNA_counts())), dataset.RNA_counts())
    dset_new.interaction_data["rna"] = dset_new.interaction_data["rna"].apply(lambda i:inds[i])
    return dset_new


def ligand_swap(dataset, seed=0):
    """dataset.target -> RNA sequence
    dataset.edge_attr, dataset.x, dataset.edge_index --> ligand
    """
    dset_new = copy.deepcopy(dataset)
    random.seed(seed)
    inds = random.sample(list(range(dataset.Mol_counts())), dataset.Mol_counts())
    dset_new.interaction_data["rna"] = dset_new.interaction_data["mol"].apply(lambda i:inds[i])
    return dset_new
