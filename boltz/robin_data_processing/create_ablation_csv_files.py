import os
import sys
import copy

import random
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def target_swap(original_df, seed=0):

    dset_new = original_df.copy(deep=True)
    random.seed(seed)
    rna_ids = list(original_df["RNA_target"].unique())
    swapped_rna_ids = random.sample(rna_ids, len(rna_ids))
    RNA_map = dict(zip(rna_ids, swapped_rna_ids))
    dset_new["RNA_target"] = dset_new["RNA_target"].map(RNA_map)
    sequence_map = dset_new.set_index("RNA_target")["RNA_target_sequence"].drop_duplicates()
    dset_new["RNA_target_sequence"] = dset_new["RNA_target"].map(sequence_map)

    return dset_new

def ligand_swap(original_df, seed=0):

    dset_new = original_df.copy(deep=True)
    random.seed(seed)
    mol_ids = list(original_df["ligand"].unique())
    swapped_mol_ids = random.sample(mol_ids, len(mol_ids))
    ligand_map = dict(zip(mol_ids, swapped_mol_ids))
    dset_new["ligand"] = dset_new["ligand"].map(ligand_map)
    smiles_map = dset_new.set_index("ligand")["ligand_smile"].drop_duplicates()
    dset_new["ligand_smile"] = dset_new["ligand"].map(smiles_map)

    return dset_new

original_csv_path = os.path.join(current_dir,"ROBIN_data.csv")
original_df = pd.read_csv(original_csv_path)

target_swapping_csv_path = os.path.join(current_dir,"ROBIN_data_target_swap.csv")
target_swap_df = target_swap(original_df)
target_swap_df.to_csv(target_swapping_csv_path, index=False)

ligand_swapping_csv_path = os.path.join(current_dir,"ROBIN_data_ligand_swap.csv")
ligand_swap_df = ligand_swap(original_df)
ligand_swap_df.to_csv(ligand_swapping_csv_path, index=False)