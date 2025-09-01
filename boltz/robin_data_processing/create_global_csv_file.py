import os
import sys

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

ligand_file_path = os.path.join(current_dir, "SMM_Target_Hits.csv")
ligands_df = pd.read_csv(ligand_file_path)
ligands_df = pd.melt(ligands_df, id_vars=['Name', 'Smile'], value_vars=list(ligands_df.columns)[2:], var_name="RNA_target")
ligands_df["RNA_target"] = ligands_df["RNA_target"].apply(lambda name:name.split("_")[0])

target_file_path = os.path.join(current_dir,"SMM_Sequence_Table_hit_rates.csv")
biomolecules_df = pd.read_csv(target_file_path)
rna_df = biomolecules_df[biomolecules_df["Biomolecule Type"]=="RNA"]

global_df = ligands_df.merge(rna_df, left_on="RNA_target", right_on="Nucleic Acid Target")
simplified_global_df = global_df[["RNA_target","Sequence (5' to 3')","Name","Smile","value"]].rename({"Name":"ligand","Smile":"ligand_smile","value":"binding","Sequence (5' to 3')":"RNA_target_sequence"}, axis=1)
simplified_global_df.to_csv(os.path.join(current_dir,"ROBIN_data.csv"))