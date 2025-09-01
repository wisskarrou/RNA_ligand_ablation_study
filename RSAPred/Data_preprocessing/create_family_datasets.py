import pandas as pd

families = ["Aptamers", "miRNA", "Repeats", "Ribosomal", "Riboswitch", "Viral_RNA"]
for family in families:
    rna_mol_df = pd.read_csv("data/"+family+"_dataset_v1.csv", sep="\t")

    # extracts RNAs from the family dataset
    rna_df = rna_mol_df[["Target_RNA_sequence", "Target_RNA_name",	"Target_RNA_ID"]]

    # remove duplicates use subset="Target_RNA_ID" to avoid counting twice a same RNA (ie having one RNA ID) even if it has 2 different names 
    rna_df.drop_duplicates(subset="Target_RNA_ID").to_csv("data/rna_data_"+family+".csv", index=False, sep='\t')

    # extracts small molecules from the family dataset
    mol_df = rna_mol_df[["SMILES", "Molecule_ID"]]
    # remove duplicates
    mol_df.drop_duplicates().to_csv("data/mol_data_"+family+".smi", index=False, header=False, sep='\t')

    