import sys
import pandas as pd
import random

def create_rna_rotation(rna_list):
    # Because of the limited number of RNAs + not unique (usually between 2 to 4 RNAs per family
    # Create a rotation of RNA IDs where each RNA maps to the next one in the list
    rotation = {}
    for i in range(len(rna_list)):
        rotation[rna_list[i]] = rna_list[(i + 1) % len(rna_list)]
    return rotation

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python swap_v2.py <input_dataset> <output_file>")
        sys.exit(1)

    # dataset_raw = positive or negative file
    dataset_raw = sys.argv[1]
    outfile = sys.argv[2]

    print(f"Loading dataset from {dataset_raw}...")
    try:
        data_df = pd.read_csv(dataset_raw, sep="\t")
        if data_df.empty:
            print("Error: The input dataset is empty.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        sys.exit(1)

    # Debugging information (to check on the RNAs and the swapping that is happening)
    print("\nDataset Information:")
    print(f"Number of rows: {len(data_df)}")
    print(f"Columns: {list(data_df.columns)}")
    print("\nFirst few rows of Target_RNA_ID:")
    print(data_df['Target_RNA_ID'].head())
    print("\nUnique RNA IDs:")
    unique_rnas = list(data_df['Target_RNA_ID'].unique())
    print(unique_rnas)
    print(f"\nNumber of unique RNAs: {len(unique_rnas)}")

    # Drop any columns with all NaN values, to resolve an error of duplicate columns
    data_df.dropna(axis=1, how='all', inplace=True)
    print("\nOriginal data:")
    print(data_df.head())
    
    # Define RNA and molecule columns
    rna_cols = ['Target_RNA_ID', 'Target_RNA_name', 'A', 'G', 'C', 'U', 'AA_x', 'AG', 'AC', 'AU', 'GA', 'GG', 'GC', 'GU', 
                'CA', 'CG', 'CC', 'CU', 'UA', 'UG', 'UC', 'UU', 'AAA', 'AAG', 'AAC', 'AAU', 'AGA', 'AGG', 'AGC', 'AGU', 
                'ACA', 'ACG', 'ACC', 'ACU', 'AUA', 'AUG', 'AUC', 'AUU', 'GAA', 'GAG', 'GAC', 'GAU', 'GGA', 'GGG', 'GGGC', 
                'GGGU', 'GCA', 'GCG', 'GCC', 'GCU', 'GUA', 'GUG', 'GUC', 'GUU', 'CAA', 'CAG', 'CAC', 'CAU', 'CGA', 'CGG', 
                'CGC', 'CGU', 'CCA', 'CCG', 'CCC', 'CCU', 'CUA', 'CUG', 'CUC', 'CUU', 'UAA', 'UAG', 'UAC', 'UAU', 'UGA', 
                'UGG', 'UGC', 'UGU', 'UCA', 'UCG', 'UCC', 'UCU', 'UUA', 'UUG', 'UUC', 'UUU']
    
    # Add all other RNA-related columns
    rna_cols.extend([col for col in data_df.columns if any(prefix in col for prefix in  
                    ['AAAA', 'AAAG', 'AAAC', 'AAAU', 'AAGA', 'AAGG', 'AAGC', 'AAGU', 'AACA', 'AACG',
                     'AACC', 'AACU', 'AAUA', 'AAUG', 'AAUC', 'AAUU', 'AGAA', 'AGAG', 'AGAC', 'AGAU', 
                     'AGGA', 'AGGG', 'AGGC', 'AGGU', 'AGCA', 'AGCG', 'AGCC', 'AGCU', 'AGUA', 'AGUG', 
                     'AGUC', 'AGUU', 'ACAA', 'ACAG', 'ACAC', 'ACAU', 'ACGA', 'ACGG', 'ACGC', 'ACGU', 
                     'ACCA', 'ACCG', 'ACCC', 'ACCU', 'ACUA', 'ACUG', 'ACUC', 'ACUU', 'AUAA', 'AUAG', 
                     'AUAC', 'AUAU', 'AUGA', 'AUGG', 'AUGC', 'AUGU', 'AUCA', 'AUCG', 'AUCC', 'AUCU', 
                     'AUUA', 'AUUG', 'AUUC', 'AUUU', 'GAAA', 'GAAG', 'GAAC', 'GAAU', 'GAGA', 'GAGG', 
                     'GAGC', 'GAGU', 'GACA', 'GACG', 'GACC', 'GACU', 'GAUA', 'GAUG', 'GAUC', 'GAUU', 
                     'GGAA', 'GGAG', 'GGAC', 'GGAU', 'GGGA', 'GGGG', 'GGGC', 'GGGU', 'GGCA', 'GGCG', 
                     'GGCC', 'GGCU', 'GGUA', 'GGUG', 'GGUC', 'GGUU', 'GCAA', 'GCAG', 'GCAC', 'GCAU', 
                     'GCGA', 'GCGG', 'GCGC', 'GCGU', 'GCCA', 'GCCG', 'GCCC', 'GCCU', 'GCUA', 'GCUG', 
                     'GCUC', 'GCUU', 'GUAA', 'GUAG', 'GUAC', 'GUAU', 'GUGA', 'GUGG', 'GUGC', 'GUGU', 
                     'GUCA', 'GUCG', 'GUCC', 'GUCU', 'GUUA', 'GUUG', 'GUUC', 'GUUU', 'CAAA', 'CAAG', 
                     'CAAC', 'CAAU', 'CAGA', 'CAGG', 'CAGC', 'CAGU', 'CACA', 'CACG', 'CACC', 'CACU', 
                     'CAUA', 'CAUG', 'CAUC', 'CAUU', 'CGAA', 'CGAG', 'CGAC', 'CGAU', 'CGGA', 'CGGG', 
                     'CGGC', 'CGGU', 'CGCA', 'CGCG', 'CGCC', 'CGCU', 'CGUA', 'CGUG', 'CGUC', 'CGUU', 
                     'CCAA', 'CCAG', 'CCAC', 'CCAU', 'CCGA', 'CCGG', 'CCGC', 'CCGU', 'CCCA', 'CCCG', 
                     'CCCC', 'CCCU', 'CCUA', 'CCUG', 'CCUC', 'CCUU', 'CUAA', 'CUAG', 'CUAC', 'CUAU', 
                     'CUGA', 'CUGG', 'CUGC', 'CUGU', 'CUCA', 'CUCG', 'CUCC', 'CUCU', 'CUUA', 'CUUG', 
                     'CUUC', 'CUUU', 'UAAA', 'UAAG', 'UAAC', 'UAAU', 'UAGA', 'UAGG', 'UAGC', 'UAGU', 
                     'UACA', 'UACG', 'UACC', 'UACU', 'UAUA', 'UAUG', 'UAUC', 'UAUU', 'UGAA', 'UGAG', 
                     'UGAC', 'UGAU', 'UGGA', 'UGGG', 'UGGC', 'UGGU', 'UGCA', 'UGCG', 'UGCC', 'UGCU', 
                     'UGUA', 'UGUG', 'UGUC', 'UGUU', 'UCAA', 'UCAG', 'UCAC', 'UCAU', 'UCGA', 'UCGG', 
                     'UCGC', 'UCGU', 'UCCA', 'UCCG', 'UCCC', 'UCCU', 'UCUA', 'UCUG', 'UCUC', 'UCUU', 
                     'UUAA', 'UUAG', 'UUAC', 'UUAU', 'UUGA', 'UUGG', 'UUGC', 'UUGU', 'UUCA', 'UUCG', 
                     'UUCC', 'UUCU', 'UUUA', 'UUUG', 'UUUC', 'UUUU', 'AA_y', 'DNC_AG', 'DNC_AC', 
                     'DNC_AU', 'DNC_GA', 'DNC_GG', 'DNC_GC', 'DNC_GU', 'DNC_CA', 'DNC_CG', 'DNC_CC', 
                     'DNC_CU', 'DNC_UA', 'DNC_UG', 'DNC_UC', 'DNC_UU', 'DNC_Feat_17', 'DNC_Feat_18', 
                     'DNC_Feat_19', 'DNC_Feat_20', 'DNC_Feat_21', 'DNC_Feat_22', 'DNC_Feat_23', 
                     'DNC_Feat_24', 'A(((', 'A((.', 'A(..', 'A(.(', 'A.((', 'A.(.', 'A..(', 'A...', 
                     'G(((', 'G((.', 'G(..', 'G(.(', 'G.((', 'G.(.', 'G..(', 'G...', 'C(((', 'C((.', 
                     'C(..', 'C(.(', 'C.((', 'C.(.', 'C..(', 'C...', 'U(((', 'U((.', 'U(..', 'U(.(', 
                     'U.((', 'U.(.', 'U..(', 'U...', 'A,A', 'A,G', 'A,C', 'A,U', 'A,A-U', 'A,U-A', 
                     'A,G-C', 'A,C-G', 'A,G-U', 'A,U-G', 'G,A', 'G,G', 'G,C', 'G,U', 'G,A-U', 'G,U-A', 
                     'G,G-C', 'G,C-G', 'G,G-U', 'G,U-G', 'C,A', 'C,G', 'C,C', 'C,U', 'C,A-U', 'C,U-A', 
                     'C,G-C', 'C,C-G', 'C,G-U', 'C,U-G', 'U,A', 'U,G', 'U,C', 'U,U', 'U,A-U', 'U,U-A', 
                     'U,G-C', 'U,C-G', 'U,G-U', 'U,U-G', 'A-U,A', 'A-U,G', 'A-U,C', 'A-U,U', 'A-U,A-U', 
                     'A-U,U-A', 'A-U,G-C', 'A-U,C-G', 'A-U,G-U', 'A-U,U-G', 'U-A,A', 'U-A,G', 'U-A,C', 
                     'U-A,U', 'U-A,A-U', 'U-A,U-A', 'U-A,G-C', 'U-A,C-G', 'U-A,G-U', 'U-A,U-G', 'G-C,A', 
                     'G-C,G', 'G-C,C', 'G-C,U', 'G-C,A-U', 'G-C,U-A', 'G-C,G-C', 'G-C,C-G', 'G-C,G-U', 
                     'G-C,U-G', 'C-G,A', 'C-G,G', 'C-G,C', 'C-G,U', 'C-G,A-U', 'C-G,U-A', 'C-G,G-C', 
                     'C-G,C-G', 'C-G,G-U', 'C-G,U-G', 'G-U,A', 'G-U,G', 'G-U,C', 'G-U,U', 'G-U,A-U', 
                     'G-U,U-A', 'G-U,G-C', 'G-U,C-G', 'G-U,G-U', 'G-U,U-G', 'U-G,A', 'U-G,G', 'U-G,C', 
                     'U-G,U', 'U-G,A-U', 'U-G,U-A', 'U-G,G-C', 'U-G,C-G', 'U-G,G-U', 'U-G,U-G', 
                     'PS_Feat_101', 'PS_Feat_102', 'PS_Feat_103', 'PS_Feat_104', 'PS_Feat_105', 
                     'PS_Feat_106', 'PS_Feat_107', 'PS_Feat_108'])])

    rna_cols = list(dict.fromkeys(rna_cols))
    
    # Get molecule columns (everything else except pKd)
    mol_cols = ['name']
    mol_cols.extend([col for col in data_df.columns if col not in rna_cols and col != 'pKd'])

    # Create RNA rotation (for example : ROBIN_RNA_1 -> ROBIN_RNA_14 -> ROBIN_RNA_24 -> ROBIN_RNA_1)
    rna_swap_dict = create_rna_rotation(unique_rnas)
    print("\nRNA rotation mapping:")
    for original, swapped in rna_swap_dict.items():
        print(f"{original} -> {swapped}")

    # Create a list to store the new rows
    new_rows = []
    
    # Build new dataset with swapped RNAs
    print("\nCreating swapped dataset...")
    skipped_rows = 0
    
    unique_molecules = data_df['name'].unique()
    
    # Create a list to store swapping summary
    swapping_summary = []
    
    # For each molecule, create a new row with swapped RNA
    for mol in unique_molecules:
        mol_rows = data_df[data_df['name'] == mol]
        
        for _, row in mol_rows.iterrows():
            try:
                original_rna = row['Target_RNA_ID']
                swapped_rna = rna_swap_dict[original_rna]
                swapped_rna_row = data_df[data_df['Target_RNA_ID'] == swapped_rna].iloc[0]
                new_row = {}
                for col in rna_cols:
                    new_row[col] = swapped_rna_row[col]
                for col in mol_cols:
                    new_row[col] = row[col]
                # Copy pKd from original row
                new_row['pKd'] = row['pKd']
                
                new_rows.append(new_row)
                
                # Add to swapping summary (so we can print at the end the original and swapped rnas to check)
                # this part is a check up but not necessary
                swapping_summary.append({
                    'Original_RNA': original_rna,
                    'Swapped_RNA': swapped_rna,
                    'Molecule': mol,
                    'pKd': row['pKd']
                })
                
            except Exception as e:
                print(f"Error processing molecule {mol} with RNA {original_rna}: {str(e)}")
                skipped_rows += 1
                continue

    # Create final dataframe from the list of rows
    if new_rows:
        final_df = pd.DataFrame(new_rows)
        final_df = final_df[data_df.columns]
        print(f"\nSaving swapped dataset to {outfile}...")
        final_df.to_csv(outfile, sep="\t", index=False)
        print(f"Original rows: {len(data_df)}, Final rows: {len(final_df)}, Skipped rows: {skipped_rows}")
        print("\nSwapped data:")
        print(final_df.head())
        
        # Create and display swapping summary
        summary_df = pd.DataFrame(swapping_summary)
        print("\nSwapping Summary (first 5 rows):")
        print(summary_df.head().to_string(index=False))
    else:
        print("Error: No rows were successfully processed.")
        sys.exit(1)
    
    print("Done!")