import os
import sys

import pandas as pd
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

global_df_path = os.path.join(current_dir,"ROBIN_data.csv")
simplified_global_df = pd.read_csv(global_df_path)

#for i  in range(len(simplified_global_df)):
for i  in range(2):
    data = {
        "version": 1,
        "sequences": [
            {
                "rna": {
                    "id": "A",
                    "sequence": simplified_global_df["RNA_target_sequence"].iloc[i]
                },
            },
            {
                "ligand": {
                    "id": "B",
                    "smiles": simplified_global_df["ligand_smile"].iloc[i]
                }
            },
        ],
        "properties": [
            {
                "affinity": {
                    "binder": "B"
                }
            }
        ]
    }

    with open(os.path.join(current_dir,f'../robin_data_example/couple_{simplified_global_df["RNA_target"].iloc[i]}_{simplified_global_df["ligand"].iloc[i]}.yaml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)