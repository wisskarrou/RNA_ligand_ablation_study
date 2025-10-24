import os
import sys

import pandas as pd
import yaml


def create_yaml_from_csv(csv_path, output_path):

    simplified_global_df = pd.read_csv(csv_path)
    for i  in range(len(simplified_global_df)):
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

        with open(os.path.join(output_path, f'couple_{simplified_global_df["RNA_target"].iloc[i]}_{simplified_global_df["ligand"].iloc[i]}.yaml'), 'w') as file:
            yaml.dump(data, file, default_flow_style=False)