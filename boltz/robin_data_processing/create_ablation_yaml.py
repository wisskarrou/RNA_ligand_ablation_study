import os
import sys

import yaml
import pandas as pd

from create_yaml import create_yaml_from_csv

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


# Create YAML files for original ROBIN data
#original_csv_path = os.path.join(current_dir,"ROBIN_data.csv")
#original_output_path = os.path.join(current_dir,f'../robin_data')
#create_yaml_from_csv(original_csv_path, original_output_path)

# Create YAML files for YAML data with target swapping
target_swapping_csv_path = os.path.join(current_dir,"ROBIN_data_target_swap.csv")
target_swapping_output_path = os.path.join(current_dir,f'../robin_data_target_swap')
create_yaml_from_csv(target_swapping_csv_path, target_swapping_output_path)

# Create YAML files for YAML data with ligand swapping
ligand_swapping_csv_path = os.path.join(current_dir,"ROBIN_data_ligand_swap.csv")
ligand_swapping_output_path = os.path.join(current_dir,f'../robin_data_ligand_swap')
create_yaml_from_csv(ligand_swapping_csv_path, ligand_swapping_output_path)
