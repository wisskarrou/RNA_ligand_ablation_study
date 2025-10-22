import os
import warnings
import itertools

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import numpy as np
import seaborn as sns
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Data
import argparse
import pandas as pd
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr,spearmanr
from torch.autograd import Variable

# from metrics import get_cindex, get_rm2
from data import RNA_dataset, Molecule_dataset
from model.deeprsma import DeepRSMA
from ablation_utils import identity, target_swap, ligand_swap, CustomDualDataset

def val(model, test_loader, device):
    with torch.set_grad_enabled(False):
        model.eval()
        y_label = []
        y_pred = []
        for step, (batch_v) in enumerate(test_loader):
            label = Variable(torch.from_numpy(np.array(batch_v[0].y))).float()
            score = model(batch_v[0].to(device), batch_v[1].to(device))

            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()

            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        p = pearsonr(y_label, y_pred).statistic
        s = spearmanr(y_label, y_pred).statistic
        rmse = np.sqrt(mean_squared_error(y_label, y_pred))
    return p, s, rmse


def run_ablations():
    cold_types = ["rna","mole","rm"]
    ablations = {
        "target-swap": target_swap,
        "ligand-swap": ligand_swap,
        "none": identity,
    }

    rows = []
    seeds = [2]
    RNA_type = 'All_sf'
    rna_dataset = RNA_dataset(RNA_type)
    molecule_dataset = Molecule_dataset(RNA_type)

    all_df = pd.read_csv('data/RSM_data/' + 'All_sf' + '_dataset_v1.csv', delimiter='\t') 

    for cold_type in cold_types:
        df1 = pd.read_csv('data/blind_test/cold_' + cold_type +'1.csv', delimiter=',')
        df2 = pd.read_csv('data/blind_test/cold_' + cold_type +'2.csv', delimiter=',')
        df3 = pd.read_csv('data/blind_test/cold_' + cold_type +'3.csv', delimiter=',')
        df4 = pd.read_csv('data/blind_test/cold_' + cold_type +'4.csv', delimiter=',')
        df5 = pd.read_csv('data/blind_test/cold_' + cold_type +'5.csv', delimiter=',')
        df = [df1, df2, df3, df4, df5]

        for ablation_name, ablation in ablations.items():
            
            for seed in seeds:

                ablation_rna_dataset, ablation_molecule_dataset =  ablation(rna_dataset, molecule_dataset)

                pcc = 0
                scc = 0
                rmse = 0

                for fold, df_f in enumerate(df):
                    test_id = df_f['Entry_ID'].tolist()
                    test_id = all_df[all_df['Entry_ID'].isin(test_id)].index.tolist()
                    test_dataset = CustomDualDataset(ablation_rna_dataset[test_id], ablation_molecule_dataset[test_id])                    
                    test_loader = DataLoader(
                        test_dataset, batch_size=16, num_workers=0, drop_last=False, shuffle=False
                    )
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    model = DeepRSMA(hidden_dim=128)
                    model_path = f'save/model5fold_{RNA_type}{seed}_{fold}_{seed}.pth'
                    if os.path.exists(model_path):

                        pretrained_dict = torch.load(model_path,map_location="cuda:0" if torch.cuda.is_available() else "cpu")
                        model_dict = model.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)

                    model = model.to(device)

                    (
                        pcc_fold,
                        scc_fold,
                        rmse_fold
                    ) = val(model, test_loader, device)
                    pcc += pcc_fold/len(df)
                    scc += scc_fold/len(df)
                    rmse += rmse_fold/len(df)
                msg = (
                    "test_pcc-%.4f, test_scc-%.4f, test_rmse-%.4f"
                    % (
                        pcc,
                        scc,
                        rmse,
                    )
                )
                print(msg)
                rows.append(
                    {
                        "PCC": pcc,
                        "SCC": scc,
                        "RMSE": rmse,
                        "ablation": ablation_name,
                        "seed": seed,
                    }
                )
                print(rows)

    df = pd.DataFrame(rows)

    # Melt the dataframe to long format
    metrics = ['PCC', 'SCC', 'RMSE']
    df_melted = df.melt(
        id_vars=['ablation', 'seed'],
        value_vars=metrics,
        var_name='metric',
        value_name='value'
    )

    # Group by ablation and metric, calculate mean and std
    grouped_melted = df_melted.groupby(['ablation', 'metric'])['value'].agg(['mean', 'std']).reset_index()
    grouped_melted['formatted'] = grouped_melted.apply(lambda x: f"${x['mean']:.3f} \pm {x['std']:.3f}$", axis=1)

    # Create the final pivot table: ablations as rows, metrics as columns
    pivot_table = pd.pivot_table(
        grouped_melted,
        values='formatted',
        columns='metric',
        index='ablation',
        aggfunc='first'
    )

    latex_output = pivot_table.to_latex(
        escape=False,
        multirow=True,
        caption="PCC results (mean ± std) across seeds",
        label="tab:ablation_results",
        position="htbp",
    )

    with open('ablation_results.tex', 'w') as f:
        f.write(latex_output)

    print(latex_output)

if __name__ == "__main__":
    run_ablations()
