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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr,spearmanr
from torch.autograd import Variable

# from metrics import get_cindex, get_rm2
from data import RNA_dataset, Molecule_dataset
from ablation_utils import identity, target_swap, ligand_swap, CustomDualDataset
from model import mole_and_rna

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def val(model, test_loader, device):
    with torch.set_grad_enabled(False):
        model.eval()
        y_label = []
        y_pred = []
        for step, (batch_rna_test,batch_mole_test) in enumerate(test_loader):

            label = Variable(torch.from_numpy(np.array(batch_rna_test.y))).float()
            score = model(batch_rna_test.to(device), batch_mole_test.to(device))
            n = torch.squeeze(score, 1)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
    pcc_fold = pearsonr(y_label, y_pred)
    scc_fold = spearmanr(y_label, y_pred)
    rmse_fold = np.sqrt(mean_squared_error(y_label, y_pred))
    return pcc_fold, scc_fold, rmse_fold

# stratified CV
class regressor_stratified_cv:
    def __init__(self,n_splits=10,n_repeats=2,group_count=10,random_state=0,strategy='quantile'):
        self.group_count=group_count
        self.strategy=strategy
        self.cvkwargs=dict(n_splits=n_splits,n_repeats=n_repeats,random_state=random_state)  #Added shuffle here
        self.cv=RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer=KBinsDiscretizer(n_bins=self.group_count,encode='ordinal',strategy=self.strategy)  
            
    def split(self,X,y,groups=None):
        kgroups=self.discretizer.fit_transform(y[:,None])[:,0]
        return self.cv.split(X,kgroups,groups)
    
    def get_n_splits(self,X,y,groups=None):
        return self.cv.get_n_splits(X,y,groups)


def run_ablations():
    ablations = {
        "target-swap": target_swap,
        "ligand-swap": ligand_swap,
        "none": identity,
    }

    rows = []
    seed = 2
    RNA_type = 'All_sf'
    rna_dataset = RNA_dataset(RNA_type)
    molecule_dataset = Molecule_dataset(RNA_type)



    for ablation_name, ablation in ablations.items():
        
        ablation_rna_dataset, ablation_molecule_dataset =  ablation(rna_dataset, molecule_dataset)

        # 5 fold
        n_splits = 5
        kf = regressor_stratified_cv(n_splits=n_splits, n_repeats=1, random_state=seed, group_count=5, strategy='uniform')
        fold = 0

        for train_id,test_id in kf.split(rna_dataset, rna_dataset.y):

            fold += 1
            print(f"test_id={test_id}")
            test_dataset = CustomDualDataset([ablation_rna_dataset[i] for i in test_id], [ablation_molecule_dataset[i] for i in test_id])                    
            test_loader = DataLoader(
                test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = mole_and_rna(hidden_dim=128, device=device)
            model_path = f'save/model5fold_{RNA_type}{seed}_{fold}_{seed}.pth'

            if os.path.exists(model_path):

                pretrained_dict = torch.load(model_path,map_location=device)
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

            model = model.to(device)

            pcc_fold, scc_fold, rmse_fold = val(model, test_loader, device)

            pcc = pcc_fold[0]
            scc = scc_fold[0]
            rmse = rmse_fold
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
                    "fold": fold,
                }
            )
            print(rows)

    df = pd.DataFrame(rows)

    # Melt the dataframe to long format
    metrics = ['PCC', 'SCC', 'RMSE']
    df_melted = df.melt(
        id_vars=['ablation', 'fold'],
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
        caption="PCC results (mean ± std) across folds",
        label="tab:ablation_results",
        position="htbp",
    )

    with open('ablation_results.tex', 'w') as f:
        f.write(latex_output)

    print(latex_output)

if __name__ == "__main__":
    run_ablations()
