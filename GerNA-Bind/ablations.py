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

# from metrics import get_cindex, get_rm2
from train_model import test
from data_utils.dataset import GerNA_dataset_from_pkl, custom_collate_fn
from net.model import GerNA
from utils.ablation_utils import target_swap, ligand_swap



def val(model, dataloader, device):

    test_performance, label, pred = test(net=model, dataLoader=dataloader, batch_size=8, mode="test", device=device, threshold = 0)

    acc = test_performance[7]
    sen = test_performance[5]
    spe = test_performance[6]
    pre = test_performance[4]
    f1score = test_performance[8]
    rocauc = test_performance[10]
    prauc = test_performance[11]
    mcc = test_performance[9]

    return acc, sen, spe, pre, f1score, rocauc, prauc, mcc, label, pred


def identity(dataset, seed=0):
    return dataset

def get_indices(dataset, split_method, split):

    rna_dataset_path = f"data/{dataset}/{dataset}_RNA.pkl"
    mol_dataset_path = f"data/{dataset}/{dataset}_Mol.pkl"
    interaction_dataset_path = f"data/{dataset}/{dataset}_interaction.csv"

    my_Dataset = GerNA_dataset_from_pkl(rna_dataset_path, mol_dataset_path, interaction_dataset_path)
    all_data_index = [ i for i in range(len(my_Dataset)) ]

    if split_method == 'random':
        kfold = KFold(n_splits=5, shuffle=True)
        _, indices = next(kfold.split(all_data_index))

    elif split_method == 'RNA':
        with open("data/"+dataset+"/"+dataset+"_"+split_method+".json", "r") as json_file:
            json_data = json.load(json_file)
            indices = json_data[split]

    elif split_method == 'mol':
        with open("data/"+dataset+"/"+dataset+"_"+split_method+".json", "r") as json_file:
            json_data = json.load(json_file)
            indices = json_data[split]

    elif split_method == 'both':
        with open("data/"+dataset+"/"+dataset+"_RNA.json", "r") as json_file:
            json_data_RNA = json.load(json_file)
            indices_rna = json_data[split]

        with open("data/"+dataset+"/"+dataset+"_mol.json", "r") as json_file:
            json_data_mol = json.load(json_file)
            indices_mol = json_data[split]

            indices = list( set(indices_rna).intersection(set(indices_mol)) )

    return indices
    

def eval_aurocs():
    # Add argument
    datasets = ["Biosensor", "Robin"]
    splits = ["random", "RNA", "mol", "both"]

    ablations = {
        "target-swap": target_swap,
        "ligand-swap": ligand_swap,
        "none": identity,
    }

    rows = []
    seeds = [0, 1, 2]

    for dataset in datasets:

        # model_path = f"save/{mode}/model/epoch-{epochs[mode]}.pt"
        model_path = f"Model/{dataset}_Model_baseline.pth"

        for split_method in splits:

            for ablation_name, ablation in ablations.items():

                rna_dataset_path = f"data/{dataset}/{dataset}_RNA.pkl"
                mol_dataset_path = f"data/{dataset}/{dataset}_Mol.pkl"
                interaction_dataset_path = f"data/{dataset}/{dataset}_interaction.csv"
                my_Dataset = GerNA_dataset_from_pkl(rna_dataset_path, mol_dataset_path, interaction_dataset_path)
                test_index = get_indices(dataset, split_method, "test")
                test_set = my_Dataset.smart_subset(test_index)

                for seed in seeds:

                    test_set = ablation(test_set, seed=seed)
                    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, collate_fn=custom_collate_fn, num_workers=10, pin_memory=True)
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    params = [4, 2, 128, 128]
                    model = GerNA(params, trigonometry = True, rna_graph = True, coors = True, coors_3_bead = True, uncertainty=True)

                    if os.path.exists(model_path):

                        pretrained_dict = torch.load(model_path,map_location="cuda:0" if torch.cuda.is_available() else "cpu")
                        model_dict = model.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)

                    model = model.to(device)

                    (
                        test_acc,
                        test_sen,
                        test_spe,
                        test_pre,
                        test_f1,
                        test_rocauc,
                        test_prauc,
                        test_mcc,
                        test_label,
                        test_pred,
                    ) = val(model, test_loader, device)
                    msg = (
                        "test_acc-%.4f, test_sen-%.4f, test_spe-%.4f, test_pre-%.4f, test_f1-%.4f, test_roauc-%.4f, test_prauc-%.4f, test_mcc-%.4f"
                        % (
                            test_acc,
                            test_sen,
                            test_spe,
                            test_pre,
                            test_f1,
                            test_rocauc,
                            test_prauc,
                            test_mcc,
                        )
                    )
                    print(msg)
                    rows.append(
                        {
                            "dataset": dataset,
                            "AuROC": test_rocauc,
                            "ablation": ablation_name,
                            "splitting": split_method,
                            "seed": seed,
                        }
                    )
                    print(rows)

        df = pd.DataFrame(rows)

    pivot_table = pd.pivot_table(df, index=["dataset", "splitting"], values=["AuROC"], columns=["ablation"])

    grouped = df.groupby(["dataset", "splitting", "ablation"])["AuROC"].agg(["mean", "std"]).reset_index()
    grouped["AuROC_formatted"] = grouped.apply(lambda x: f"${x['mean']:.3f} \pm {x['std']:.3f}$", axis=1)
    pivot_table = pd.pivot_table(
        grouped,
        values="AuROC_formatted",
        columns=["dataset", "splitting"],
        index=["ablation"],
        aggfunc="first",  # Take the first (and only) formatted string
    )

    latex_output = pivot_table.to_latex(
        escape=False,
        multirow=True,
        caption="AuROC results (mean ± std) across seeds",
        label="tab:auroc_results",
        position="htbp",
    )

    with open('auroc_results.tex', 'w') as f:
        f.write(latex_output)

    print(latex_output)

if __name__ == "__main__":
    eval_aurocs()
