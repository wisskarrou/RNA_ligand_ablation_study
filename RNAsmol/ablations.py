import os
import warnings
import itertools

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Data
import argparse
import pandas as pd

# from metrics import get_cindex, get_rm2
from metrics import accuracy, sensitivity, specificity, precision, f1_score, roc_auc, pr_auc, mcc_score, recall
from dataset import *
from model import MCNN_GCN
from utils import *
from metrics import *


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    pred_cls_list = []
    label_list = []

    for data in dataloader:
        data.y = data.y.long()
        data = data.to(device)

        with torch.no_grad():
            ligand_x, protein_x, f, pred = model(data)
            loss = criterion(pred, data.y)
            pred_cls = torch.argmax(pred, dim=-1)

            pred_prob = F.softmax(pred, dim=-1)
            pred_prob, indices = torch.max(pred_prob, dim=-1)
            pred_prob[indices == 0] = 1.0 - pred_prob[indices == 0]

            pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
            pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(data.y.detach().cpu().numpy())
            running_loss.update(loss.item(), data.y.size(0))

    pred = np.concatenate(pred_list, axis=0)
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc = accuracy(label, pred_cls)
    sen = sensitivity(label, pred_cls)
    spe = specificity(label, pred_cls)
    pre = precision(label, pred_cls)
    rec = recall(label, pred_cls)
    f1score = f1_score(label, pred_cls)
    rocauc = roc_auc(label, pred)
    prauc = pr_auc(label, pred)
    mcc = mcc_score(label, pred_cls)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, sen, spe, pre, rec, f1score, rocauc, prauc, mcc, label, pred, f


def identity(dataset, seed=0):
    return dataset


def eval_aurocs():
    # Add argument
    modes = ["rnaperturbation", "molperturbation", "netperturbation"]
    # modes = ["rnaperturbation", "molperturbation"]
    epochs = {"rnaperturbation": 119, "molperturbation": 119, "netperturbation": 0}

    names = {"rnaperturbation": r"$\rho_r$", "molperturbation": r"$\rho_m$", "netperturbation": r"$\rho_n$"}

    ablations = {
        "target-swap": target_swap,
        "ligand-swap": ligand_swap,
        "target-ones": target_ones,
        "target-shuffle": target_shuffle,
        "anone": identity,
    }

    distributions = {}

    data_root = "data"
    rows = []
    seeds = [0, 1, 2]
    for mode in modes:
        # model_path = f"save/{mode}/model/epoch-{epochs[mode]}.pt"
        model_path = f"saved_model/pdb_bothaug_{mode}.pt"

        for test in [mode]:
            fpath = os.path.join(data_root, f"pdb_{test}")

            for ablation_name, ablation in ablations.items():
                test_set = GNNDataset(fpath, types="test")
                for seed in seeds:
                    test_set = ablation(test_set, seed=seed)

                    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)

                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    model = MCNN_GCN(3, 25 + 1, embedding_size=96, filter_num=32, out_dim=2, ban_heads=2).to(device)

                    criterion = nn.CrossEntropyLoss()
                    load_model_dict(model, model_path)

                    (
                        test_loss,
                        test_acc,
                        test_sen,
                        test_spe,
                        test_pre,
                        test_rec,
                        test_f1,
                        test_rocauc,
                        test_prauc,
                        test_mcc,
                        test_label,
                        test_pred,
                        test_att,
                    ) = val(model, criterion, test_loader, device)
                    msg = (
                        "test_loss-%.4f, test_acc-%.4f, test_sen-%.4f, test_spe-%.4f, test_pre-%.4f, test_rec-%.4f, test_f1-%.4f, test_roauc-%.4f, test_prauc-%.4f, test_mcc-%.4f"
                        % (
                            test_loss,
                            test_acc,
                            test_sen,
                            test_spe,
                            test_pre,
                            test_rec,
                            test_f1,
                            test_rocauc,
                            test_prauc,
                            test_mcc,
                        )
                    )

                    rows.append(
                        {
                            "train-set": names[mode],
                            "AuROC": test_rocauc,
                            "ablation": ablation_name,
                            "test-set": names[test],
                            "seed": seed,
                        }
                    )

        df = pd.DataFrame(rows)

    pivot_table = pd.pivot_table(df, index=["train-set", "test-set"], values=["AuROC"], columns=["ablation"])

    grouped = df.groupby(["train-set", "test-set", "ablation"])["AuROC"].agg(["mean", "std"]).reset_index()
    grouped["AuROC_formatted"] = grouped.apply(lambda x: f"${x['mean']:.3f} \pm {x['std']:.3f}$", axis=1)
    pivot_table = pd.pivot_table(
        grouped,
        values="AuROC_formatted",
        columns=["train-set", "test-set"],
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

    print(latex_output)


def eval_base(mode="rnaperturbation"):
    # Add argument
    epochs = {"rnaperturbation": 119, "molperturbation": 119, "netperturbation": 0}

    names = {"rnaperturbation": r"$\rho_r$", "molperturbation": r"$\rho_m$"}

    data_root = "data"
    seeds = [0, 1, 2]
    model_path = f"saved_model/pdb_bothaug_{mode}.pt"

    fpath = os.path.join(data_root, f"pdb_{mode}")
    test_set = GNNDataset(fpath, types="test")
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MCNN_GCN(3, 25 + 1, embedding_size=96, filter_num=32, out_dim=2, ban_heads=2).to(device)

    criterion = nn.CrossEntropyLoss()
    load_model_dict(model, model_path)

    (
        loss,
        acc,
        sen,
        spe,
        pre,
        rec,
        f1,
        rocauc,
        prauc,
        mcc,
        label,
        pred,
        att,
    ) = val(model, criterion, test_loader, device)

    return {"f1": f1, "mcc": mcc}


if __name__ == "__main__":
    eval_aurocs()
