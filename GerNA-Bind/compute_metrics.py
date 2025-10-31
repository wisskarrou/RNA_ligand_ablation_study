import os
import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch


import pandas as pd
import math
from sklearn.metrics import roc_curve,auc,roc_auc_score,classification_report,precision_recall_curve,f1_score,average_precision_score
from scipy.stats import pearsonr,spearmanr


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def custom_multi_agg(x, threshold=0.5):
    y_pred = x.predicted_affinity
    y_true = x.true_affinity

    if len(x)<2:
        pcc = np.nan
        scc = np.nan
        rmse = np.nan
        
    else:
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        TP = TN = FP = FN = 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] >= threshold:
                TP += 1
            elif y_true[i] == 1 and y_pred[i] < threshold:
                FN += 1
            elif y_true[i] == 0 and y_pred[i] >= threshold:
                FP += 1
            elif y_true[i] == 0 and y_pred[i] < threshold:
                TN += 1
        Pre = 0 if (TP + FP) == 0 else (TP / (TP + FP))
        Sen = 0 if (TP + FN) == 0 else (TP / (TP + FN))
        Spe = 0 if (TN + FP) == 0 else (TN / (TN + FP))
        Acc = 0 if (TP + FP + TN + FN) == 0 else ((TP + TN) / (TP + FP + TN + FN))
        try:
            AUC = roc_auc_score(y_true, y_pred)
        except:
            AUC = np.nan

        F1_score = 0 if (Pre + Sen) == 0 else ( 2 * Pre * Sen / (Pre + Sen) )
        AUPRC = average_precision_score(y_true, y_pred)
        
        numerator = (TP * TN - FP * FN)
        denominator = (math.sqrt(TP + FP) * math.sqrt(TN + FN) * math.sqrt(TP + FN) * math.sqrt(TN + FP))
        if denominator == 0:
            mcc = 0
        else:
            mcc = numerator / denominator
    
    return pd.Series(
        {
            "Pre": Pre,
            "Sen": Sen,
            "Spe": Spe,
            "Acc": Acc,
            "F1_score": F1_score,
            "MCC": mcc,
            "AUC": AUC,
            "AUPRC": AUPRC
        }
    )

def compute_metrics(preds_path):
    df_inference = pd.read_csv(preds_path)

    df_global_metrics = df_inference.groupby(["dataset","split_method","ablation","seed"]).apply(custom_multi_agg).groupby(["dataset","split_method","ablation"]).agg(lambda x: f"${x.mean():.3f} \pm {x.std():.3f}$")
    df_target_wise_metrics = df_inference.groupby(["dataset","split_method","ablation","seed","RNA_id"]).apply(custom_multi_agg).add_suffix("_target").groupby(["dataset","split_method","ablation","seed"]).mean().groupby(["dataset","split_method","ablation"]).agg(lambda x: f"${x.mean():.3f} \pm {x.std():.3f}$")
    df_metrics = df_global_metrics.join(df_target_wise_metrics)

    latex_output = df_metrics.to_latex(
        escape=False,
        multirow=True,
        caption="Results (mean ± std) across folds",
        label="tab:ablation_results",
        position="htbp",
    )

    with open('ablation_results.tex', 'w') as f:
        f.write(latex_output)

    print(latex_output)

if __name__ == "__main__":
    preds_path = "inference_results.csv"
    compute_metrics(preds_path)