import os
import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch


import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr,spearmanr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def custom_multi_agg(x):

    if len(x)<2 or x.true_affinity.std()==0 or x.predicted_affinity.std()==0:

        pcc = np.nan
        scc = np.nan
        rmse = np.nan
        
    else:

        pcc = pearsonr(x.true_affinity,x.predicted_affinity)[0]
        scc = spearmanr(x.true_affinity,x.predicted_affinity)[0]
        rmse = np.sqrt(mean_squared_error(x.true_affinity,x.predicted_affinity))

    return pd.Series(
        {
            "PCC": pcc,
            "SCC": scc,
            "RMSE": rmse,
        }
    )

def compute_metrics(preds_path):
    df_inference = pd.read_csv(preds_path)

    df_global_metrics = df_inference.groupby(["ablation","seed","fold"]).apply(custom_multi_agg).groupby(["ablation","seed"]).mean().groupby("ablation").agg(lambda x: f"${x.mean():.3f} \pm {x.std():.3f}$")
    df_target_wise_metrics = df_inference.groupby(["ablation","seed","fold","RNA_id"]).apply(custom_multi_agg).add_suffix("_target").groupby(["ablation","seed"]).mean().groupby("ablation").agg(lambda x: f"${x.mean():.3f} \pm {x.std():.3f}$")
    df_metrics = df_global_metrics.join(df_target_wise_metrics)

    latex_output = df_metrics.to_latex(
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
    preds_path = "inference_results.csv"
    compute_metrics(preds_path)