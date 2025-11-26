import os
import pandas as pd

def compute_metrics(df):

    df_metrics = df[["decoys","ablation","seed","score"]].groupby(["decoys","ablation","seed"]).mean().groupby(["decoys","ablation"]).agg(lambda x: f"${x.mean():.3f} \pm {x.std():.3f}$")
    
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
    model_name = "docknat"
    model_seed = 42
    res_dir = "outputs/pockets"
    SWAPPING_SEEDS = [0,1,2]

    list_dfs = []
    original_csv = os.path.join(res_dir, f"{model_name}_{model_seed}.csv")
    original_df = pd.read_csv(original_csv)
    for swapping_seed in SWAPPING_SEEDS:
        list_dfs.append(original_df.assign(ablation="none", seed=swapping_seed))

    for swapping_seed in SWAPPING_SEEDS:
        swapped_csv = os.path.join(res_dir, f"{model_name}_{model_seed}_swapping_seed{swapping_seed}.csv")
        swapped_df = pd.read_csv(swapped_csv)
        list_dfs.append(swapped_df.assign(ablation="target_swap", seed=swapping_seed))
    
    global_df = pd.concat(list_dfs, ignore_index=True)
    compute_metrics(global_df)