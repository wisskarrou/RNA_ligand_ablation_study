import os
import sys

from loguru import logger
import numpy as np
import pandas as pd
import pathlib
from sklearn import metrics
from collections import defaultdict
import random

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.learning.dataset import get_systems_from_cfg
from rnamigos.learning.dataloader import get_vs_loader
from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.utils.mixing_utils import mix_two_scores, mix_two_dfs, get_mix_score, unmix, mix_all
from rnamigos.utils.virtual_screen import get_results_dfs, raw_df_to_mean_auroc
from scripts_fig.plot_utils import group_df


def pdb_eval(cfg, model, dump=True, verbose=True, decoys=None, rognan=False, reps_only=False, disable_fix_points=False):
    # Final VS validation on each decoy set
    if verbose:
        logger.info(f"Loading VS graphs from {cfg.data.pocket_graphs}")
        logger.info(f"Loading VS ligands from {cfg.data.ligand_db}")
    test_systems = get_systems_from_cfg(cfg, return_test=True)
    model = model.to("cpu")
    rows_aurocs, rows_raws = [], []
    if decoys is None:
        decoys = ["pdb", "pdb_chembl", "decoy_finder"]
    elif isinstance(decoys, str):
        decoys = [decoys]
    for decoy_mode in decoys:
        dataloader = get_vs_loader(
            systems=test_systems,
            decoy_mode=decoy_mode,
            cfg=cfg,
            cache_graphs=False,
            reps_only=reps_only,
            verbose=verbose,
            rognan=rognan,
            disable_fix_points=disable_fix_points,
        )
        print(f"len(dataloader={len(dataloader)}")
        print(f"dataset[0]={next(iter(dataloader))}")
        decoy_df_aurocs, decoys_dfs_raws = get_results_dfs(
            model=model, dataloader=dataloader, decoy_mode=decoy_mode, cfg=cfg, verbose=verbose
        )
        print(f"decoy_df_raws={decoys_dfs_raws}")
        print(f"decoy_df_aurocs={decoy_df_aurocs}")
        rows_aurocs.append(decoy_df_aurocs)
        rows_raws.append(decoys_dfs_raws)

    # Make it a df
    df_aurocs = pd.concat(rows_aurocs)
    df_raw = pd.concat(rows_raws)
    if dump:
        d = pathlib.Path(cfg.result_dir, parents=True, exist_ok=True)
        base_name = pathlib.Path(cfg.name).stem
        out_csv = d / (base_name + ".csv")
        out_csv_raw = d / (base_name + "_raw.csv")
        df_aurocs.to_csv(out_csv, index=False)
        df_raw.to_csv(out_csv_raw, index=False)

        # Just printing the results
        #df_chembl = df_aurocs.loc[df_aurocs["decoys"] == "chembl"]
        df_pdbchembl = df_aurocs.loc[df_aurocs["decoys"] == "pdb_chembl"]
        #df_chembl_grouped = group_df(df_chembl)
        df_pdbchembl_grouped = group_df(df_pdbchembl)
        logger.info(f"{cfg.name} Mean AuROC on pdbchembl: {np.mean(df_pdbchembl['score'].values)}")
        logger.info(f"{cfg.name} Mean grouped AuROC on pdbchembl: {np.mean(df_pdbchembl_grouped['score'].values)}")
    return df_aurocs, df_raw


def get_perf_model(models, res_dir, decoy_modes=("pdb", "chembl", "pdb_chembl"), reps_only=True, recompute=False):
    """
    This is quite similar to below, but additionally computes rognan.
     Also, only does it on just one decoy, and only on representatives.
    Could be merged
    """
    model_dir = "results/trained_models/"
    os.makedirs(res_dir, exist_ok=True)
    for model_name, model_path in models.items():
        decoys_df_aurocs, decoys_df_raws = list(), list()
        out_csv = os.path.join(res_dir, f"{model_name}.csv")
        out_csv_raw = os.path.join(res_dir, f"{model_name}_raw.csv")

        dict_decoys_df_aurocs_rognan, dict_decoys_df_raws_rognan = defaultdict(list), defaultdict(list)
        out_csv_rognan = {}
        out_csv_raw_rognan = {}

        for swapping_seed in SWAPPING_SEEDS:

            out_csv_rognan[swapping_seed] = os.path.join(res_dir, f"{model_name}_swapping_seed{swapping_seed}.csv")
            out_csv_raw_rognan[swapping_seed] = os.path.join(res_dir, f"{model_name}_swapping_seed{swapping_seed}_raw.csv")

        
        for decoy_mode in decoy_modes:

            # get model
            full_model_path = os.path.join(model_dir, model_path)
            model, cfg = get_model_from_dirpath(full_model_path, return_cfg=True)
            recompute_normal_results = recompute or not os.path.exists(out_csv)
            recompute_swapping_results = recompute or not os.path.exists(out_csv_rognan[SWAPPING_SEEDS[0]])
            if recompute_normal_results:
                recompute_normal_results = True
                print("RECOMPUTE NORMAL RESULTS")
                # get normal results
                df_aurocs, df_raw = pdb_eval(
                    cfg, model, verbose=False, dump=False, decoys=decoy_mode, reps_only=reps_only
                )
                print(f"AUROCS:{df_aurocs}")
                decoys_df_aurocs.append(df_aurocs)
                decoys_df_raws.append(df_raw)

            if recompute_swapping_results:
                recompute_swapping_results = True
                print("RECOMPUTE SWAPPING RESULTS")
                # get rognan results
                dict_df_aurocs_rognan = {}

                for swapping_seed in SWAPPING_SEEDS:
                    random.seed(swapping_seed)
                    df_aurocs_rognan, df_raw_rognan = pdb_eval(
                        cfg, model, verbose=False, dump=False, decoys=decoy_mode, rognan=True, reps_only=reps_only, disable_fix_points=True,
                    )
                    dict_decoys_df_aurocs_rognan[swapping_seed].append(df_aurocs_rognan)
                    dict_decoys_df_raws_rognan[swapping_seed].append(df_raw_rognan)
                    dict_df_aurocs_rognan[swapping_seed] = dict_decoys_df_aurocs_rognan[swapping_seed][-1]

        if recompute_normal_results:
            all_df_aurocs = pd.concat(decoys_df_aurocs)
            all_df_raws = pd.concat(decoys_df_raws)
            all_df_aurocs.to_csv(out_csv, index=False)
            all_df_raws.to_csv(out_csv_raw, index=False)

        else:
            df_aurocs = pd.read_csv(out_csv)

        if recompute_swapping_results:
            for swapping_seed in SWAPPING_SEEDS:
                all_df_aurocs_rognan = pd.concat(dict_decoys_df_aurocs_rognan[swapping_seed])
                all_df_raws_rognan = pd.concat(dict_decoys_df_raws_rognan[swapping_seed])
                all_df_aurocs_rognan.to_csv(out_csv_rognan[swapping_seed], index=False)
                all_df_raws_rognan.to_csv(out_csv_raw_rognan[swapping_seed], index=False)    
        else:
            dict_df_aurocs_rognan = {swapping_seed:pd.read_csv(out_csv_rognan[swapping_seed]) for swapping_seed in SWAPPING_SEEDS}

        # Just printing the results
        # We need this special case for rdock
        decoy = None
        
        if "decoys" in df_aurocs.columns:
            decoy = decoy_modes[-1]
            df_aurocs = df_aurocs.loc[df_aurocs["decoys"] == decoy]
            df_aurocs_rognan_to_print = {}
            for swapping_seed in SWAPPING_SEEDS:
                seed_df_auroc_rognan = dict_df_aurocs_rognan[swapping_seed]
                df_aurocs_rognan_to_print[swapping_seed] = seed_df_auroc_rognan.loc[seed_df_auroc_rognan["decoys"] == decoy]

        for swapping_seed in SWAPPING_SEEDS:
            test_auroc = np.mean(df_aurocs["score"].values)
            print(df_aurocs_rognan_to_print[swapping_seed])
            test_auroc_rognan = np.mean(df_aurocs_rognan_to_print[swapping_seed]["score"].values)
            gap_score = 2 * test_auroc - test_auroc_rognan
            print(f"{model_name}, {decoy}, swapping seed {swapping_seed}: AuROC {test_auroc:.3f} Rognan {test_auroc_rognan:.3f} GapScore {gap_score:.3f}")

def compute_mix_csvs(recompute=False):
    def merge_csvs(to_mix):
        """
        Aggregate native and dock results add mixing strategies
        """
        decoy_modes = ("pdb", "pdb_chembl")
        all_big_raws = []
        
        for decoy in decoy_modes:
            raw_dfs = [pd.read_csv(f"outputs/pockets/{r}_raw.csv") for r in to_mix]
            raw_dfs = [df.loc[df["decoys"] == decoy] for df in raw_dfs]
            raw_dfs = [df[["pocket_id", "smiles", "is_active", "raw_score"]] for df in raw_dfs]
            raw_dfs = [group_df(df) for df in raw_dfs]

            for df in raw_dfs:
                df["smiles"] = df["smiles"].str.strip()

            raw_dfs[0]["dock"] = raw_dfs[0]["raw_score"].values
            raw_dfs[1]["native"] = raw_dfs[1]["raw_score"].values
            raw_dfs = [df.drop("raw_score", axis=1) for df in raw_dfs]

            big_df_raw = raw_dfs[0]
            big_df_raw = big_df_raw.merge(raw_dfs[1], on=["pocket_id", "smiles", "is_active"], how="inner")
            print(f"big_raw_df={big_df_raw.head()}")
            big_df_raw = big_df_raw[["pocket_id", "smiles", "is_active", "dock", "native"]]

            def smaller_merge(df, score1, score2, outname):
                return mix_two_scores(df,
                                      score1=score1,
                                      score2=score2,
                                      outname_col=outname,
                                      use_max=True,
                                      add_decoy=False)[2]

            print(big_df_raw.head())
            raw_df_docknat = smaller_merge(big_df_raw, "dock", "native", "docknat")
            big_df_raw = big_df_raw.merge(raw_df_docknat, on=["pocket_id", "smiles", "is_active"], how="outer")

            dumb_decoy = [decoy for _ in range(len(big_df_raw))]
            big_df_raw.insert(len(big_df_raw.columns), "decoys", dumb_decoy)
            all_big_raws.append(big_df_raw)
        big_df_raw = pd.concat(all_big_raws)
        return big_df_raw

    for seed in SEEDS:
        for swapping_seed in [None]+SWAPPING_SEEDS:
            suffix = f"_swapping_seed{swapping_seed}" if swapping_seed is not None else ""
            out_path_raw = f"""outputs/pockets/big_df_{seed}{suffix}_raw.csv"""
            if not os.path.exists(out_path_raw) or recompute:
                # Combine the learnt methods and dump results
                TO_MIX = [f"""dock_{seed}{suffix}""",
                          f"""native_{seed}{suffix}"""]
                big_df_raw = merge_csvs(to_mix=TO_MIX)
                big_df_raw.to_csv(out_path_raw)

                # Dump aurocs dataframes for newly combined methods
                for method in ["docknat"]:
                    outpath = f"outputs/pockets/{method}_{seed}{suffix}.csv"
                    unmix(big_df_raw, score=method, outpath=outpath)



if __name__ == "__main__":
    GROUPED = True
    SEEDS = [42]
    SWAPPING_SEEDS = [0,1,2]

    MODELS = {
        "dock_42": "dock/dock_42",
        "native_42": "is_native/native_42",
    }
    RUNS = list(MODELS.keys())

    PAIRS = {
        ("native_42", "dock_42"): "docknat_42",
    }

    # GET INFERENCE CSVS
    decoys = ("pdb", "pdb_chembl")
    get_perf_model(models=MODELS, res_dir="outputs/pockets",
                   decoy_modes=decoys,
                   reps_only=GROUPED,
                   recompute=False)

    # PARSE INFERENCE CSVS AND MIX THEM
    compute_mix_csvs(recompute=True)
