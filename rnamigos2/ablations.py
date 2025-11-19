import os

from typing import Any, List
import random

from rnamigos.learning.dataset import get_systems_from_cfg
from rnamigos.learning.dataloader import get_vs_loader
from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.utils.virtual_screen import get_auroc, enrichment_factor, run_virtual_screen

def guaranteed_derangement(items: List[Any], seed: int=0) -> List[Any]:
    """
    Generates a permutation of 'items' that is guaranteed to be a derangement.
    A derangement ensures that no element remains in its original position.
    
    This function uses a shuffle-and-fix approach to resolve fixed points.
    
    Args:
        items: The list of elements to be deranged (e.g., [1, 5, 9, 12]).

    Returns:
        A list representing the derangement of the input list.
    """
    random.seed(seed)
    n = len(items)
    if n <= 1:
        # Cannot derange a list of 0 or 1 elements
        return items[:]

    shuffled = items[:]
    
    # We loop until we find a derangement or manually resolve all fixed points
    attempts = 0
    while attempts < 100:
        random.shuffle(shuffled)
        
        # 1. Identify fixed points: where shuffled[i] == items[i]
        fixed_indices = [i for i in range(n) if shuffled[i] == items[i]]
        
        if not fixed_indices:
            # Success! Found a derangement.
            return shuffled
        
        # If fixed points exist, manually resolve them with simple swaps
        # This loop should ensure we resolve the fixed points in the current permutation
        if len(fixed_indices) == 1:
            i = fixed_indices[0]
            # If only one fixed point, swap it with its neighbor (j)
            j = (i + 1) % n
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            # Since n > 1, this swap guarantees the original fixed point 'i' is resolved.
            
        elif len(fixed_indices) > 1:
            # If multiple, swap pairs of fixed points (i, j)
            for k in range(0, len(fixed_indices) - 1, 2):
                i = fixed_indices[k]
                j = fixed_indices[k+1]
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            
            # If there's an odd one out, swap it with a random non-fixed point
            if len(fixed_indices) % 2 != 0:
                i = fixed_indices[-1]
                
                # Find a non-fixed point index j to swap with
                non_fixed_indices = [k for k in range(n) if k not in fixed_indices]
                if non_fixed_indices:
                    j = random.choice(non_fixed_indices)
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
                else:
                    # Fallback: swap with the next element (only happens if all elements were fixed, which is impossible for N>1)
                    j = (i + 1) % n
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        # Check again if the resolution worked (it should for all but the rarest N=2 case)
        if all(shuffled[i] != items[i] for i in range(n)):
            return shuffled

        attempts += 1 # Only happens if the resolution created new fixed points
        
    # As a last resort, return the best effort, though the while loop above should always succeed
    print("Warning: Derangement construction failed to fully resolve after many attempts.")
    return shuffled

def run_ablations():
    reps_only = True
    verbose = False
    rognan = False

    decoys = ["chembl", "pdb", "pdb_chembl", "decoy_finder"]
    model_name = "native_42"
    #model_name = "dock_42"
    model_path = "is_native/native_42"
    #model
    model_dir = "results/trained_models/"
    SWAP = 0
    res_dir = "outputs/robin" if SWAP == 0 else f"outputs/robin_swap_{SWAP}"
    ablations = {
        "target-swap": target_swap,
        "ligand-swap": ligand_swap,
        "none": identity,
    }
    seeds = [0, 1, 2]
    dfs_to_concat = []

    os.makedirs(res_dir, exist_ok=True)

        
    for decoy_mode in decoys:

        for ablation_name, ablation in ablations.items():

            for seed in seeds:

                full_model_path = os.path.join(model_dir, model_path)
                model, cfg = get_model_from_dirpath(full_model_path, return_cfg=True)
                test_systems = get_systems_from_cfg(cfg, return_test=True)
                model = model.to("cpu")
                dataloader = get_vs_loader(
                    systems=test_systems,
                    decoy_mode=decoy_mode,
                    cfg=cfg,
                    cache_graphs=False,
                    reps_only=reps_only,
                    verbose=verbose,
                    rognan=rognan,
                )
                lower_is_better = cfg.train.target in ["dock", "native_fp"]
                metric = enrichment_factor if decoy_mode == "robin" else get_auroc
                aurocs, scores, status, pocket_names, all_smiles = run_virtual_screen(
                    model, dataloader, metric=metric, lower_is_better=lower_is_better, verbose=verbose
                )

if __name__ == "__main__":
    run_ablations()