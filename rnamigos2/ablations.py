import os

from rnamigos.learning.dataset import get_systems_from_cfg
from rnamigos.learning.dataloader import get_vs_loader
from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.utils.virtual_screen import get_auroc, enrichment_factor, run_virtual_screen


reps_only = True
verbose = False
rognan = False

decoys = ["chembl", "pdb", "pdb_chembl", "decoy_finder"]
models = {
    "dock_42": "dock/dock_42",
    "native_42": "is_native/native_42",
}
model_name = "native_42"
#model_name = "dock_42"
model_path = "is_native/native_42"
#model
model_dir = "results/trained_models/"
SWAP = 0
res_dir = "outputs/robin" if SWAP == 0 else f"outputs/robin_swap_{SWAP}"

os.makedirs(res_dir, exist_ok=True)

for model_name, model_path in models.items():
    
    for decoy_mode in decoys:

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