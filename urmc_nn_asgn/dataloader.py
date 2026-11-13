# urmc_nn_asgn/dataloader.py

# Dataloader to load 0, 1, or 2 categoricals for the model
# 0 categoricals (accepts either "none" or [])
#
# Use as:
# 0 cats: loader = get_split_loader(X_path, y_path, use_cats="none", shuffle=True)
# 1 cat:  loader = get_split_loader(X_path, y_path, use_cats=["ccsr_code"], shuffle=False)
# 2 cats: loader = get_split_loader(X_path, y_path, use_cats=["icd10_cm_code", "ccsr_code"], shuffle=False)
#
# Set shuffle=True for train loader, shuffle=False for eval, test, holdout

from pathlib import Path
from torch.utils.data import DataLoader

import urmc_nn_asgn.config as config
from urmc_nn_asgn.dataset import VisitDataset


def _resolve_cat_indices(use_cats: list[str] | str) -> list[int]:
    """Converts ['ccsr_code'], [], or 'none' into column indices."""
    if isinstance(use_cats, str):
        if use_cats.lower() == "none":
            return []
        raise ValueError(f"Unknown string for use_cats: '{use_cats}'. Pass a list of columns or 'none'.")

    # Validate column names and map to their saved index in the .npy file
    invalid = set(use_cats) - set(config.DATA_COLS_CAT)
    if invalid:
        raise ValueError(f"Invalid categorical columns {invalid}. Must be from {config.DATA_COLS_CAT}.")

    return [config.DATA_COLS_CAT.index(col) for col in use_cats]


def get_split_loader(
    X_path: str | Path,
    y_path: str | Path,
    use_cats: list[str] | str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    shuffle: bool = False,
    log_target: bool = True,
) -> DataLoader:
    """Loads a single split ('train', 'eval', 'test', or 'holdout')."""
    cat_indices = _resolve_cat_indices(use_cats)

    ds = VisitDataset(
        X_path=X_path,
        y_path=y_path,
        active_cat_indices=cat_indices,
        total_cat_cols=len(config.DATA_COLS_CAT),
        log_target=log_target,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )