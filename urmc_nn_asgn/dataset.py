# urmc_nn_asgn/dataset.py

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VisitDataset(Dataset):
    """Tabular dataset supporting 0, 1, or 2 categorical features.

    Assumes X columns are ordered as output by build_nn_preprocessing_pipeline:
    [icd10_cm_code (0), ccsr_code (1), numeric_features..., binary_features...]
    """

    def __init__(
        self,
        X_path: str | Path,
        y_path: str | Path,
        active_cat_indices: list[int] | None = None,
        total_cat_cols: int = 2,
    ):
        X = np.load(X_path)
        y = np.load(y_path)

        if active_cat_indices is None:
            active_cat_indices = [0, 1]

        # Continuous + binary features start after all categorical columns
        # Kept as float32 for linear layers
        self.x_cont = torch.tensor(X[:, total_cat_cols:], dtype=torch.float32)

        # Slice only the requested categorical columns as int64 (torch.long) for nn.Embedding
        if len(active_cat_indices) > 0:
            self.x_cat = torch.tensor(X[:, active_cat_indices], dtype=torch.long)
        else:
            # Empty tensor with shape (N, 0) so DataLoader batching works uniformly
            self.x_cat = torch.empty((len(X), 0), dtype=torch.long)

        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]