# urmc_nn_asgn/checkpoints.py

from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

from urmc_nn_asgn.model import VisitCostNN


def save_checkpoint(
    path: str | Path,
    model: VisitCostNN,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Saves model weights, optimizer state, and architecture config to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_cont_features": model.num_cont_features,
            "embedding_specs": model.embedding_specs,
            "hidden_dims": model.hidden_dims,
            "dropout": model.dropout,
        },
        "cat_cols": model.cat_cols,
        "epoch": epoch,
        "metrics": metrics or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    eval_mode: bool = True,
) -> tuple[VisitCostNN, dict[str, Any]]:
    """Rebuilds VisitCostNN from saved config, loads weights, and returns (model, metadata)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model = VisitCostNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if eval_mode:
        model.eval()

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    metadata = {
        "cat_cols": checkpoint.get("cat_cols", model.cat_cols),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
    }
    return model, metadata