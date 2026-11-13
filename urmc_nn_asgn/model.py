# urmc_nn_asgn/model.py

import torch
import torch.nn as nn

import urmc_nn_asgn.config as config


class VisitCostNN(nn.Module):
    """Feedforward NN for tabular visit data supporting 0, 1, or 2 categoricals.

    Architecture:
      - 0+ Embedding layers (concatenated along dim=1)
      - Concatenation of [cat_embeddings, x_cont]
      - 2 hidden Linear layers with ReLU + Dropout
      - Single numeric regression output
    
    The order of the categoricals in embedding_specs must be the same as in 
    your data loader.
    """

    def __init__(
        self,
        num_cont_features: int = len(config.DATA_COLS_NUM),
        embedding_specs: dict[str, tuple[int, int]] | None = None,
        hidden_dims: tuple[int, int] = (config.NN_DIM_1, config.NN_DIM_2),
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        
        self.embedding_specs = embedding_specs or {}

        self.num_cont_features = num_cont_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Validate column names against config schema
        invalid = set(self.embedding_specs.keys()) - set(config.DATA_COLS_CAT)
        if invalid:
            raise ValueError(
                f"Invalid categorical columns {invalid}. Must be from {config.DATA_COLS_CAT}."
            )

        # Preserve exact column order expected by forward(x_cat, x_cont)
        self.cat_cols: list[str] = list(self.embedding_specs.keys())

        # Create one nn.Embedding per active categorical feature (empty if 0 cats)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_cats, embedding_dim=emb_dim)
            for num_cats, emb_dim in self.embedding_specs.values()
        ])

        total_emb_dim = sum(emb_dim for _, emb_dim in self.embedding_specs.values())
        in_features = num_cont_features + total_emb_dim

        # 2 hidden layers + 1 numeric output head
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )
        
    def get_cat_embeddings(self, x_cat: torch.Tensor) -> torch.Tensor | None:
        """Extracts and concatenates categorical embeddings for visualization."""
        
        if len(self.embeddings) == 0:
            return None

        emb_list = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
    
        return torch.cat(emb_list, dim=1)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        cat_emb = self.get_cat_embeddings(x_cat)

        if cat_emb is not None:
            x = torch.cat([cat_emb, x_cont], dim=1)
        else:
            x = x_cont

        out = self.mlp(x).squeeze(-1)

        if return_embeddings:
            return out, cat_emb
        return out