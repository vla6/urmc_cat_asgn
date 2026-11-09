# urmc_nn_asgn/pipeline.py
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import urmc_nn_asgn.config as config
from urmc_nn_asgn.ordinal_nn_encoder import OrdinalNNEncoder

def build_nn_preprocessing_pipeline(
    categorical_features: list[str] = config.DATA_COLS_CAT,
    numeric_features: list[str] = config.DATA_COLS_CONT,
    binary_features: list[str] = config.DATA_COLS_BINARY,
    encode_missing_unknown: str = "separate",
    ) -> ColumnTransformer:
    """Creates a ColumnTransformer pipeline for neural network preprocessing.

    - Categorical features are encoded using OrdinalNNEncoder.
    - Numeric features are imputed with the median and standard-scaled.
    - Binary features are passed through
    """

    # Categorical Pipeline
    cat_pipeline = Pipeline(
        steps=[("ordinal_nn",
                OrdinalNNEncoder(encode_missing_unknown=encode_missing_unknown,
                                 dtype=np.int64))]
    )

    # Numeric Pipeline
    num_pipeline = Pipeline(steps=[("scaler", RobustScaler()),])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_features),
            ("num", num_pipeline, numeric_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    
    return preprocessor


