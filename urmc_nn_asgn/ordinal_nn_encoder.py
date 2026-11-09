# urmc_nn_asgn/ordinal_nn_encoder.py
#
# Defines an sklearn pipeline transformer that does ordinal encoding,
# but maps missing/unknown to 0/1 for NN applications

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class OrdinalNNEncoder(BaseEstimator, TransformerMixin):
    """
    Ordinal Encoder wrapper with custom missing and unknown value handling, to assist with 
    encodings for neural networks.

    Parameters
    ----------
    encode_missing_unknown : 
        {'none', 'missing_only', 'unknown_only', 'separate', 'combined'}, default='none'
        Strategy for handling missing and unknown values.
    categories : 'auto' or list of array-like, default='auto'
        Categories per feature passed directly to underlying OrdinalEncoder.
    dtype : number type, default=np.float64
        Desired dtype of output.
    """

    ALLOWED_STRATEGIES = {'none', 'missing_only', 'unknown_only', 'separate', 'combined'}

    def __init__(
        self,
        encode_missing_unknown='none',
        categories='auto',
        dtype=np.float64,
    ):
        self.encode_missing_unknown = encode_missing_unknown
        self.categories = categories
        self.dtype = dtype

    def _validate_strategy(self):
        if self.encode_missing_unknown not in self.ALLOWED_STRATEGIES:
            raise ValueError(
                f"Invalid encode_missing_unknown='{self.encode_missing_unknown}'. "
                f"Must be one of {sorted(self.ALLOWED_STRATEGIES)}."
            )

    def _set_encoded_attributes(self):
        """Sets encoded_missing_value and encoded_unknown_value attributes."""
        strategy = self.encode_missing_unknown

        if strategy == 'none':
            self.encoded_missing_value_ = None
            self.encoded_unknown_value_ = None
            self.shift_value_ = 0
        elif strategy == 'missing_only':
            self.encoded_missing_value_ = 0.0
            self.encoded_unknown_value_ = None
            self.shift_value_ = 1
        elif strategy == 'unknown_only':
            self.encoded_missing_value_ = None
            self.encoded_unknown_value_ = 0.0
            self.shift_value_ = 1
        elif strategy == 'separate':
            self.encoded_missing_value_ = 0.0
            self.encoded_unknown_value_ = 1.0
            self.shift_value_ = 2
        elif strategy == 'combined':
            self.encoded_missing_value_ = 0.0
            self.encoded_unknown_value_ = 0.0
            self.shift_value_ = 1
            

    def fit(self, X, y=None):
        self._validate_strategy()
        self._set_encoded_attributes()
        
        strategy = self.encode_missing_unknown
        
        # Missing handling
        if self.encoded_missing_value_ is None:
            encoded_missing_value=np.nan
        elif strategy == 'separate':
            encoded_missing_value = -2
        else:
            encoded_missing_value = -1

        # Unknown handling
        if self.encoded_unknown_value_ is None:
            handle_unknown = 'error'
            unknown_value = None
        else:
            handle_unknown = 'use_encoded_value'
            unknown_value = -1
            
        self.encoder_ = OrdinalEncoder(
            categories=self.categories,
            dtype=self.dtype,
            handle_unknown=handle_unknown,
            unknown_value=unknown_value,
            encoded_missing_value=encoded_missing_value,
        )

        self.encoder_.fit(X, y)

        # Retain feature names and categories from wrapped encoder
        if hasattr(self.encoder_, 'feature_names_in_'):
            self.feature_names_in_ = self.encoder_.feature_names_in_
        self.categories_ = self.encoder_.categories_
        
        # Calculate category counts & max encoded value per feature
        # (Useful for setting PyTorch nn.Embedding input dimensions)
        self.n_categories_ = np.array([
            len(cats) + self.shift_value_ for cats in self.categories_
        ])
        
        # Max encoded value per column (0-indexed, so n_categories - 1)
        self.max_encoded_value_ = self.n_categories_ - 1

        return self

    def transform(self, X):
        self._validate_strategy()

        # Get raw output from underlying OrdinalEncoder
        X_trans = self.encoder_.transform(X)
        
        # Shift the output
        X_out =  X_trans + self.shift_value_
    
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_out, columns=X.columns, index=X.index)

        return X_out

    def get_feature_names_out(self, input_features=None):
        return self.encoder_.get_feature_names_out(input_features)