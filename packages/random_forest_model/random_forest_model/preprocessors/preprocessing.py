import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# https://github.com/trainindata/deploying-machine-learning-models/blob/53bc67c6a94e01f3fdaf05cbfa2b49465a0c7a1f/packages/regression_model/regression_model/processing/preprocessors.py
class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

def rename_id_col(df: pd.DataFrame):
    """Renaming the id column."""
    for col in df.columns:
        if "id" in col:
            df.rename(columns={col: col.replace("-", "_")}, inplace=True)
    return df
