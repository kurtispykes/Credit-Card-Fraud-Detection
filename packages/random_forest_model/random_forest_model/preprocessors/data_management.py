import logging

import joblib
import pandas as pd

from random_forest_model.config import config
from random_forest_model.preprocessors import preprocessing as pp
from random_forest_model import __version__ as _version

_logger = logging.getLogger(__name__)

def read_all_data():
    """Read all data files into memory."""
    train_transactions = pd.read_csv(config.TRAIN_TRANSACTIONS)
    train_identity = pd.read_csv(config.TRAIN_IDENTITY)
    test_transactions = pd.read_csv(config.TEST_TRANSACTIONS)
    test_identity = pd.read_csv(config.TEST_IDENTITY)
    return train_transactions, train_identity, test_transactions, test_identity

def read_test_data():
    """Read only the configures test file"""
    test_data = pd.read_csv(config.TEST_DATA)
    return test_data

def save_pipeline(save_file_name: str, to_persist: str) -> None:
    """Persist the pipeline"""
    save_file_name_version = f"{save_file_name}{_version}.pkl"
    save_path = config.MODEL_DIR / save_file_name_version
    joblib.dump(value=to_persist, filename=save_path)
    _logger.info(f"Model Saved as: {save_file_name_version}")

def load_pipeline(save_file_name:str, model_path:str= config.MODEL_DIR):
    """Load persisted model"""
    path_to_model = model_path / save_file_name
    saved_model = joblib.load(path_to_model)
    return saved_model

def merge_data(df1:pd.DataFrame, df2:pd.DataFrame, train_data:bool=True) -> None:
    """Merge the data. If it is not train_data then we save to data file"""
    merged_df = df1.merge(df2, how="left", on=config.ID)
    merged_df = pp.rename_id_col(df= merged_df)
    cat_cols = merged_df.select_dtypes(include=['object']).columns
    num_cols = merged_df.select_dtypes(include=['number']).columns
    if train_data:
        merged_df[cat_cols] = merged_df[cat_cols].fillna('NONE')
        merged_df[num_cols] = merged_df[num_cols].fillna(0)
        return merged_df
    else:
        merged_df[cat_cols] = merged_df[cat_cols].fillna('NONE')
        merged_df[num_cols] = merged_df[num_cols].fillna(0)
        merged_df.to_csv(config.TEST_DATA, index=False)
        return merged_df