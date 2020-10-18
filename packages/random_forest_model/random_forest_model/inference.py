import logging

import pandas as pd

from random_forest_model.config import config
from random_forest_model.preprocessors import data_management as dm
from random_forest_model import __version__ as _version

_logger = logging.getLogger(__name__)

def predict(input_data) -> dict:
    """Make a prediction using the saved model"""
    df = pd.DataFrame(input_data)
    predictions = None

    for FOLD in range(5):
        encoders = dm.load_pipeline(save_file_name=f"{config.ENCODERS_NAME}{FOLD}_v{_version}.pkl")
        clf = dm.load_pipeline(save_file_name=f"{config.MODEL_NAME}{FOLD}_v{_version}.pkl")
        for c in encoders:
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df.loc[:, c].values.tolist())

        preds = clf.predict_proba(df.fillna(0))[:, 1]

        _logger.info(
            f"Making Predictions with model fold: {FOLD}",
            f"Inputs: {df.fillna(0)}",
            f"Predictions: {preds}"
        )

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5
    response = {"predictions": predictions}

    _logger.info(f"Model Version: {_version}")
    return response