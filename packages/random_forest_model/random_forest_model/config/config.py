import pathlib

import random_forest_model

# Core Directory
PACKAGE_ROOT = pathlib.Path(random_forest_model.__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "input"
MODEL_DIR = PACKAGE_ROOT / "models"

# Training data
TRAINING_DATA = DATA_DIR / "train_folds.csv"
TRAIN_TRANSACTIONS = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = DATA_DIR / "train_identity.csv"

# Test data
TEST_DATA = DATA_DIR / "test_df.csv"
TEST_TRANSACTIONS = DATA_DIR / "test_transaction.csv"
TEST_IDENTITY = DATA_DIR / "test_identity.csv"

# ID
ID = "TransactionID"

# Categorical Features
CATEGORICAL_FEATURES = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1',
    'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15',
    'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33',
    'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
    'DeviceInfo'
]

# Target
TARGET = "isFraud"

# Models
MODEL_NAME = "random_forest_"
ENCODERS_NAME = "label_encoder_"



