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
    "ProductCD", "card1", "card2", "card3", "card4",
    "card5", "card6", "addr1", "addr2", "P_emaildomain",
    "R_emaildomain", "M1", "M2", "M3", "M4", "M5",
    "M6", "M7", "M8", "M9", "DeviceType", "DeviceInfo",
    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17",
    "id_18", "id_19", "id_20", "id_21", "id_22", "id_23",
    "id_24", "id_25", "id_26", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_32", "id_33", "id_34", "id_35",
    "id_36", "id_37", "id_38"
]

# Target
TARGET = "isFraud"

# Models
MODEL_NAME = "random_forest_"
ENCODERS_NAME = "label_encoder_"



