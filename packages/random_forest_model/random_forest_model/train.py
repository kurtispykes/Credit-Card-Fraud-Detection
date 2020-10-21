import logging

from sklearn import preprocessing, model_selection, ensemble, metrics

from random_forest_model.config import config
from random_forest_model.preprocessors import data_management as dm
from random_forest_model.preprocessors import preprocessing as pp

_logger = logging.getLogger(__name__)

def run_training() -> None:
    """
    Training the  model
    """
    # read training and test data
    train_transactions, train_identity, test_transactions, test_identity = dm.read_all_data()

    # merge the datasets
    df_train = dm.merge_data(df1=train_transactions,
                             df2=train_identity,
                             train_data=True)
    df_test = dm.merge_data(df1=test_transactions,
                            df2=test_identity,
                            train_data=False)

    del train_transactions, train_identity, test_transactions, test_identity

    # fix typo in id columns of test data
    df_test = pp.rename_id_col(df_test)

    # save training data
    df_train.to_csv(config.TRAINING_DATA)

    # set predictors and target
    X = df_train.drop(config.TARGET, axis=1)
    y = df_train[config.TARGET]

    # cross validatioon
    label_encoders = {}
    skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X=X, y=y)):
        X_train, X_valid = X.loc[train_idx, :], X.loc[test_idx, :]
        y_train, y_valid = y.loc[train_idx], y.loc[test_idx]
        # label encoding for categorical features
        for c in config.CATEGORICAL_FEATURES:
            lbl = preprocessing.LabelEncoder()
            X_train.loc[:, c] = X_train.loc[:, c].astype(str)
            X_valid.loc[:, c] = X_valid.loc[:, c].astype(str)
            df_test.loc[:, c] = df_test.loc[:, c].astype(str)
            lbl.fit(X_train[c].values.tolist() +
                    X_valid[c].values.tolist() +
                    df_test[c].values.tolist())
            X_train.loc[:, c] = lbl.transform(X_train[c].values.tolist())
            X_valid.loc[:, c] = lbl.transform(X_valid[c].values.tolist())
            df_test.loc[:, c] = lbl.transform(df_test[c].values.tolist())
            label_encoders[c] = lbl
        # saving the label encoder
        dm.save_pipeline(save_file_name=f"{config.ENCODERS_NAME}{fold}_v", to_persist=label_encoders)

        # modelling
        rf = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=True, random_state=24)
        rf.fit(X_train, y_train)
        preds = rf.predict_proba(X_valid)[:, 1]
        _logger.info(f"Fold {fold} ROC_AUC Score: {metrics.roc_auc_score(y_valid, preds)}")
        dm.save_pipeline(save_file_name=f"{config.MODEL_NAME}{fold}_v", to_persist=rf)


if __name__ == "__main__":
    run_training()
