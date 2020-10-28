#!/usr/bin/env bash

wget "https://www.dropbox.com/s/a0jd4xlsqdloa1k/test_df.csv?" -O test_df.csv t-P random_forest_model/random_forest_model/input/
wget "https://www.dropbox.com/s/pcf72af8ywisai6/train_folds.csv?dl=0" -O train_folds.csv -P random_forest_model/random_forest_model/input/