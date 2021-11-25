# author: Arlin Cherian, Michelle Wang
# date: 2021-11-24

"""Fits a Linear Regression Ridge Model and Random Forest Regressor model 
on the pre-processed training data on coffee quality rating and saves the output table and images.
Usage: src/ml_analysis.py --train=<train> --test=<test> --table_file=<table_file>
  
Options:
--train=<train>             Path (including filename) to training data in csv format
--test=<test>               Path (including filename) to testing data in csv format
--table_file=<table_file>   Path (including filename) where results table should be written
"""

import os
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
from docopt import docopt
from sklearn import datasets
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy.stats import loguniform
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)

opt = docopt(__doc__)

# train = "data/processed/train_df.csv"
# test = "data/processed/test_df.csv"
# table = "../results/model_comparison.csv"
# image = "../results/feat_importance.png"



def main(train, test, table_file):

    # Define cross val function
    def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
        """
        Returns mean and std of cross validation

        Parameters
        ----------
        model :
            scikit-learn model
        X_train : numpy array or pandas DataFrame
            X in the training data
        y_train :
            y in the training data

        Returns
        ----------
            pandas Series with mean scores from cross_validation
        """
        scores = cross_validate(model, X_train, y_train, **kwargs)

        mean_scores = pd.DataFrame(scores).mean()
        std_scores = pd.DataFrame(scores).std()
        out_col = []

        for i in range(len(mean_scores)):
            out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

        return pd.Series(data=out_col, index=mean_scores.index)
        

    # Split into x and y
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
    
    X_train = train_df.drop(columns=["total_cup_points"])
    X_test = test_df.drop(columns=["total_cup_points"])

    y_train = train_df["total_cup_points"]
    y_test = test_df["total_cup_points"]
    
    # Create preprocessor
    numeric_features = [
    "moisture",
    "quakers",
    "altitude_mean_meters"
    ]

    categorical_features = [
        "country_of_origin",
        "harvest_year",
        "variety",
        "processing_method",
        "category_one_defects",
        "color",
        "category_two_defects",
        "region"
        ]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features)
    )

    # Building regression models
    models = {
        "Ridge": Ridge(),
        "RForest_Regressor": RandomForestRegressor(random_state=123)
    }
    results_dict = {}
    for k, v in models.items():
        pipe_multi = make_pipeline(preprocessor, v)
        models_score = mean_std_cross_val_scores(pipe_multi, X_train, y_train, 
                                                cv=5, return_train_score=True, error_score='raise')
        results_dict[k] = models_score
        pd.DataFrame.from_dict(results_dict)

    # Table output: Model comparison
    results = pd.DataFrame(results_dict)
    results.to_csv(table_file)


if __name__ == "__main__":
    main(opt["--train"], opt["--test"], opt["--table_file"])