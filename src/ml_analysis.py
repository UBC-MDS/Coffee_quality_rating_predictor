# Authors: Arlin Cherian, Michelle Wang
# Created date: 2021-11-24

"""Fits a Ridge Regression Model, Random Forest Regressor model and a Random Forest Classification Model
on the pre-processed training data on coffee quality rating and saves the output table and images.
Usage: src/ml_analysis_script.py --train=<train> --test=<test> --table_file=<table_file> --image1_file=<image1_file> --image2_file=<image2_file>
  
Options:
--train=<train>             Path (including filename) to training data in csv format
--test=<test>               Path (including filename) to testing data in csv format
--table_file=<table_file>   Path (including filename) where results table should be written
—-image1_file=<image1_file>   Path (including filename) where results figures should be saved
—-image2_file=<image2_file>   Path (including filename) where results figures should be saved
"""

import os
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
from docopt import docopt
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
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
# table = "results/model_comparison.csv"
# image1 = "results/feature_importance_rfr_plot.png"
# image2 = "results/feature_importance_rfc_plot2.png"


def main(train, test, table_file, image1_file, image2_file):
    """
    Runs through all functions involved in the regression and classification 
    analysis and generate output tables and figures.
  
    Parameters:
    ----------
    train: (filepath) file path to the training data
    test: (filepath) file path to the test data
    table_file : (filepath) file path to the output table
    output: (filepath) file path to the output figures
    """
    # Reading in the train and test data and splitting 
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)

    X_train = train_df.drop(columns=["total_cup_points"])
    X_test = test_df.drop(columns=["total_cup_points"])

    y_train = train_df["total_cup_points"]
    y_test = test_df["total_cup_points"]

    # Create a preprocessor for feature transformations

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
        

# Building regression models

    models = {
    "Ridge": Ridge(random_state=123),
    "RForest_Regressor": RandomForestRegressor(random_state=123)
    }

    results_dict = {}

    for k, v in models.items():
        pipe_multi = make_pipeline(preprocessor, v)
        results_dict[k] = mean_std_cross_val_scores(pipe_multi, X_train, y_train, 
                                             cv=5, return_train_score=True)
    
    
    # Table output: Model comparison
    results_dict = pd.DataFrame(results_dict)

    # ==> RESULT: We select RF Regressor with better CV performance: 0.177 for R squared 
    # ==> NEXT: We try hyperparameter optimization on RF

    # Hyperparameter Optimization for RF Regressor Model

    param_dist = {'randomforestregressor__max_depth': np.arange(1, 20),
              'randomforestregressor__max_features': np.arange(1, 124),
              'randomforestregressor__n_estimators': np.arange(100, 1000, 100)
               }

    pipe = make_pipeline(preprocessor, RandomForestRegressor())

    random_search = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_jobs=-1,
    n_iter=10, cv=5, random_state=123,
    return_train_score=True)

    random_search.fit(X_train, y_train)
    random_search.best_score_

    # ==> RESULT: Hyperparameter optimization ends up with better score of ~ 0.25
    # ==> NEXT: We view the feature importance using parameters by this best estimator

    # Inspect important features from best RF model

    X_transformed = preprocessor.fit_transform(X_train)

    column_names = (
        numeric_features +
        preprocessor.named_transformers_["onehotencoder"].get_feature_names_out().tolist()
    )

    # Top 10 features
    importances = random_search.best_estimator_['randomforestregressor'].feature_importances_
    feat_df = pd.DataFrame({'features': column_names, 'importances': importances})
    feat_df = feat_df.sort_values('importances', ascending=False)[:5]

    # --> SECOND OUTPUT: Barplot of feature importances
    sns.barplot(x="importances", y="features", data=feat_df, color="salmon").set_title('Feature Importance from RF Regression', weight='bold')
    plt.xlabel("Importances")
    plt.ylabel("Features")
    
    # Saving plot as an output           
    plt.savefig(image1_file, bbox_inches = "tight")


    # Testing a classfication model on train dataset

    train_df['total_cup_grade'] = train_df['total_cup_points'].apply(lambda x: 'Good' if x>82 else 'Poor')
    test_df['total_cup_grade'] = test_df['total_cup_points'].apply(lambda x: 'Good' if x>82 else 'Poor')

    X_train_new = train_df.drop(columns=["total_cup_points", "total_cup_grade"])
    X_test_new = test_df.drop(columns=["total_cup_points", "total_cup_grade"])

    y_train_new = train_df["total_cup_grade"]
    y_test_new = test_df["total_cup_grade"]


    # Building a classfication pipeline 
    pipe = make_pipeline(preprocessor, RandomForestClassifier())

    results_dict["RF_classification"] = pd.DataFrame(mean_std_cross_val_scores(pipe, X_train_new, y_train_new, cv=10, 
                                       return_train_score=True, scoring='roc_auc'))
    results_dict.to_csv(table_file)


    # Inspect important features from best RF model

    pipe.fit(X_train_new, y_train_new)
    pipe['randomforestclassifier'].feature_importances_

    column_names = (
        numeric_features +
        preprocessor.named_transformers_["onehotencoder"].get_feature_names_out().tolist()
        )

    # Top 5 features
    importances = pipe['randomforestclassifier'].feature_importances_
    feat_df = pd.DataFrame({'features': column_names, 'importances': importances})
    feat_df = feat_df.sort_values('importances', ascending=False)[:5]

    # --> Third OUTPUT: Barplot of feature importances
    sns.barplot(x="importances", y="features", data=feat_df, color="salmon").set_title('Feature Importance from RF Classifier', weight='bold')
    plt.xlabel("Importances")
    plt.ylabel("Features")
    
    # Saving plot as an output           
    plt.savefig(image2_file, bbox_inches = "tight")
    
    # Testing performance of best regression model on test set
    random_search.score(X_test, y_test)

    # Testing performance of classification model on test set
    pipe.score(X_test_new, y_test_new)

# Call the main function
if __name__ == "__main__":
    main(opt["--train"], opt["--test"], opt["--table_file"], opt["--image1_file"], opt["--image2_file"])
