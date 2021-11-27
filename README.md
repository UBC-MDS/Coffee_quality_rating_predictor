# Coffee Quality Predictor

*__Creators__: Kristin Bunyan, Arlin Cherian, Berkay Bulut, Michelle Wang*


## Introduction 

This data analysis was performed as part of the DSCI 522, Data Science Workflows course at the University of British Columbia's Master of Data Science Program. Our team will use machine learning techniques to predict coffee quality scores on arabica coffees. We will be using an exploratory data analysis roadmap and predictive modeling to create this project. 

***

## Dataset

We will be analyzing the *[Coffee Quality Dataset](https://github.com/jldbc/coffee-quality-database)*, collected by the Coffee Quality Institute in January 2018. The data was retrieved from tidytuesday, courtesy of James LeDoux, a Data Scientist at Buzzfeed. The data is collected on Arabica coffee beans from across the world and professionally rated on a 0-100 scale based on factors like acidity, sweetness, fragrance, balance, etc. The dataset also contains information about coffee bean origin country, harvesting and grading date, colour, defects, processing and packaging details.  
* [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
* Data format: csv file, 1311 rows and 44 columns
***

## Research Question

The research question that we will be attempting to answer in this analysis is: **given a set characteristics, what is the quality of a cup of arabica coffee?**

Our primary goal is to be able to predict the quality of a cup of arabica coffee globally, using the given categories provided in the coffee survey from the Coffee Quality Institute. We will begin our analysis by exploring the natural inferential sub-question of which features correlate strongly with coffee quality, which will help to inform our secondary inferential sub-question: which feature is the most influential in determining coffee quality. 
***

## Plan for Analysis

### Approach
To address our predictive research question, we will be building and comparing three machine learning models using pythons sklearn. Since the outcome we are trying to predict is continuous, the approaches we are comparing will be limited to the following models: Linear Regression with Ridge, Random Forest Regressor and Decision Tree Regressor.

### Exploratory EDA
We will first establish from our data our target and feature variables. Our target will be `total_cup_points` and our features will be `country_of_origin`, `region`, `harvest_year`, `grading_date`, `processing_method`, `moisture`, `category_one_defects`, `quakers`, `color`, `category_two_defects`,`expiration`, and `altitude_mean_meters`. For our analysis, the data is be sectioned into a training and test set split of 80% and 20% of the original data, respectively.

To get a sense of our data we have performed some exploratory EDA [here](https://github.com/UBC-MDS/DSCI_522_GROUP3_COFFEERATINGS/blob/main/src/coffee_rating.ipynb), in which we plotted distributions of select categorical features, `country_of_origin` and `color`, from the training dataset. In doing so, we see that the average coffee rating differed between the various countries of origin: the highest was from Ethiopia and the lowest was from Haiti. However, we didn’t see much variation in the average coffee quality rating among various colours of the coffee beans. To explore our first inferential sub-question, we created a scatterplot matrix of the numeric features. We hope to expand this to explore the relationship between the categorical features and the total coffee quality rating with additional cleaning and preprocessing of the dataset at a later date.

For altair to save images in png format we need to install additional dependencies. We will be using the [altair](https://altair-viz.github.io/) library to create our figures.

`pip install altair_saver`
`pip install selenium`
`brew install geckodriver`


To execute our exploratory visualisations, we will use the following code:

`python src/plot_visualisations.py --input_data='data/processed/train_df.csv' --out_dir='reports/images/`

### Exploratory Visualisation
We will now begin our analysis by exploring the relationship between the numeric features and the total coffee quality rating. We will use the following visualisations: (1) a histogram of how the rating has changed for the target variable, (2) a scatterplot matrix of the numeric features and the target variable, and (3) a heatmap of the correlation between the numeric features and the target variable.

Post visualisation, we iterated over data cleaning steps, and added new processing parameters to remove outliers from our dataset. This has significantly improved our dataset in terms of data quality which will be used to train our models.

### Model Building and Hyperparameter Optimization
These next steps describe how we plan to build and evaluate our models to carry out our analysis.

Given that we have a mix of features that are both categorical and continuous we will need to clean, transform and scale our data appropriately for each model with column transformers and pipelines, so the features can be input effectively.  

We will be performing hyperparameter optimization using a RandomGridSearchCV for each of the following:
- Linear Regression with Ridge:
    - 'alpha'
- Random Forest Regressor:
    - 'bootstrap','max_depth','max_features', 'min_samples_leaf','min_samples_split','n_estimators'
- Decision Tree Regressor:
    - 'max_depth', 'max_features', 'max_leaf_nodes', 'min_samples_leaf', 'min_weight_fraction_leaf', 'splitter'
    
The hyperparameters for each model will be selected by running cross-validation of 50 folds. We feel this number of folds is appropriate given that our dataset is not very large, with 1311 observations before the removal of any null or erroneous values. We will then compare the best performing hyperparameter combination for each model to the other models using a similar cross-validation approach, and select the best performing model based on it's mean cross-validation test score. A line plot of each model's cross-validation test scores will be included in the final report for this analysis as a means of sharing the results.

### Model Performance Assessment
After selecting our best performing model, we will re-fit that model on the entire training set, and score it on the test set to assess it's performance. 

To end our analysis, we will re-visit our secondary sub-question and extract the coefficients or feature importance of each feature using the appropriate attribute for the final model, and visualize these results into a dataframe.
***

## Anticipated Dependencies 
(as of Nov. 19 2021, subject to change throughout project)

  - Python 3.7.4 and Python packages:
      - docopt=0.6.2
      - pandas=1.3.3
      - scikit-learn=1.0
      - requests=2.24.0
      - altair=4.1.0
      - altair_saver=0.5.0
      - seaborn=0.11.2
      - selenium=4.1.0
***
## References

[1] DeLoux, J. "coffee-quality-database" June 2018. <https://github.com/jldbc/coffee-quality-database>

[2] Coffee Quality Institute's review pages, January 2018. <https://database.coffeeinstitute.org/>

