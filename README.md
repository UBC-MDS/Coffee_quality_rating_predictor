# Coffee Quality Predictor

*__Creators__: Kristin Bunyan, Arlin Cherian, Berkay Bulut, Michelle Wang*


## Introduction 

This data analysis was performed as part of the DSCI 522, Data Science Workflows course at the University of British Columbia's Master of Data Science Program. Our team will use machine learning techniques to predict coffee quality on arabica coffees. We will be using an exploratory data analysis roadmap and predictive modeling to create this project. 


## About

### Summary

In this analysis, we attempt to find a supervised machine learning model which uses the features of the Coffee Quality Dataset, collected by the Coffee Quality Institute in January 2018, to predict the quality of a cup of arabica coffee to answer the research question: **given a set characteristics, what is the quality of a cup of arabica coffee?** 

We begin our analysis by exploring the natural inferential sub-question of which features correlate strongly with coffee quality, which will help to inform our secondary inferential sub-question: which features are most influential in determining coffee quality? We then begin to build our models for testing.

After initially exploring regression based models, Ridge Regression and Random Forest Regressor, our analysis deviated to re-processing our data and exploring classification models. As you will see in our analysis below, predicting a continuous target variable proved quite difficult with many nonlinear features, and was not very interpretable in a real sense of what we were trying to predict. Broadening the target variable and transforming it into classes: “Good” and “Poor”, based on a threshold at the median, helped with these issues. 

Our final model, using Random Forest Classification, performed averagely on an unseen test data set, with an ROC score of 0.67. We recommend continuing to study to improve this prediction model before it is put to any use, as incorrectly classifying the quality of coffee could have a large economic impact on a producers income. We have described how one might do that at the end of our analysis.
 
***

### Dataset
We will be analyzing the *[Coffee Quality Dataset](https://github.com/jldbc/coffee-quality-database)*, collected by the Coffee Quality Institute in January 2018. The data was retrieved from tidytuesday, courtesy of James LeDoux, a Data Scientist at Buzzfeed (DeLoux, J). The data is collected on Arabica coffee beans from across the world and professionally rated on a 0-100 scale based on factors like acidity, sweetness, fragrance, balance, etc. The dataset also contains information about coffee bean origin country, harvesting and grading date, colour, defects, processing and packaging details.  
* [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
* Data format: csv file, 1311 rows and 44 columns.
***

### Report
The final report can be found *[here](https://rpubs.com/acherian/840439)*.
***

### Usage
To replicate the analysis, do the following: 
1. clone this GitHub repository
2. create a conda environment with all the dependencies using the environment.yaml file:
    - conda env create -f environment.yaml
    - conda activate 522_group_03
3. run this script to install the R-dependencies:
    - Rscript -e 'install.packages("knitr", repos="https://cloud.r-project.org")'
4. Follow the commands below, in your terminal:


    # download data
    python src/download_data.py --url=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv --  out_file=data/raw/coffee_ratings.csv

    # pre-process data
    python src/prepare_data.py --input_data=data/raw/coffee_ratings.csv --out_dir=data/processed/
    
    # run eda analysis
    python src/plot_visualisations.py --input_data='data/processed/train_df.csv' --out_dir='results/images/'
    
    # run the main analysis
    python src/ml_analysis.py --train="data/processed/train_df.csv" --test="data/processed/test_df.csv" --table_file="results/model_comparison.csv" --out_dir="results/images"

    # render final report
    Rscript -e "rmarkdown::render('reports/coffee_rating_prediction_report.rmd', output_format = 'html_document')"


## Dependencies 

- Python 3.7.4 and Python packages:
  - docopt==0.6.2
  - pandas==1.3.3
  - scikit-learn==1.0
  - requests==2.24.0
  - altair==4.1.0
  - altair-saver==0.5.0
  - seaborn=0.11.2
  - selenium=4.1.0
  - numpy==1.21.2
- R version 4.1.1 and packages:
  - knitr==1.36
***

## References

DeLoux, J. "coffee-quality-database" June 2018. <https://github.com/jldbc/coffee-quality-database>

Coffee Quality Institute's review pages, January 2018. <https://database.coffeeinstitute.org/>

