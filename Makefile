# Makefile for Coffee Quality Rating Prediction and Analysis 
# author: Michelle Wang, Arlin Cherian
# date: 2021-12-01

# This driver script completes the analysis for Coffee Quality Rating Predictions on the arabica coffee dataset. 
# How it works:
# It first downloads raw data into (into data/raw folder).
# Then it processes and cleans the data into train and test sets (into data/processed folder)
# The exploratory data-analysis script uses train set to produce EDA plots (correlation matrix heatmap, and histogram plot of target).
# Followed by model building script which outputs a table (of model performance scores) and 2 images.
# Finally, the final report HTML file output will be produced in the reports folder.

# This script takes no arguments.

# example usage:
# make all

all : reports/coffee_rating_prediction_report.html

# download data
data/raw/coffee_ratings.csv: src/download_data.py
	python src/download_data.py --url=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv --out_file=data/raw/coffee_ratings.csv

# pre-process data 
data/processed/train_df.csv data/processed/test_df.csv: src/prepare_data.py data/raw/coffee_ratings.csv
	python src/prepare_data.py --input_data=data/raw/coffee_ratings.csv --out_dir=data/processed/

# exploratory data-analysis 
results/images/correlation_matrix_heatmap.png results/images/target_histogram.png: src/plot_visualisations.py data/processed/train_df.csv
	python src/plot_visualisations.py --input_data='data/processed/train_df.csv' --out_dir='results/images/'

# model building and testing 
results/model_comparison.csv results/images/feature_importance_rfr_plot.png results/images/feature_importance_rfc_plot.png: src/ml_analysis.py data/processed/train_df.csv data/processed/test_df.csv
	python src/ml_analysis.py --train="data/processed/train_df.csv" --test="data/processed/test_df.csv" --table_file="results/model_comparison.csv" --out_dir="results/images/"

# render final report 
reports/coffee_rating_prediction_report.html: reports/coffee_rating_prediction_report.rmd \
results/images/correlation_matrix_heatmap.png \
results/images/target_histogram.png \
results/model_comparison.csv \
results/images/feature_importance_rfr_plot.png \
results/images/feature_importance_rfc_plot.png 
	Rscript -e "rmarkdown::render('reports/coffee_rating_prediction_report.rmd', output_format = 'html_document')"

# Usage: make clean (removes all results, images and report outputs)
clean:
	rm -rf results/images/correlation_matrix_heatmap.png results/images/target_histogram.png
	rm -rf results/model_comparison.csv results/images/feature_importance_rfr_plot.png results/images/feature_importance_rfc_plot.png
	rm -rf reports/coffee_rating_prediction_report.html  
	rm -rf data/raw/coffee_ratings.csv data/processed/train_df.csv data/processed/test_df.csv