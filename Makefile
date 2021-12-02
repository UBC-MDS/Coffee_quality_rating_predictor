# Makefile for Coffee quality rating prediction 
# author: Michelle Wang, Arlin Cherian
# date: 2021-01-01

# This driver script completes the analysis for coffee predictions on arabica coffee dataset.
# This script takes no arguments.

# example usage:
# make all

all:reports/coffee_rating_prediction_report.html

# download data
data/raw/coffee_ratings.csv: src/download_data.py
    python src/download_data.py --url=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv --out_file=data/raw/coffee_ratings.csv

# pre-process data 
data/processed/train_df.csv data/processed/test_df.csv: src/prepare_data.py data/raw/coffee_ratings.csv
    python src/prepare_data.py --input_data=data/raw/coffee_ratings.csv --out_dir=data/processed/

# exploratory data-analysis 
results/images/imagescorrelation_matrix_heatmap.png results/images/imagestarget_histogram.png: src/plot_visualisations.py data/processed/train_df.csv
	python src/plot_visualisations.py --input_data='data/processed/train_df.csv' --out_dir='results/images/'

# model building and testing 
results/model_comparison.csv results/images/feature_importance_rfr_plot.png results/images/feature_importance_rfc_plot.png: src/ml_analysis.py data/processed/train_df.csv
	python src/ml_analysis.py --train="data/processed/train_df.csv" --test="data/processed/test_df.csv" --table_file="results/model_comparison.csv" --out_dir="results/images/"

# render final report 
reports/coffee_rating_prediction_report.html: reports/coffee_rating_prediction_report.rmd 
	Rscript -e "rmarkdown::render('reports/coffee_rating_prediction_report.rmd', output_format = 'html_document')"

# Usage: make clean 
clean :
	rm -rf results/images/imagescorrelation_matrix_heatmap.png results/images/imagestarget_histogram.png 
	rm -rf results/model_comparison.csv results/images/feature_importance_rfr_plot.png results/images/feature_importance_rfc_plot.png
	rm -rf reports/coffee_rating_prediction_report.html