# Coffee quality rating prediction 
# author: Michelle Wang, Arlin Cherian
# date: 2021-01-01

all: 

# download data
data/raw/coffee_ratings.csv: src/download_data.py
	python src/download_data.py --url=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv --out_file=data/raw/coffee_ratings.csv

# pre-process data 
data/processed/train_df.csv data/processed/test_df.csv: src/prepare_data.py data/raw/coffee_ratings.csv
	python src/prepare_data.py --input_data=data/raw/coffee_ratings.csv --out_dir=data/processed/
# exploratory data-analysis 
results/images/imagescorrelation_matrix_heatmap.png results/images/imagestarget_histogram.png: 
	python src/plot_visualisations.py --input_data='data/processed/train_df.csv' --out_dir='results/images/'

# model building and testing 



# render final report 



clean:
	