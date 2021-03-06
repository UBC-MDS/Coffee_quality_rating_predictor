# author: Arlin Cherian, Kristin Bunyan, Michelle Wang, Berkay Bulut
# date: 2021-11-24

"""Runs visualisation such as histogram, boxplots on processed data of Coffee Quality Database dataset (https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-07-07/readme.md).
Saves the images to report folder

Usage: src/plot_visualisations.py --input_data=<input_data> --out_dir=<out_dir>

Options:
--input_data=<input_data>  Path (including filename) to feed data to visualisation pipeline
--out_dir=<out_dir>   Path to directory where the processed images will be saved in
" -> doc
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from docopt import docopt

opt = docopt(__doc__)


def plot_target_histogram(
    dataframe: pd.DataFrame, target_feature: str, output_dir: str = "../results/images/"
):
    """plot_target_histogram [summary]

    Plots histogram of target value

    Args:
        dataframe (pd.DataFrame): Dataframe to use for plotting
        target_feature (str): Feature name accepted as target
        output_dir (str, optional): Directory output for image saving. Defaults to "../results/images/".
    """
    plt.figure(figsize=(16, 6))

    # Histogram Plot of Target Variable
    sns.histplot(dataframe, x=target_feature, bins=20).set_title(
        "Distribution of total cup points (target variable)", weight="bold"
    )
    plt.xlabel("Total cup points")
    plt.ylabel("Count")

    plt.savefig(output_dir + "target_histogram.png")
    print("Target Feature Histogram Plotted")


def plot_correlation_matrix(
    dataframe: pd.DataFrame, output_dir: str = "../results/images/"
):
    """plot_correlation_matrix [summary]

    Plots correlation matrix of features

    Args:
        dataframe (pd.DataFrame): Data to be used in correlation
        output_dir (str, optional): Directory output for image saving. Defaults to "../results/images/".

    Returns:
        [type]: Images Plots and Print Statement
    """
   # dataframe = pd.read_csv("data/processed/train_df.csv")
    assert type(dataframe) == pd.core.frame.DataFrame
    print("Success")
    
    assert type(output_dir) == str
    print("Success")
    
    graphing = dataframe.copy()
    graphing = graphing.drop(columns = ["category_one_defects", "category_two_defects"])

    # Correlation Plot - Diagonal Removed
    plt.figure(figsize=(16, 6))

    correlation_matrix = graphing.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=np.bool_))

    sns.heatmap(
        correlation_matrix, mask=mask, vmin=-1, vmax=1, annot=True, cmap="BrBG"
    ).set_title("Correlation Heatmap of numerical variables with target variable", fontdict={"fontsize": 12}, pad=12)
 
    plt.savefig(output_dir + "correlation_matrix_heatmap.png")

    return print("Correlation Matrix Plotted")


def plot_visualisations(input_data, output_dir):
    """plot_visualisations

    Runs the pipeline to plot visualisations

    Args:
        input_data ([str]): Directory of data including file name
        output_dir ([str]): Directory to save plots in
    """

    # Read Dataframe
    dataframe = pd.read_csv(input_data)

    # Run pipelines
    plot_target_histogram(dataframe, "total_cup_points", output_dir)
    plot_correlation_matrix(dataframe, output_dir)


if __name__ == "__main__":
    plot_visualisations(opt["--input_data"], opt["--out_dir"])
