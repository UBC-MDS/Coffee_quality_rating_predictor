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
import os

import pandas as pd
import numpy as np
import altair as alt

import matplotlib.pyplot as plt
import seaborn as sns

from docopt import docopt

opt = docopt(__doc__)

from selenium import webdriver
from selenium.webdriver.firefox.options import Options as firefox_Options

options = firefox_Options()
options.headless = True
executable_path = "/opt/homebrew/bin/geckodriver"
driver = webdriver.Firefox(options=options, executable_path=executable_path)


def plot_target_histogram(
    dataframe: pd.DataFrame, target_feature: str, output_dir: str = "../reports/images/"
):
    """plot_target_histogram [summary]

    Plots histogram of target value

    Args:
        dataframe (pd.DataFrame): Dataframe to use for plotting
        target_feature (str): Feature name accepted as target
        output_dir (str, optional): Directory output for image saving. Defaults to "../reports/images/".
    """

    # Histogram Plot of Target Variable
    histogram_plot = (
        alt.Chart(dataframe, title="Target variable histogram")
        .mark_bar()
        .encode(
            x=alt.X(f"{target_feature}:Q", bin=True),
            y="count()",
        )
    )

    histogram_plot.save(output_dir + "target_histogram.png", webdriver=driver)
    print("Target Feature Histogram Plotted")


def plot_correlation_matrix(
    dataframe: pd.DataFrame, output_dir: str = "../reports/images/"
):
    """plot_correlation_matrix [summary]

    Plots correlation matrix of features

    Args:
        dataframe (pd.DataFrame): Data to be used in correlation
        output_dir (str, optional): Directory output for image saving. Defaults to "../reports/images/".

    Returns:
        [type]: Images Plots and Print Statement
    """

    # Correlation Plot - Diagonal Removed
    plt.figure(figsize=(16, 6))

    correlation_matrix = dataframe.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=np.bool_))

    heatmap = sns.heatmap(
        correlation_matrix, mask=mask, vmin=-1, vmax=1, annot=True, cmap="BrBG"
    )

    heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 12}, pad=12)

    fig = heatmap.get_figure()
    fig.savefig(output_dir + "correlation_matrix_heatmap.png")

    return print("Correlation Matrix Plotted")


def plot_pairwise(
    dataframe: pd.DataFrame, target_feature: str, output_dir: str = "../reports/images/"
):
    """plot_pairwise [summary]

    Plots pairwise scatter plot fo features with respect to target columns

    Args:
        dataframe (pd.DataFrame): Data to be used in pairwise plot
        target_feature (str): Feature name accepted as target
        output_dir (str, optional): Directory output for image saving. Defaults to "../reports/images/".

    Returns:
        [type]: Images Plots and Print Statement
    """

    splom = (
        alt.Chart(dataframe, title="Explanatory variables pair plot")
        .mark_point(opacity=0.3)
        .encode(
            x=alt.X(
                alt.repeat("column"), type="quantitative", scale=alt.Scale(zero=False)
            ),
            y=alt.Y(
                alt.repeat("row"), type="quantitative", scale=alt.Scale(zero=False)
            ),
        )
        .properties(width=200, height=200)
        .repeat(
            row=[f"{target_feature}"],
            column=[
                "moisture",
                "category_one_defects",
                "quakers",
                "category_two_defects",
                "altitude_mean_meters",
            ],
        )
    )

    splom.save(output_dir + "pairwise_plots.png", webdriver=driver)
    return print("PairWise Scatter Plotted")


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
    plot_pairwise(dataframe, "total_cup_points", output_dir)
    plot_correlation_matrix(dataframe, output_dir)


if __name__ == "__main__":
    plot_visualisations(opt["--input_data"], opt["--out_dir"])
