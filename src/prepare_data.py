# author: Arlin Cherian, Kristin Bunyan, Michelle Wang, Berkay Bulut
# date: 2021-11-24

"""Cleans, preps, and feature engineers the data for the Coffee Quality Database dataset (https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-07-07/readme.md).
Writes the training and test data to separate csv files.

Usage: src/pre_process_coffee.py --input_data=<input_data> --out_dir=<out_dir>
  
Options:
--input_data=<input_data>  Path (including filename) to downloaded data retrieved from download_data.py script 
--out_dir=<out_dir>   Path to directory where the processed data should be written
" -> doc
"""
import string
import numpy as np
from sklearn.model_selection import train_test_split

from docopt import docopt
import os
import pandas as pd

opt = docopt(__doc__)

def main(input_data, out_dir):
    df = pd.read_csv(input_data, header=1)
    
    #drops the columns we are not using
    df = df.drop(columns=['owner',
                          'farm_name',
                          'lot_number',
                          'mill',
                          'ico_number',
                          'company',
                          'altitude',
                          'producer',
                          'number_of_bags',
                          'bag_weight',
                          'in_country_partner',
                          'certification_body',
                          'certification_address',
                          'certification_contact',
                          'unit_of_measurement',
                          'altitude_low_meters',
                          'altitude_high_meters',
                          'owner_1',
                          'aroma',
                          'clean_cup',
                          'sweetness',
                          'cupper_points',
                          'flavor',
                          'aftertaste',
                          'body',
                          'balance',
                          'uniformity',
                          'acidity',
                          'region',
                          'grading_date',
                          'expiration'])
        
    #filter for Arabica
    df = df.query('species == "Arabica"')
    df = df.drop(columns=['species'])

    #drop the null values
    df = df.dropna()
    
    #function to clean region
    def coffee_region(text):
        """
        Returns the official coffee region of the associated country.

        Parameters:
        ------
        text: (str)
        the input text

        Returns:
        -------
        Official Coffee region: (str)
        """
        region_1 = "East Africa and the Arabian Peninsula"
        region_2 = "Southeast Asia and the Pacific"
        region_3 = "Latin America"

        country_list3 = ("Mexico",
                         "Guatemala",
                         "Colombia",
                         "Brazil",
                         "Honduras",
                         "Costa Rica",
                         "El Salvador",
                         "Nicaragua",
                         "Haiti",
                         "Panama",
                        "United States (Puerto Rico)",
                        "Peru")
        country_list2 = ("Taiwan",
                         "Indonesia",
                         "China",
                         "Thailand",
                         "Myanmar",
                         "Vietnam",
                         "Philippines",
                         "Laos",
                         "United States")
        country_list1 = ("Tanzania, United Republic Of",
                         "Uganda",
                         "Kenya",
                         "Malawi",
                         "Ethiopia",
                         "Laos",
                         "Rwanda") 

        for country in country_list1:
            if text == country:
                return region_1

        for country in country_list2:
            if text == country:
                return region_2    

        for country in country_list3:
            if text == country:
                return region_3
    
    # creates coffee region column
    df = df.assign(region=df["country_of_origin"].apply(coffee_region))
    
    #function to format the file path name
    def file_path_name(file_path, data_frame):
        """
        Returns the file path name.

        Parameters:
        ------
        file_path: (str)
        the name of the file path

        data_frame: (str)
        the name of the dataframe

        Returns:
        -------
        The fill filepath name: (str)
        """

        texts = file_path + data_frame + ".csv"
        texts.replace("//", "/")
        return texts
    
    #creates the cleaned dataset
    try: 
        df.to_csv(file_path_name(out_dir, "df"), index=False)
    except:
        os.makedirs(os.path.dirname(file_path_name(out_dir, "df")))
        df.to_csv(file_path_name(out_dir, "df"), index = False)
    
    #splits the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    
    #creates the train and test csv files
    try: 
        train_df.to_csv(file_path_name(out_dir, "train_df"), index=False)
    except:
        os.makedirs(os.path.dirname(file_path_name(out_dir, "train_df")))
        train_df.to_csv(file_path_name(out_dir, "train_df"), index = False)
    
    try: 
        test_df.to_csv(file_path_name(out_dir, "test_df"), index=False)
    except:
        os.makedirs(os.path.dirname(file_path_name(out_dir, "test_df")))
        test_df.to_csv(file_path_name(out_dir, "test_df"), index = False)
    

if __name__ == "__main__":
    main(opt["--input_data"], opt["--out_dir"])
  