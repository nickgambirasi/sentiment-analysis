"""
Data Processing
---------------
Developer: Nick Gambirasi
Date: 31 August 2023

Purpose
---------------
This script performs processing on a twitter airline sentiment dataset, available at
[this link](https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment).

This script only performs processing, and does not include exploratory data analysis.

Returns
---------------
As the dataset is provided in one file, this script will return three dataset file, one for a training
dataset, one for a validation dataset, and one for a test dataset. It will also return files that
include the distinct labels detected in the dataset, and also a file that includes the airlines
represented in the dataset.

Note that the test dataset will include equal amounts of tweets for each airline, with the goal of
comparing airline performances to one another is a web-based dashboard.
"""
# system libraries
import sys
import os
import shutil

# data handling libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# huggingface libraries
from datasets import load_dataset, Dataset

if __name__ == "__main__":
    # define the link to the dataset on huggingface
    DATASET_HUB_LINK = "osanseviero/twitter-airline-sentiment"

    # constant that defines the columns of the dataset that
    # will be used for model training
    COLUMNS_TO_KEEP_FOR_TRAINING = ["tweet_id", "airline_sentiment", "airline", "text"]

    # discover path to root directory
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

    # build the path to the dataset
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    # build the folder for the dataset
    try:
        os.mkdir(DATA_DIR)

    except Exception as e:
        print(f"Cleaning out already existing data folder at {DATA_DIR}")
        shutil.rmtree(DATA_DIR)
        os.mkdir(DATA_DIR)

    # define the link to the dataset on huggingface
    DATASET_HUB_LINK = "osanseviero/twitter-airline-sentiment"

    # constant that defines the columns of the dataset that
    # will be used for model training
    COLUMNS_TO_KEEP_FOR_TRAINING = ["tweet_id", "airline_sentiment", "airline", "text"]
    DATA_TO_KEEP_FOR_PRODUCTION = ["tweet_id", "airline", "text"]

    # load data from huggingface space
    data = load_dataset(DATASET_HUB_LINK)

    # print information about the dataset columns
    print(f"The dataset has the following columns: {data['train'].column_names}\n")
    print(
        f"For the purposes of model training, only the following columns will be kept:{COLUMNS_TO_KEEP_FOR_TRAINING}\nAll other columns will be removed.\n"
    )

    # remove columns unnecessary for model training
    try:
        data = data.remove_columns(
            [
                column
                for column in data["train"].column_names
                if column not in COLUMNS_TO_KEEP_FOR_TRAINING
            ]
        )

    except Exception as e:
        sys.exit("There was an error removing columns from the dataset")

    print("Successfully removed unecessary columns from the dataset.\n")

    # create file for label names
    print(f"Writing unique sentiments values to {DATA_DIR}/labels.txt...\n")

    sentiments = pd.Series(data["train"]["airline_sentiment"]).unique().tolist()
    try:
        with open(os.path.join(DATA_DIR, "labels.txt"), "w") as f:
            f.writelines(sentiment + "\n" for sentiment in sentiments)

    except Exception as e:
        sys.exit(
            "There was an error writing the sentiment labels to the specified file"
        )

    print("Successfully wrote labels to the specified file")

    # create file for unique airline names
    print(f"Writing unique sentiments values to {DATA_DIR}/airlines.txt...\n")

    airlines = pd.Series(data["train"]["airline"]).unique().tolist()

    try:
        with open(os.path.join(DATA_DIR, "airlines.txt"), "w") as f:
            f.writelines(airline + "\n" for airline in airlines)

    except Exception as e:
        sys.exit("There was an error writing the airline names to the specified file")

    print("Successfully wrote airline names to the specified file")

    # change data column names to match huggingface model tuning requirements
    data = data.rename_columns({"tweet_id": "idx", "airline_sentiment": "label"})

    # create the dataset files

    data_df = pd.DataFrame(data["train"])

    # production dataset: contains 200 observations from every dataset to use for prod comparison
    print("Creating production dataset...\n")
    prod_df = pd.DataFrame(columns=data["train"].column_names)

    for airline in airlines:
        airline_subset = pd.DataFrame(
            data["train"]
            .filter(lambda x: x["airline"] == airline)
            .shuffle()
            .select(range(200))
        )
        prod_df = pd.concat([prod_df, airline_subset], join="inner")
    prod_df.drop(columns=["label"])
    prod_df.to_parquet(os.path.join(DATA_DIR, "prod.parquet"))

    # training, validation, and test datasets: 70% train, 20% val, 10% test
    print("Creating training, validation, and test sets...\n")
    data_df = (
        pd.merge(data_df, prod_df, indicator=True, how="outer")
        .query('_merge=="left_only"')
        .drop("_merge", axis=1)
    )
    data_df.drop(columns=["airline"], inplace=True)

    train_df, val_df = train_test_split(data_df, test_size=0.3)
    val_df, test_df = train_test_split(val_df, test_size=0.33)

    train_df.to_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df.to_parquet(os.path.join(DATA_DIR, "valid.parquet"))
    test_df.to_parquet(os.path.join(DATA_DIR, "test.parquet"))

    sys.exit("Data processing complete.")
