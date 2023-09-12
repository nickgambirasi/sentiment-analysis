# system libraries
import os
import json

# data processing libraries
import pandas as pd

# data plotting libraries
import plotly.express as px
import plotly.graph_objects as go
import plotly

# huggingface libraries
import transformers
import datasets
import torch

# huggingface objects
from datasets import Dataset, ClassLabel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# pytorch objects
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_DIR = os.path.join(ROOT_DIR, "output")
DATA_DIR = os.path.join(ROOT_DIR, "data")


def collect_and_predict() -> pd.DataFrame:
    """
    Utility function that collects all production data
    and makes sentiment predictions on each data point
    using the previously trained model

    Paramters
    ----------
    `None`: Works directly on production data that has
    been predefined in the data processing phase

    Outputs
    ----------
     - `prod_df`: a dataframe of the production data,
        complete with model sentiment predictions
    """
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    def processing_function(examples):
        """
        Simple processing function used to tokenize
        the production data for prediction
        """
        return tokenizer(examples["text"], truncation=True)

    # define device to whcih we send models and data
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # read the production data into a pandas dataframe
    prod_df = pd.read_parquet(os.path.join(DATA_DIR, "prod.parquet"))
    prod_df = prod_df.reset_index()
    prod_df = prod_df.drop(columns="index")

    # instantiate the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)  # puts model on torch device

    # create dataset from dataframe and tokenize the dataset
    prod_ds = Dataset.from_pandas(prod_df)
    tokenized_prod_ds = prod_ds.map(
        processing_function, remove_columns=prod_ds.column_names, batched=True
    )

    # construct data collator and data loader
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset=tokenized_prod_ds, shuffle=False, batch_size=16, collate_fn=collator
    )

    # construct class label
    class_label = ClassLabel(names_file=os.path.join(MODEL_DIR, "labels.txt"))

    # make predictions on the production data
    predictions = []
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.append(class_label.int2str(preds))

    # flatten prediction list into 1-d list
    predictions = [pred for pred_list in predictions for pred in pred_list]

    # add predictions to the pandas dataframe
    prod_df["sentiment"] = pd.Series(predictions)

    return prod_df


def plot_data(data_df):
    """
    Utility function that utilizes plotly.js
    to render plots of the data for the selected
    airline, returns an html object that can be
    rendered on the analysis webpage for the
    selected airline (i.e. `/analysis/<airline_name>`)

    Parameters
    ----------
    - `data_df`: Pandas dataframe that should contain
        production data for a specific airline

    Outputs
    ----------
    - `plot_json`: An JSON object created through
        plotly that can be rendered in an HTML template
    """

    # quick assertion to ensure that the dataframe has been
    # properly loaded, otherwise return a NoneType than can
    # be checked for
    if not all(pd.Series(data_df["sentiment"])):
        return None

    # get counts of the sentiment values and create pie chart
    # for it
    airline_sentiment_counts = (
        pd.DataFrame(data_df["sentiment"].value_counts())
        .reset_index()
        .sort_values(by=["sentiment"])
    )

    # create figure from the sentiment counts dataframe
    fig = go.Figure(
        data=[
            go.Pie(
                labels=airline_sentiment_counts["sentiment"],
                values=airline_sentiment_counts["count"],
                sort=False,
                marker=dict(colors=["red", "blue", "green"]),
            )
        ]
    )

    # create figure widget than can be updated each time new data comes in
    # fig2 = go.FigureWidget(fig)

    # write figure to json
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
