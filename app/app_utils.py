import os

import pandas as pd

import transformers
import datasets
import torch

from datasets import Dataset, ClassLabel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedModel,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader

from typing import Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_DIR = os.path.join(ROOT_DIR, "output")
DATA_DIR = os.path.join(ROOT_DIR, "data")


def collect_and_predict() -> pd.DataFrame:
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
        dataset=tokenized_prod_ds, batch_size=16, collate_fn=collator
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
