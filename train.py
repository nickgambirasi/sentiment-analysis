"""
Model Training Workflow
-------------------------
Devloper: Nick Gambirasi
Date: 31 August 2023

Purpose
-------------------------
This script executes the process of training the machine learning
model for sentiment analysis tasks in base Pytorch. I chose to
opt for base Pytorch over other options because it allows for
further control of the tuning workflow.

This workflow collects the training, validation, and test sets
(if specified by the user in the CLI call) and prepares the
datasets for training and evaluation by going through the
following pipeline:

    1) Reads in pandas dataframes, creates datasets, and
        populates a hf dataset dictionary object

    2) Initializes the model to be trained (model is
        specified by the user when calling the CLI,
        could be `bert`, `roberta`, or `distilbert`)

    3) Initializes the data tokenizer used to prepare
        the data for the model

    4) Initializes data collator, data loader (batch size
        specified at CLI call), and optimizer (learning
        rate and optimizer type also specified at CLI
        call)

    5) Goes through the model fine-tuning and validation
        process (number of epochs for this process specified
        at CLI call)

    6) Returns the tuned model to a local directory, along
        with information about the tuning run and the results

"""
# system
import sys
import os
import shutil
import time

# cli
from argparse import ArgumentParser

# huggingface
import transformers, datasets, evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)
from datasets import Dataset, DatasetDict, ClassLabel
from evaluate import load

# pytorch
import torch
from torch.utils.data import DataLoader

# tuning progress
from tqdm.auto import tqdm

# data processing
import pandas as pd

# internal libraries
from utils import continue_loop


# define the tokenizer preprocess function
def preprocess_function(examples):
    examples["labels"] = class_labels.str2int(examples["label"])
    return tokenizer(examples["text"], truncation=True)


if __name__ == "__main__":
    # change logging values
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    evaluate.logging.set_verbosity_error()

    # create command line parser and add its arguments
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        action="store",
        dest="model",
        required=True,
        type=str.lower,
        choices=["bert", "distilbert", "roberta"],
        help="name of base model to be fine-tuned for the task",
    )

    parser.add_argument(
        "--batch",
        action="store",
        dest="batch",
        type=int,
        default=16,
        help="number of observations to collate into one batch",
    )

    parser.add_argument(
        "--optim",
        action="store",
        dest="optim",
        required=True,
        type=str.lower,
        choices=["adamw", "sgd", "rmsprop"],
        help="optimizer used to fine-tune the model",
    )

    parser.add_argument(
        "--lr",
        action="store",
        dest="lr",
        type=float,
        default=1.0e-4,
        help="initial learning rate to use for the optimizer",
    )

    parser.add_argument(
        "--epochs",
        action="store",
        dest="epochs",
        type=int,
        default=10,
        help="number of epochs to tune the model for",
    )

    parser.add_argument(
        "--run-test",
        action="store_true",
        dest="test",
        default=False,
        help="flag that indicates whether or not test should be run on test set",
    )

    # get the args from the CLI call and assign them to variables
    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batch
    optim = args.optim
    learning_rate = args.lr
    num_epochs = args.epochs
    run_test = args.test

    # determine which device will be used for training
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\nTuning on device: {device}\n")

    # root directory
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

    # create output directory
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    try:
        os.mkdir(OUTPUT_DIR)
    except Exception as e:
        flag = continue_loop(
            message="You are about to delete an existing output directory. Are you sure you want to continue?"
        )
        if not flag:
            sys.exit("Cannot complete tuning when output directory already exists")
        shutil.rmtree(OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    DATA_DIR = os.path.join(ROOT_DIR, "data")

    # construct classlabels from labels file
    try:
        class_labels = ClassLabel(names_file=os.path.join(DATA_DIR, "labels.txt"))
        num_labels = len(class_labels.names)
    except Exception as e:
        sys.exit(
            "There was an error creating the class labels from the dataset labels file"
        )

    # collect the datasets from the data directory
    try:
        train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(DATA_DIR, "valid.parquet"))
    except Exception as e:
        sys.exit("There was an error opening training and validation dataset files.")

    # construct pytorch datasets from the dataframes
    try:
        train_data = Dataset.from_pandas(train_df)
        val_data = Dataset.from_pandas(val_df)
    except Exception as e:
        sys.exit("There was an error creating datasets from the dataframes")

    # Construct dataset dictionary with the training and validation datasets
    datasets = DatasetDict({"train": train_data, "validation": val_data})

    # initialize the model to be tuned
    try:
        match model_name:
            case "bert":
                model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels=num_labels
                )
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            case "roberta":
                model = AutoModelForSequenceClassification.from_pretrained(
                    "roberta-base", num_labels=num_labels
                )
                tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            case "distilbert":
                model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=num_labels
                )
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            case _:
                pass
        # cast the model to whatever device is being used
        model.to(device)
    except Exception as e:
        sys.exit("There was an error creating the requested model and tokenizer")

    # map the datasets to tokenized datasets
    try:
        tokenized_datasets = datasets.map(
            preprocess_function,
            remove_columns=datasets["train"].column_names,
            batched=True,
        )
    except Exception as e:
        sys.exit("There was an error tokenizing the datasets")

    # construct data collator
    try:
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
    except Exception as e:
        sys.exit("There was an error constructing the data collator")

    # construct the data loader
    try:
        train_data_loader = DataLoader(
            dataset=tokenized_datasets["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collator,
        )
        val_data_loader = DataLoader(
            dataset=tokenized_datasets["validation"],
            batch_size=batch_size,
            collate_fn=collator,
        )
    except Exception as e:
        sys.exit("There was an error creating the training and validation dataloaders")

    # construct optimizer
    try:
        match optim:
            case "adamw":
                optimizer = torch.optim.AdamW(
                    params=model.parameters(), lr=learning_rate
                )
            case "sgd":
                optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
            case "rmsprop":
                optimizer = torch.optim.RMSprop(
                    params=model.parameters(), lr=learning_rate
                )
            case _:
                pass
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs * len(train_data_loader),
        )
    except Exception as e:
        sys.exit(
            "There was an error constructing the optimizer and learning rate scheduler"
        )

    # load metrics
    metric = load("accuracy")

    # tune and validate the mode for specified number of epochs
    best_model = None
    best_model_score = -1

    print("\nStarting tuning...")
    time.sleep(1)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}\n")
        for batch in tqdm(train_data_loader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        for batch in tqdm(val_data_loader, desc="Performing validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        score = metric.compute().get("accuracy")
        print(f"Validation score: {score}")

        # determine best model by metric
        if epoch == 0:
            best_model = model
            best_model_score = score
        else:
            if score > best_model_score:
                best_model = model
                best_model_score = score

    print("Saving model...\n")
    try:
        model.save_pretrained(OUTPUT_DIR, from_pt=True)
        tokenizer.save_pretrained(OUTPUT_DIR, from_pt=True)
        shutil.copy2(os.path.join(DATA_DIR, "labels.txt"), OUTPUT_DIR)
    except Exception as e:
        sys.exit("There was an error saving the tuned model")

    # test workflow
    if run_test:
        print("Starting model testing...\n")

        # load test dataframe
        try:
            test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
        except Exception as e:
            sys.exit("Could not load test data into dataframe")

        # create test dataset
        try:
            test_data = Dataset.from_pandas(test_df)
        except Exception as e:
            sys.exit("Test dataset could not be created from dataframe")

        # tokenize test dataset
        try:
            tokenized_test_data = test_data.map(
                preprocess_function, remove_columns=test_data.column_names, batched=True
            )
        except Exception as e:
            sys.exit("There was an error tokenizing the test data")

        # create test data loader
        try:
            test_data_loader = DataLoader(
                dataset=tokenized_test_data, batch_size=batch_size, collate_fn=collator
            )
        except Exception as e:
            sys.exit("There was an error creating the data loader for the test data")

        for batch in tqdm(test_data_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        test_score = metric.compute().get("accuracy")

        print(f"Model achieved test score of {test_score}\n")

    sys.exit("Training process completed successfully")
