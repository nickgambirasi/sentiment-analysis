"""
Flask Application for Sentiment Comparison
--------------------------------------------
Developer: Nick Gambirasi
Date: 02 September 2023

Purpose
--------------------------------------------
This script is for the flask app. The purpose of the flask
app is to employ the model on a "production" dataset to
evaluate which airlines have the  best customer satisfaction
rates.

The application begins with the screen that is contains a
single button. When the user clicks this button, they will
be redirected to the main application page. In the meantime,
the program will read in the production dataset and make all
of the predictions on the production data. After the predictions
are made, then the user will be redirected again to a page that
shows the comparisons of the airlines in some graphics.
"""

from flask import Flask, render_template

import os
from app_utils import collect_and_predict, plot_data

APP_ROOT = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

global pred_data
global airlines

pred_data = None
airlines = []


@app.route("/")
def redirect():
    return render_template("loading.html")


@app.route("/select-airline")
def select_airline():
    global pred_data
    global airlines

    if pred_data is None:
        pred_data = collect_and_predict()
    if not airlines:
        with open(os.path.join(APP_ROOT, os.pardir, "data/airlines.txt")) as f:
            for line in f.readlines():
                airlines.append(line.strip().replace(" ", "_"))
    context = {"airlines": airlines}
    return render_template("homepage.html", **context)


@app.route("/analysis/<airline_name>")
def results(airline_name: str):
    """
    Queries the production data for the first airline selected
    from the initial webpage.
    """
    global pred_data
    global airlines

    airlines_copy = airlines.copy()
    try:
        airlines_copy.remove(airline_name)
    except ValueError as e:
        pass
    airline_data = pred_data.query("airline==@airline_name.replace('_', ' ')")
    context = {
        "graphJSON": plot_data(data_df=airline_data),
        "airline": airline_name,
        "unselected_airlines": airlines_copy,
    }
    return render_template("results.html", **context)


@app.route("/analysis/compare/<airline1_name>+<airline2_name>")
def compare(airline1_name: str, airline2_name: str):
    global pred_data

    # airline1_name = airline1_name.replace("_", " ")
    # airline2_name = airline2_name.replace("_", " ")

    airline1_data = pred_data.query("airline==@airline1_name.replace('_', ' ')")
    airline2_data = pred_data.query("airline==@airline2_name.replace('_', ' ')")

    context = {
        "airline1_name": airline1_name,
        "airline2_name": airline2_name,
        "airline1_graphJSON": plot_data(data_df=airline1_data),
        "airline2_graphJSON": plot_data(data_df=airline2_data),
    }

    return render_template("compare.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
