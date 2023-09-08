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
from app_utils import collect_and_predict

APP_ROOT = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

global pred_data


@app.route("/")
def redirect():
    return render_template("loading.html")


@app.route("/select-airline")
def select_airline():
    global pred_data
    pred_data = collect_and_predict()
    airlines = []
    with open(os.path.join(APP_ROOT, os.pardir, "data/airlines.txt")) as f:
        for line in f.readlines():
            airlines.append(line.strip().replace(" ", "_"))
        print(airlines)
    context = {"airlines": airlines}
    return render_template("homepage.html", **context)


@app.route("/analysis/<airline_name>")
def results(airline_name: str):
    """
    Queries the production data for the first airline selected
    from the initial webpage.
    """
    # collects all prediction data
    airline_name = airline_name.replace("_", " ")
    airline_data = pred_data.query("airline==@airline_name")
    return render_template(
        "results.html", data=airline_data.to_html(classes="table table-stripped")
    )


if __name__ == "__main__":
    app.run(debug=True)
