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

from app_utils import collect_and_predict

app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template("index.html")


@app.route("/analysis/loading")
def redirect():
    return render_template("loading.html")


@app.route("/analysis/results")
def results():
    """
    Prior to loading the comparison page, completes
    all calculations and predictions associated with
    the production data.

    Should return a pandas dataframe as the context
    for the website, so that it can be operated on
    from the results window.
    """
    pred_data = collect_and_predict()
    return render_template(
        "results.html", data=pred_data.to_html(classes="table table-stripped")
    )


if __name__ == "__main__":
    app.run(debug=True)
