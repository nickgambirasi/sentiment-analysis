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

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')