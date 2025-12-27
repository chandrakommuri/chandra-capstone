import logging

from flask import Flask, render_template, request
import pandas as pd
from model import recommend_item_based, sentiment_based_top5

app = Flask(__name__)

# Load dataset ONLY for reviews
df = pd.read_csv('data/sample30.csv')
df['reviews_username'] = df['reviews_username'].astype(str).str.strip()
usernames = sorted(df['reviews_username'].unique().tolist())

@app.route('/', methods=['GET', 'POST'])
def index():
    username = None
    recommendations = None
    error = None

    if request.method == 'POST':
        username = request.form['username']
        top_20 = recommend_item_based(username, 20)

        if top_20.empty:
            error = "User not found or no recommendations available."
        else:
            recommendations = sentiment_based_top5(top_20, df)

    return render_template(
        'index.html',
        username=username,
        recommendations=recommendations,
        usernames=usernames,
        error=error
    )

if __name__ == '__main__':
    logging.log(logging.INFO, 'Starting app')
    app.run()
