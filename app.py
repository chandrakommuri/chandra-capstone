import logging

from flask import Flask, render_template, request
import pandas as pd
from model import recommend_item_based, sentiment_based_top5

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("Chandra's capstone app initialized")

# Load dataset ONLY for reviews
logging.info("Loading dataset: sample30.csv")
df = pd.read_csv('data/sample30.csv')
df['reviews_username'] = df['reviews_username'].astype(str).str.strip()
logging.info(f"Dataset loaded successfully. Rows: {len(df)}")
usernames = sorted(df['reviews_username'].unique().tolist())

@app.route('/', methods=['GET', 'POST'])
def index():
    logging.info(f"Received {request.method} request")
    username = None
    recommendations = None
    error = None

    try:
        if request.method == 'POST':
            username = request.form['username']
            logging.info(f"Username received: {username}")

            logging.info("Calling recommend_item_based()")
            top_20 = recommend_item_based(username, 20)
            logging.info(f"Top-20 recommendations generated. Count: {len(top_20)}")

            if top_20.empty:
                error = "User not found or no recommendations available."
                logging.warning(error)
            else:
                logging.info("Applying sentiment-based filtering")
                recommendations = sentiment_based_top5(top_20, df)
                logging.info("Top-5 sentiment-based recommendations ready")
    except Exception as e:
        logging.exception("Error occurred during request processing")
        error = "An internal error occurred. Please check logs."

    logging.log(logging.INFO, "Rendering index.html")
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
