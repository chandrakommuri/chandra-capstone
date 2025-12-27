import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify

from model import recommend_item_based, sentiment_based_top5

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("Chandra's capstone app initialized")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
logging.info("Loading dataset: sample30.csv")
df = pd.read_csv("data/sample30.csv")
df["reviews_username"] = df["reviews_username"].astype(str).str.strip()

usernames = sorted(df["reviews_username"].unique().tolist())
logging.info(f"Dataset loaded successfully. Rows: {len(df)}")

# --------------------------------------------------
# UI Route
# --------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        usernames=usernames
    )

# --------------------------------------------------
# AJAX API Route
# --------------------------------------------------
@app.route("/recommend", methods=["POST"])
def recommend_api():
    try:
        data = request.get_json()
        username = data.get("username")

        logging.info(f"AJAX request received for user: {username}")

        top_20 = recommend_item_based(username, 20)

        if top_20.empty:
            return jsonify({"error": "User not found or no recommendations available"}), 400

        top_5 = sentiment_based_top5(top_20, df)

        results = []
        for _, row in top_5.iterrows():
            results.append({
                "product_name": row["product_name"],
                "sentiment_score": round(row["sentiment_score"], 2)
            })

        return jsonify({
            "username": username,
            "recommendations": results
        })

    except Exception as e:
        logging.exception("Error during recommendation")
        return jsonify({"error": "Internal server error"}), 500

# Uncomment below lines for local testing
if __name__ == '__main__':
    logging.log(logging.INFO, 'Starting app')
    app.run()
