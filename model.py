import pickle
import pandas as pd
import re

# from nltk.corpus import stopwords
# from nltk import WordNetLemmatizer

# -----------------------
# Load Pickle Files
# -----------------------
with open('pickles/sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

with open('pickles/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('pickles/train_user_item_matrix_filled.pkl', 'rb') as f:
    train_user_item_matrix_filled = pickle.load(f)

with open('pickles/item_similarity_df.pkl', 'rb') as f:
    item_similarity_df = pickle.load(f)

with open('pickles/product_id_name_map.pkl', 'rb') as f:
    product_id_name_map = pickle.load(f)

# -----------------------
# Text Cleaning Function
# -----------------------
def clean_review_text(text):
    # stop_words = set(stopwords.words('english'))
    # lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    # words = [word for word in words if word not in stop_words]
    # words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# -----------------------
# Recommendation Function
# -----------------------
def recommend_item_based(username, n_recommendations=20):
    username = str(username).strip()

    if username not in train_user_item_matrix_filled.index:
        return pd.DataFrame(columns=['product_id', 'product_name', 'score'])

    user_ratings = train_user_item_matrix_filled.loc[username]
    rated_products = user_ratings[user_ratings > 0].index

    if len(rated_products) == 0:
        return pd.DataFrame(columns=['product_id', 'product_name', 'score'])

    similarity_scores = item_similarity_df[rated_products]
    weighted_scores = similarity_scores.dot(user_ratings[rated_products])

    recommendations = weighted_scores.drop(index=rated_products)

    top_recommendations = recommendations.sort_values(
        ascending=False
    ).head(n_recommendations)

    result = pd.DataFrame({
        'product_id': top_recommendations.index,
        'score': top_recommendations.values
    })

    result['product_name'] = result['product_id'].map(product_id_name_map)

    return result[['product_id', 'product_name', 'score']]

# -----------------------
# Sentiment Filtering
# -----------------------
def sentiment_based_top5(recommendations_df, reviews_df):
    sentiment_scores = []

    for _, row in recommendations_df.iterrows():
        product_id = row['product_id']
        product_name = row['product_name']

        product_reviews = reviews_df[reviews_df['id'] == product_id]['reviews_text']
        if len(product_reviews) == 0:
            continue

        cleaned_reviews = product_reviews.apply(clean_review_text)
        vectors = tfidf_vectorizer.transform(cleaned_reviews)
        probs = sentiment_model.predict_proba(vectors)[:, 1]

        sentiment_scores.append({
            'product_id': product_id,
            'product_name': product_name,
            'sentiment_score': probs.mean()
        })

    sentiment_df = pd.DataFrame(sentiment_scores)

    return sentiment_df.sort_values(
        by='sentiment_score',
        ascending=False
    ).head(5)
