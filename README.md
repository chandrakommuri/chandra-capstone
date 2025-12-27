# Sentiment-Based Product Recommendation System  
**Capstone Project – UpGrad and IIITB Machine Learning & AI**  
**Author:** Chandra Sekhar Kommuri  

![Capstone](https://img.shields.io/badge/Capstone-Machine%20Learning-orange)
![Deployed](https://img.shields.io/badge/Deployed-Heroku-blueviolet)
![Python](https://img.shields.io/badge/Python-3.9+-blue)

---

## 🧠 Project Overview

This project implements an **end-to-end sentiment-based product recommendation system** for an e-commerce platform named **Ebuss**.

The system:
1. Recommends products using **Collaborative Filtering (ratings-based)**
2. Refines recommendations using **Sentiment Analysis** on user reviews

The final output is a **Top-5 sentiment-enhanced product recommendation list** for a given user.

---

## 🚀 Live Demo

🔗 **Heroku App:** https://chandra-capstone-b3a6e671479f.herokuapp.com/  

> ⚠️ Note: The app is hosted on a **free Heroku dyno**.  
> Initial requests may be slower due to cold start and on-demand model loading.

---

## 🏗️ Architecture Overview

```
Flask App (app.py)
│
├── AJAX Endpoint (/recommend)
├── UI Route (/)
│
├── Recommendation System
│   └── Item-Based Collaborative Filtering
│
├── Sentiment Analysis
│   └── TF-IDF + Logistic Regression
│
└── Deployment
    └── Flask + Gunicorn + Heroku
```

---

## ✨ Key Features

### 🔍 Recommendation Logic
- Item-based collaborative filtering
- Sentiment-based refinement using review text
- Final Top-5 products selected by average positive sentiment

### 🎨 UI & UX
- Bootstrap-based responsive UI
- AJAX-based recommendations (no page reload)
- Username autocomplete dropdown
- Loading indicator during inference
- Informational note about Heroku latency

---

## 📦 Dataset

- **File:** `sample30.csv`
- **Size:** ~30,000 reviews
- **Users:** 20,000+
- **Products:** 200+
- **Key Columns:**
  - `reviews_username`
  - `reviews_rating`
  - `reviews_text`
  - `id` (product id)

---

## 🛠️ Tech Stack

- **Python**
- **Flask**
- **Scikit-learn**
- **NLTK**
- **Pandas / NumPy**
- **Bootstrap 5**
- **Gunicorn**
- **Heroku**

---

## 🧪 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/chandrakommuri/chandra-capstone.git
cd chandra-capstone
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
python app.py local
```

Visit: `http://127.0.0.1:5000`

---

## 🚀 Deployment (Heroku)

### Procfile
```
web: gunicorn app:app
```

### requirements.txt
Ensure `gunicorn` is included.

### Deployment Steps
1. Push code to GitHub
2. Create Heroku app
3. Connect GitHub repo
4. Deploy `main` branch
5. Enable web dyno

---

## 📂 Important Files

| File | Description |
|-----|-------------|
| `app.py` | Flask app & AJAX API |
| `model.py` | Recommendation & sentiment logic |
| `templates/index.html` | UI |
| `requirements.txt` | Dependencies |
| `Procfile` | Heroku process |
| `sample30.csv` | Dataset |

---

## 📈 Evaluation Highlights

- Multiple ML models evaluated for sentiment analysis
- Class imbalance handled during training
- Recommendation systems evaluated using Hit Rate@20
- Final system selected based on performance and business relevance
- Deployed end-to-end with UI

---

## ⚠️ Limitations

- Cold start latency on free Heroku dyno
- No caching layer implemented
- Designed for fixed users/products in dataset

---

## 🙌 Acknowledgements

- Dataset inspired by Kaggle product reviews
- Built as part of **UpGrad Data Science & Machine Learning Capstone**

---

## 📫 Contact

**Chandra Sekhar Kommuri**  
GitHub: https://github.com/chandrakommuri
