# Stock Market Sentiment Analysis

This project predicts **stock market sentiment** (Positive / Negative) from financial news headlines using **Natural Language Processing (NLP)** and **Machine Learning**.  
It trains multiple models (Logistic Regression, Naive Bayes, SVM) on **TF-IDF** and **Word2Vec** features, evaluates them, and deploys the best model using **Flask**.

---

## 🚀 Features
- Text cleaning (remove HTML, URLs, punctuation, stopwords)
- Tokenization using **NLTK**
- Feature extraction with **TF-IDF** and **Word2Vec**
- Model comparison and evaluation (F1-score, precision, recall, ROC AUC)
- Flask-ready deployment with saved model and vectorizer

---

## 📦 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/stock-market-sentiment-analysis.git
cd stock-market-sentiment-analysis
