# Stock Market Sentiment Analysis

This project predicts **stock market sentiment** (Positive / Negative) from financial news headlines using **Natural Language Processing (NLP)** and **Machine Learning**.  
It trains multiple models (Logistic Regression, Naive Bayes, SVM) on **TF-IDF** and **Word2Vec** features, evaluates them, and deploys the best model using **Flask**.

---

## 🚀 Features
- Robust text cleaning pipeline: removes HTML tags, URLs, punctuation, and stopwords  
- Tokenization with **NLTK** for effective preprocessing  
- Feature extraction using both **TF-IDF** (with unigrams and bigrams) and **Word2Vec** embeddings  
- Multiple model training and evaluation with metrics including accuracy, precision, recall, weighted F1-score, confusion matrix, and ROC AUC (where applicable)  
- Comparison of classical models: Logistic Regression, Multinomial Naive Bayes, and Linear SVM  
- Automated selection of the best performing model for deployment  
- Flask web application for real-time sentiment prediction via an intuitive UI  

---

## 📦 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/iamsohaib52/stock-market-sentiment-analysis.git
cd stock-market-sentiment-analysis
```

### 2. Create and activate a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Convert Notebook to Python Script (if needed)

The repository currently includes the training code as a Jupyter notebook (`train.ipynb`).  
If you prefer to run training from a Python script instead of the notebook, you can convert it using the following command:

```bash
jupyter nbconvert --to script train.ipynb
```

This will generate a `train.py` file which you can then run directly:

```bash
python train.py
```

Otherwise, you can simply open and run the notebook cells in Jupyter or any compatible environment.

## ▶️ Running the Project

### Train models and generate deployment files:

```bash
python train.py
```

This will preprocess data, train multiple models, evaluate them, and save the best TF-IDF model and vectorizer in the `webapp/` folder.

### Run the Flask web application:

```bash
cd webapp
python app.py
```

Open your browser and visit:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📂 Project Structure

```
stock-market-sentiment-analysis/
│
├── stock_data.csv          # Input dataset (financial news headlines + sentiment labels)
├── train.py                # Training and evaluation script (from .ipynb converted to .py)
├── requirements.txt        # Python dependencies
├── webapp/                 
│   ├── app.py              # Flask web app for deployment
│   ├── model_tfidf.pkl     # Saved TF-IDF sentiment classification model
│   ├── vectorizer_tfidf.pkl# Saved TF-IDF vectorizer
│   └── ...                 # Additional saved artifacts
└── README.md               # Project documentation
```

---

## 📊 Model Evaluation Summary (Example)

| Model                   | Feature Type | Weighted F1-score |
| ----------------------- | ------------ | ----------------- |
| Logistic Regression     | TF-IDF       | 0.76              |
| Multinomial Naive Bayes | TF-IDF       | 0.75              |
| Linear SVM              | TF-IDF       | 0.78              |
| Logistic Regression     | Word2Vec     | 0.67              |
| Linear SVM              | Word2Vec     | 0.70              |

*TF-IDF based models generally outperform Word2Vec embedding-based models in this task.*

---

## 🧰 Technologies & Libraries Used

* Python 3.x
* pandas, numpy
* nltk (tokenization, stopwords)
* scikit-learn (vectorizers, classifiers, evaluation)
* gensim (Word2Vec embeddings)
* beautifulsoup4 (HTML cleaning)
* Flask (web deployment)

---

## 👤 Author

Muhammad Sohaib
BS - Computer Science  
Semester Project | Course: Artificial Intelligence
COMSATS University Islamabad, Vehari Campus
Email: [sp22-bcs-057@cuivehari.edu.pk](mailto:sp22-bcs-057@cuivehari.edu.pk)

