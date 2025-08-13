from flask import Flask, render_template, request
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_tfidf.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer_tfidf.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("text")
        if input_text:
            transformed_text = vectorizer.transform([input_text])
            pred = model.predict(transformed_text)[0]
            prediction = pred if pred in ["Positive", "Negative"] else "Unknown"

    return render_template("index.html", prediction=prediction, input_text=input_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
