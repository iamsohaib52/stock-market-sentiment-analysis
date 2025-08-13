from flask import Flask, render_template, request
import pickle
import os

MODEL_PATH = "model_tfidf.pkl"
VECTORIZER_PATH = "vectorizer_tfidf.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("text")
        if input_text:
            transformed_text = vectorizer.transform([input_text])
            pred = model.predict(transformed_text)[0]
            if pred in ["Positive", "Negative"]:
                prediction = pred
            else:
                prediction = "Unknown"  # This should never happen unless model is broken

    return render_template("index.html", prediction=prediction, input_text=input_text)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)