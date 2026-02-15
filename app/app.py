"""
Flask Web Application for Natural Language Inference (NLI) Prediction.

This app uses a custom-trained Sentence-BERT model to predict the relationship
between a premise and hypothesis sentence (entailment, neutral, or contradiction).

Author: Dechathon Niamsa-ard [st126235]
"""

from flask import Flask, render_template, request, jsonify
from utils import load_model, predict_nli


# ==================== Initialise ====================

app = Flask(__name__)

# Load model once at startup
bert_model, classifier_head, word2id, device = load_model(model_dir='../model')
print("Model loaded successfully!")


# ==================== Routes ====================

@app.route("/")
def index():
    """Serve the main NLI prediction page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict NLI relationship between premise and hypothesis.

    Expects JSON with 'premise' and 'hypothesis' fields.
    Returns JSON with 'label', 'confidence', 'similarity', and 'color'.
    """
    data = request.get_json()
    premise = data.get("premise", "").strip()
    hypothesis = data.get("hypothesis", "").strip()

    if not premise or not hypothesis:
        return jsonify({"error": "Both premise and hypothesis are required"}), 400

    result = predict_nli(premise, hypothesis, bert_model, classifier_head, word2id, device)
    return jsonify(result)


# ==================== Entry Point ====================

if __name__ == "__main__":
    print("Starting NLI Web Application on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
