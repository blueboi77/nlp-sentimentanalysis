"""
app.py: Flask backend server for E-Commerce Sentiment Analysis Web Application
Provides API endpoints for single review ingestion and multi-line .txt file batch processing.
"""

import os
from flask import Flask, render_template, request, jsonify, send_file, Response, redirect
import webbrowser
from model import SentimentAnalyzer

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize sentiment analyzer (loads cached joblib model)
analyzer = SentimentAnalyzer()

SAMPLE_FILE_PATH = os.path.join(os.path.dirname(__file__), "sample_reviews.txt")


@app.route("/")
def index():
    """Render the main sentiment analysis webpage."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_review():
    """
    Ingest a single customer product review and return its sentiment analysis.
    Request JSON: { "review": "..." }
    """
    data = request.get_json(silent=True) or {}
    review_text = data.get("review", "").strip()
    
    if not review_text:
        return jsonify({"error": "Review text is required"}), 400

    result = analyzer.analyze(review_text)
    return jsonify({
        "status": "success",
        "data": result
    })


@app.route("/api/analyze-file", methods=["POST"])
def analyze_file():
    """
    Ingest a .txt file containing customer reviews on separate lines.
    Analyzes each line individually and calculates the overall average sentiment.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part provided in request"}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected for upload"}), 400

    if not uploaded_file.filename.lower().endswith(".txt"):
        return jsonify({"error": "Invalid file format. Please upload a .txt file with reviews on separate lines."}), 400

    try:
        raw_bytes = uploaded_file.read()
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = raw_bytes.decode("latin-1")

        lines = content.splitlines()
        # Filter empty lines for analysis
        cleaned_lines = [line.strip() for line in lines if line.strip()]

        if not cleaned_lines:
            return jsonify({"error": "The uploaded .txt file does not contain any valid review text lines."}), 400

        batch_result = analyzer.analyze_batch(cleaned_lines)
        batch_result["filename"] = uploaded_file.filename

        return jsonify({
            "status": "success",
            "data": batch_result
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500


@app.route("/api/sample-txt", methods=["GET"])
def get_sample_txt():
    """Download or view sample .txt file for testing."""
    if os.path.exists(SAMPLE_FILE_PATH):
        return send_file(SAMPLE_FILE_PATH, mimetype="text/plain", as_attachment=False)
    return jsonify({"error": "Sample file not found"}), 404


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """
    Submit manual rating/sentiment feedback for a review to train the model live.
    Request JSON: { "review": "...", "rating": 5, "sentiment": "Positive" }
    """
    data = request.get_json(silent=True) or {}
    review_text = data.get("review", "").strip()
    rating = data.get("rating")
    sentiment = data.get("sentiment")

    if not review_text:
        return jsonify({"error": "Review text is required for feedback"}), 400

    if rating is None and not sentiment:
        return jsonify({"error": "Rating (1-5) or sentiment is required"}), 400

    try:
        rating_int = int(rating) if rating is not None else (5 if sentiment == "Positive" else (1 if sentiment == "Negative" else 3))
        result = analyzer.add_feedback(review_text, rating_int, sentiment)
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({"error": f"Failed to record feedback: {str(e)}"}), 500


@app.route("/api/feedback-stats", methods=["GET"])
def feedback_stats():
    """Get current feedback samples count."""
    return jsonify({
        "total_feedback": analyzer.get_feedback_count()
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "nlp-sentiment-analysis"})


@app.route("/ppt")
def serveppt():
    """Serve either the local ppt or send to google slides"""
    try:
         os.startfile("presentation.pptx")
         return jsonify({"status": "success"})
    except:
        return redirect("https://docs.google.com/presentation/d/19GMcBICGNVm4ydrjVryXyhJIoDDVpPsZCGZb3d62_EU/edit?usp=sharing")


@app.route("/report")
def servereport():
    """Serve either the local docx or send to google docs"""
    try:
        os.startfile("report.docx")
        return jsonify({"status": "success"})
    except:
        return redirect("https://docs.google.com/document/d/1qjUkwyCw_-foVY5ZcyRnH4ctMhVS4SMWK0zPYIG9l8U/edit?usp=sharing")

if __name__ == "__main__":
    # Run locally on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
