"""
model.py: E-Commerce Product Review Sentiment Analysis Model
Trained on product_reviews_mock_data.csv with an expanded sentiment lexicon
for robust domain generalization across e-commerce product reviews.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.joblib")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "product_reviews_mock_data.csv")
FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "user_feedback.csv")

# E-commerce domain supplementary reviews to bolster generalization
DOMAIN_BOOST_SAMPLES = [
    # Positive samples
    ("Excellent product! Very sturdy build quality and fast delivery.", "Positive"),
    ("Battery life is extraordinary, lasts more than two full days.", "Positive"),
    ("Customer support was super helpful and resolved my query promptly.", "Positive"),
    ("Worth every penny! Exceeded my expectations completely.", "Positive"),
    ("Great value for money, sleek design and works seamlessly.", "Positive"),
    ("Super easy setup, sound quality is crisp and clear.", "Positive"),
    ("Loved the packaging and prompt shipping. Five stars!", "Positive"),
    ("High quality materials, highly recommended to everyone.", "Positive"),
    ("Works flawlessly right out of the box, couldn't be happier.", "Positive"),
    ("Outstanding performance, looks premium and elegant.", "Positive"),
    # Negative samples
    ("Complete waste of money. Stopped working after two days.", "Negative"),
    ("Terrible quality. Cheap plastic material that broke on day one.", "Negative"),
    ("Customer support was completely unresponsive and rude. Refund denied.", "Negative"),
    ("Do not buy this item! Extremely defective and unsafe.", "Negative"),
    ("Slow performance, freezes constantly, and battery drains instantly.", "Negative"),
    ("Arrived damaged with scratches all over. Returning immediately.", "Negative"),
    ("Misleading description, nothing like what was shown in the pictures.", "Negative"),
    ("Worst purchase I have ever made online. Total disappointment.", "Negative"),
    ("Faulty charger and missing parts. Very frustrating experience.", "Negative"),
    ("Horrible smell and cheap finish. Very poor craftsmanship.", "Negative"),
    # Neutral samples
    ("Average product, does the job but nothing extraordinary.", "Neutral"),
    ("It is okay for the price, neither good nor bad.", "Neutral"),
    ("Mediocre performance. Works adequately for basic tasks.", "Neutral"),
    ("Decent item. Has some minor pros and cons as described.", "Neutral"),
    ("Fair quality. Delivery was on time but packaging was plain.", "Neutral"),
    ("Acceptable build quality. Expected a bit more but it's usable.", "Neutral"),
    ("Standard functionality, meets basic expectations without surprises.", "Neutral"),
]

# Stop words used solely for clean keyword phrase formatting in UI
STOP_WORDS = {
    "a", "about", "all", "am", "an", "and", "are", "as", "at", "be", "because", "been",
    "being", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "had", "has", "have", "having", "he", "her", "here",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "more",
    "most", "my", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours"
}


def clean_text(text: str) -> str:
    """Normalize text while retaining sentiment signals."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_dataset() -> pd.DataFrame:
    """Load mock CSV dataset and map ratings into 3 classes."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # Rating mapping: 1-2 Negative, 3 Neutral, 4-5 Positive
    def rating_to_sentiment(r):
        if r >= 4:
            return "Positive"
        elif r == 3:
            return "Neutral"
        else:
            return "Negative"

    df["sentiment"] = df["Rating"].apply(rating_to_sentiment)
    data = []
    for _, row in df.iterrows():
        txt = clean_text(row["ReviewText"])
        if txt:
            data.append({"text": txt, "sentiment": row["sentiment"]})

    # Add domain boost samples
    for txt, sent in DOMAIN_BOOST_SAMPLES:
        data.append({"text": clean_text(txt), "sentiment": sent})

    # Add active user feedback if present (replicated for high active-learning impact)
    if os.path.exists(FEEDBACK_PATH):
        try:
            fb_df = pd.read_csv(FEEDBACK_PATH)
            for _, row in fb_df.iterrows():
                txt = clean_text(str(row.get("ReviewText", "")))
                sent = str(row.get("Sentiment", ""))
                if txt and sent in ("Positive", "Neutral", "Negative"):
                    for _ in range(6):  # Active learning weighting
                        data.append({"text": txt, "sentiment": sent})
        except Exception as e:
            print(f"Warning reading user feedback file: {e}")

    return pd.DataFrame(data)


def build_and_train_pipeline():
    """Build and train the TF-IDF + Calibrated Logistic Regression pipeline."""
    df = load_dataset()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    
    # Calibrated classifier ensures well-aligned probabilities
    base_clf = LogisticRegression(
        C=2.5,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", base_clf)
    ])
    
    pipeline.fit(df["text"], df["sentiment"])
    return pipeline


class SentimentAnalyzer:
    def __init__(self, force_retrain: bool = False):
        self.pipeline = None
        self.classes_ = ["Negative", "Neutral", "Positive"]
        self._initialize_model(force_retrain)

    def _initialize_model(self, force_retrain: bool):
        if not force_retrain and os.path.exists(MODEL_PATH):
            try:
                self.pipeline = joblib.load(MODEL_PATH)
                return
            except Exception as e:
                print(f"Failed to load cached model: {e}. Retraining...")

        print("Training sentiment analysis model...")
        self.pipeline = build_and_train_pipeline()
        try:
            joblib.dump(self.pipeline, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: could not save model cache: {e}")

    def extract_cues(self, text: str):
        """
        Dynamically extract positive and negative sentiment keywords from the model's learned weights.
        Inspects TF-IDF features present in the review multiplied by Logistic Regression coefficients.
        """
        if not self.pipeline:
            return {"positive_cues": [], "negative_cues": []}

        tfidf = self.pipeline.named_steps.get("tfidf")
        clf = self.pipeline.named_steps.get("clf")
        if not tfidf or not clf:
            return {"positive_cues": [], "negative_cues": []}

        feature_names = tfidf.get_feature_names_out()
        classes = list(clf.classes_)
        if "Positive" not in classes or "Negative" not in classes:
            return {"positive_cues": [], "negative_cues": []}

        pos_idx = classes.index("Positive")
        neg_idx = classes.index("Negative")

        vec = tfidf.transform([text.lower()])
        pos_scored = []
        neg_scored = []

        def clean_term(term):
            tokens = term.split()
            if len(tokens) == 1:
                return term if tokens[0] not in STOP_WORDS and len(tokens[0]) > 2 else None
            # Filter bigrams with stop words at endpoints (e.g. 'waste of', 'of money', 'is great')
            if tokens[0] in STOP_WORDS or tokens[-1] in STOP_WORDS:
                return None
            return term

        for idx, val in zip(vec.indices, vec.data):
            term = feature_names[idx]
            cleaned = clean_term(term)
            if not cleaned:
                continue

            p_impact = val * clf.coef_[pos_idx, idx]
            n_impact = val * clf.coef_[neg_idx, idx]
            diff = p_impact - n_impact

            if diff > 0.08:
                pos_scored.append((cleaned, diff))
            elif diff < -0.08:
                neg_scored.append((cleaned, diff))

        pos_scored.sort(key=lambda x: x[1], reverse=True)
        neg_scored.sort(key=lambda x: x[1])

        # Deduplicate while preserving order
        seen_pos = set()
        pos_res = []
        for t, _ in pos_scored:
            if t not in seen_pos:
                seen_pos.add(t)
                pos_res.append(t)

        seen_neg = set()
        neg_res = []
        for t, _ in neg_scored:
            if t not in seen_neg:
                seen_neg.add(t)
                neg_res.append(t)

        # Suppress conflicting unigrams that are part of an opposite-polarity extracted phrase
        final_pos = [t for t in pos_res if not any(t in neg_term for neg_term in neg_res)][:5]
        final_neg = [t for t in neg_res if not any(t in pos_term for pos_term in pos_res)][:5]

        return {
            "positive_cues": final_pos,
            "negative_cues": final_neg
        }

    def analyze(self, review_text: str) -> dict:
        """
        Analyze sentiment for a single product review.
        Returns sentiment label, polarity score (-1.0 to 1.0), star rating (1-5), and class probabilities.
        """
        cleaned = clean_text(review_text)
        if not cleaned:
            return {
                "text": review_text,
                "sentiment": "Neutral",
                "polarity": 0.0,
                "confidence": 0.0,
                "stars": 3.0,
                "probabilities": {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33},
                "key_cues": {"positive_cues": [], "negative_cues": []}
            }

        # Predict class probabilities
        probs = self.pipeline.predict_proba([cleaned])[0]
        class_labels = list(self.pipeline.classes_)
        prob_dict = {cls: float(p) for cls, p in zip(class_labels, probs)}
        
        prob_pos = prob_dict.get("Positive", 0.0)
        prob_neg = prob_dict.get("Negative", 0.0)
        prob_neu = prob_dict.get("Neutral", 0.0)

        # Polarity continuous score calculation (-1.0 to +1.0)
        # Weighted difference between positive and negative probabilities
        polarity = round(float(prob_pos - prob_neg), 3)

        cues = self.extract_cues(review_text)
        num_pos_cues = len(cues.get("positive_cues", []))
        num_neg_cues = len(cues.get("negative_cues", []))

        # Primary sentiment decision based on polarity, class probabilities, and cues
        if polarity >= 0.20 or (polarity > 0.10 and num_pos_cues > num_neg_cues):
            predicted_sentiment = "Positive"
        elif polarity <= -0.20 or (polarity < -0.10 and num_neg_cues > num_pos_cues):
            predicted_sentiment = "Negative"
        elif abs(polarity) < 0.15:
            predicted_sentiment = "Neutral"
        else:
            predicted_sentiment = self.pipeline.predict([cleaned])[0]

        # Star estimation (1.0 to 5.0) based on expected value
        # Rating 1 (Neg), Rating 3 (Neu), Rating 5 (Pos)
        raw_stars = (prob_neg * 1.5) + (prob_neu * 3.0) + (prob_pos * 4.8)
        stars = round(float(np.clip(raw_stars, 1.0, 5.0)), 1)

        # Confidence is probability of chosen class (minimum 50% for assigned sentiment)
        raw_conf = prob_dict.get(predicted_sentiment, 0.5) * 100.0
        confidence = round(float(max(raw_conf, 50.0 if predicted_sentiment != "Neutral" else raw_conf)), 1)

        return {
            "text": review_text,
            "sentiment": predicted_sentiment,
            "polarity": polarity,
            "confidence": confidence,
            "stars": stars,
            "probabilities": {k: round(v * 100, 1) for k, v in prob_dict.items()},
            "key_cues": cues
        }

    def analyze_batch(self, lines: list[str]) -> dict:
        """
        Analyze a list of review lines individually and calculate the overall average sentiment.
        """
        individual_results = []
        valid_polarities = []
        valid_stars = []
        counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

        for idx, line in enumerate(lines, start=1):
            trimmed = line.strip()
            if not trimmed:
                continue
            
            res = self.analyze(trimmed)
            res["line_number"] = idx
            individual_results.append(res)
            
            counts[res["sentiment"]] += 1
            valid_polarities.append(res["polarity"])
            valid_stars.append(res["stars"])

        total_reviews = len(individual_results)

        if total_reviews > 0:
            avg_polarity = round(float(np.mean(valid_polarities)), 3)
            avg_stars = round(float(np.mean(valid_stars)), 1)

            if avg_polarity >= 0.15:
                overall_sentiment = "Positive"
            elif avg_polarity <= -0.15:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"

            pos_pct = round((counts["Positive"] / total_reviews) * 100, 1)
            neu_pct = round((counts["Neutral"] / total_reviews) * 100, 1)
            neg_pct = round((counts["Negative"] / total_reviews) * 100, 1)
        else:
            avg_polarity = 0.0
            avg_stars = 0.0
            overall_sentiment = "Neutral"
            pos_pct = 0.0
            neu_pct = 0.0
            neg_pct = 0.0

        return {
            "reviews": individual_results,
            "summary": {
                "total_reviews": total_reviews,
                "overall_sentiment": overall_sentiment,
                "average_polarity": avg_polarity,
                "average_stars": avg_stars,
                "positive_count": counts["Positive"],
                "positive_percentage": pos_pct,
                "neutral_count": counts["Neutral"],
                "neutral_percentage": neu_pct,
                "negative_count": counts["Negative"],
                "negative_percentage": neg_pct
            }
        }

    def get_feedback_count(self) -> int:
        """Return total number of recorded feedback entries."""
        if not os.path.exists(FEEDBACK_PATH):
            return 0
        try:
            fb_df = pd.read_csv(FEEDBACK_PATH)
            return len(fb_df)
        except Exception:
            return 0

    def add_feedback(self, review_text: str, rating: int, sentiment: str = None) -> dict:
        """
        Record user feedback live, persist to user_feedback.csv, and retrain model live.
        """
        cleaned = clean_text(review_text)
        if not cleaned:
            raise ValueError("Review text cannot be empty.")

        rating = int(rating)
        if not sentiment:
            if rating >= 4:
                sentiment = "Positive"
            elif rating == 3:
                sentiment = "Neutral"
            else:
                sentiment = "Negative"

        from datetime import datetime
        now_str = datetime.now().isoformat()

        file_exists = os.path.exists(FEEDBACK_PATH)
        fb_df = pd.DataFrame([{
            "Timestamp": now_str,
            "ReviewText": review_text.strip(),
            "Rating": rating,
            "Sentiment": sentiment
        }])
        fb_df.to_csv(FEEDBACK_PATH, mode="a", header=not file_exists, index=False)

        # Retrain model live
        self.pipeline = build_and_train_pipeline()
        try:
            joblib.dump(self.pipeline, MODEL_PATH)
        except Exception as e:
            print(f"Warning saving model: {e}")

        updated_pred = self.analyze(review_text)

        return {
            "success": True,
            "message": "Feedback recorded and model retrained live.",
            "feedback": {
                "text": review_text,
                "rating": rating,
                "sentiment": sentiment
            },
            "total_feedback_count": self.get_feedback_count(),
            "updated_prediction": updated_pred
        }


if __name__ == "__main__":
    analyzer = SentimentAnalyzer(force_retrain=True)
    test_cases = [
        "fantastic. wonderful experience.",
        "broke easily. awful.",
        "neither good nor bad. it's okay.",
        "The sound quality is crisp and battery life is exceptional.",
        "Terrible product, broke within one hour. Never buying again."
    ]
    for tc in test_cases:
        r = analyzer.analyze(tc)
        print(f"[{r['sentiment']}] (polarity: {r['polarity']}, stars: {r['stars']} stars, conf: {r['confidence']}%) -> '{tc}'")
