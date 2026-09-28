"""
test_app.py: Automated tests for E-Commerce Sentiment Analysis Web Application
Tests endpoints: GET /, POST /api/analyze, POST /api/analyze-file, GET /api/sample-txt
"""

import io
import unittest
from app import app


class SentimentAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    def test_homepage_renders(self):
        """Verify homepage renders with required title, icons, and cardboard accent classes."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        # Check title requirement
        self.assertIn("<title>sentiment analysis</title>", html)
        self.assertIn("sentiment analysis</h1>", html)
        # Check top-right links to example.com/whatever
        self.assertIn('href="https://example.com/whatever"', html)
        # Check squircle classes
        self.assertIn("squircle-card", html)
        self.assertIn("squircle-dropzone", html)
        self.assertIn("squircle-textarea", html)

    def test_single_review_positive(self):
        """Test analyzing a positive review."""
        payload = {"review": "Fantastic product! Exceeded my expectations, works perfectly and high quality."}
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]
        self.assertEqual(data["sentiment"], "Positive")
        self.assertGreater(data["polarity"], 0.0)
        self.assertGreaterEqual(data["stars"], 3.5)

    def test_single_review_negative(self):
        """Test analyzing a negative review."""
        payload = {"review": "Broke easily on the first day. Terrible quality, complete waste of money."}
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]
        self.assertEqual(data["sentiment"], "Negative")
        self.assertLess(data["polarity"], 0.0)
        self.assertLessEqual(data["stars"], 2.5)

    def test_single_review_neutral(self):
        """Test analyzing a neutral review."""
        payload = {"review": "Average product. Does what it is supposed to do, neither good nor bad."}
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]
        self.assertEqual(data["sentiment"], "Neutral")

    def test_single_review_empty_error(self):
        """Test empty review payload error handling."""
        res = self.client.post("/api/analyze", json={"review": "   "})
        self.assertEqual(res.status_code, 400)

    def test_batch_file_upload(self):
        """Test batch uploading a .txt file with reviews on separate lines."""
        file_content = (
            "Fantastic product, love it so much!\n"
            "Terrible customer service and awful broken device.\n"
            "Average item, does the job adequately.\n"
            "Wonderful experience and fast delivery.\n"
        )
        data = {
            "file": (io.BytesIO(file_content.encode("utf-8")), "test_reviews.txt")
        }
        res = self.client.post(
            "/api/analyze-file",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()["data"]
        
        # Verify individual review analyses
        reviews = json_data["reviews"]
        self.assertEqual(len(reviews), 4)
        self.assertEqual(reviews[0]["line_number"], 1)
        self.assertEqual(reviews[0]["sentiment"], "Positive")
        self.assertEqual(reviews[1]["line_number"], 2)
        self.assertEqual(reviews[1]["sentiment"], "Negative")

        # Verify summary and overall average sentiment
        summary = json_data["summary"]
        self.assertEqual(summary["total_reviews"], 4)
        self.assertIn(summary["overall_sentiment"], ["Positive", "Neutral", "Negative"])
        self.assertIn("average_polarity", summary)
        self.assertIn("average_stars", summary)
        self.assertEqual(summary["positive_count"] + summary["neutral_count"] + summary["negative_count"], 4)

    def test_sample_txt_endpoint(self):
        """Test GET /api/sample-txt returns text/plain sample."""
        res = self.client.get("/api/sample-txt")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(len(res.data) > 0)

    def test_live_feedback_training(self):
        """Test submitting live training feedback and verifying instant model update."""
        review_text = "Unique gadget with peculiar design."
        # Initial analysis
        init_res = self.client.post("/api/analyze", json={"review": review_text})
        self.assertEqual(init_res.status_code, 200)

        # Teach model that this is 5-star Positive
        fb_payload = {
            "review": review_text,
            "rating": 5,
            "sentiment": "Positive"
        }
        fb_res = self.client.post("/api/feedback", json=fb_payload)
        self.assertEqual(fb_res.status_code, 200)
        data = fb_res.get_json()["data"]
        self.assertTrue(data["success"])
        self.assertGreater(data["total_feedback_count"], 0)
        # Updated prediction should be Positive
        self.assertEqual(data["updated_prediction"]["sentiment"], "Positive")

    def test_feedback_stats(self):
        """Test GET /api/feedback-stats."""
        res = self.client.get("/api/feedback-stats")
        self.assertEqual(res.status_code, 200)
        self.assertIn("total_feedback", res.get_json())


if __name__ == "__main__":
    unittest.main()
