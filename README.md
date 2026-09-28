# NLP Mini Project: Product Review Sentiment Analysis

An e-commerce customer review sentiment analysis web application powered by a Python backend and machine learning pipeline trained on product review dataset.

![Screenshot of the UI](/static/screenshot.png)

## Features

- **Single Review Sentiment Ingestion**: Ingests product reviews to output sentiment classification (Positive, Neutral, Negative), continuous polarity score (-1.00 to +1.00), estimated 1–5 star rating, confidence percentage, class probabilities, and extracted sentiment keywords.
- **Batch Textfile Processing**: Upload any .txt file with reviews on separate lines to analyze each line individually and calculate the **overall average sentiment** across the whole file.
- **Interactive Review Browser**: Filter reviews by sentiment (All, Positive, Neutral, Negative), live search review content, and export processed results to CSV.
- **Live Training**: Debug mode allows input reviews to be saved and used to further optimize and train the model. Reviews are saved in user_feedback.csv.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Open your browser at [http://localhost:5000](http://localhost:5000).

### 3. Run Automated Tests
```bash
python -m unittest test_app.py
```

## API Endpoints

- `GET /`: Primary sentiment analysis webpage.
- `POST /api/analyze`: Ingest single review (`{ "review": "..." }`).
- `POST /api/analyze-file`: Multipart upload of `.txt` review files.
- `GET /api/sample-txt`: Download / inspect sample multi-line review text file.
