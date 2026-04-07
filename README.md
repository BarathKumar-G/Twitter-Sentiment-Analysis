# Twitter Sentiment Analysis

## Overview
This project implements an end-to-end NLP pipeline designed for sentiment classification on noisy social media data. It features comprehensive preprocessing improvements to aggressively clean social media text, extraction layers, multiple machine learning models, and deployment environments utilizing both a Streamlit dashboard and a FastAPI interface.

## Features
- Comprehensive preprocessing targeting noisy syntax (lemmatization, negation handling, and elongation normalization)
- Scalable feature extraction applying CountVectorizer and TF-IDF indexing
- Training pipelines processing Logistic Regression and Naive Bayes models
- Dedicated evaluation frameworks resolving evaluation metrics
- Visual model explainability via top words mapping
- Automated Exploratory Data Analysis (EDA) plotting logic
- Deployed through a streamlined FastAPI API and interactive Streamlit frontend

## Project Structure
```text
.
├── src/
│   ├── api.py
│   ├── evaluation.py
│   ├── features.py
│   ├── inference.py
│   ├── models.py
│   └── preprocessing.py
├── frontend/ (optional integration paths)
├── outputs/
├── models/
├── main.py
├── app.py
└── requirements.txt
```

## Installation
Setup a virtual environment and install the required sequence dependencies natively:

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Unix: source venv/bin/activate

pip install -r requirements.txt
```

Ensure the necessary secondary NLTK components are fully downloaded:
```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

## Running the Pipeline
To initialize, train, and test the analysis sequence, run:

```bash
python main.py
```

Outputs generated during runtime explicitly include:
- Models saved directly into the /models directory (best configurations)
- Plots generated mapping analytics exported dynamically to the /outputs directory
- Top words saved representing feature importances inside outputs/top_words.txt

## Streamlit Dashboard
To launch the interactive frontend dashboard, run:

```bash
streamlit run app.py
```

The accessible UI natively connects to:
- Real-time user text sentiment prediction
- EDA mapping and chart review
- A preprocessing demo outlining pipeline sanitation procedures

## FastAPI Inference API
To initialize the backend inference node directly, run:

```bash
uvicorn src.api:app --reload
```

The core service resolves endpoints locally:
**Endpoint:** `/predict`

**Example Request:**
```json
{
  "text": "This is an amazing project"
}
```

**Example Response:**
```json
{
  "prediction": "Positive",
  "probability": 0.8932
}
```

## Model Performance
- Best Model Evaluated: Logistic Regression + TF-IDF
- Core Accuracy ≈ 83%
- Core ROC-AUC ≈ 0.90

## Key Improvements
Aggressive adjustments ensure normalization algorithms preserve validity while parsing slang arrays correctly:
- Strict negation phrasing converts tokens (`not good` strictly translates into `not_good`)
- Elongation normalization accurately limits excessive token repetitions (`loooove` is clipped effectively into `love`)
- Automated lemmatization securely ties active terms downwards into dictionary strings (`running` into `run`)

## Example Predictions

| Input Text | Preprocessed Tokens | Sentiment Prediction |
| :--- | :--- | :--- |
| This is wonderful | wonder | Positive |
| I strongly dislike waiting | strongly dislike wait | Negative |
| This is not good | not_good | Negative |

## Limitations
- Frequent struggles classifying heavy sarcasm securely.
- General lack of context stemming from short parameter word unigrams.
- Significant ambiguity processing modern slang structures without proper context arrays.

## Future Improvements
- Rebuilding sequential modeling toward BERT and transformers configurations.
- Focusing logic into modeling better context handling over larger document bodies.

## Summary
The Twitter Sentiment Analysis pipeline establishes an effective, thoroughly sanitized NLP architecture specifically suited for predicting localized sentiment across noisy social media arrays by marrying rigorous regex token mapping with performant mathematical scaling distributions.
