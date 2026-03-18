# Twitter Sentiment Analysis System

An end-to-end NLP system for sentiment classification on real-world tweet data, combining traditional machine learning and deep learning approaches to analyze, compare, and understand text-based sentiment patterns.

---

## 🚀 Overview

This project implements a complete sentiment analysis pipeline that transforms raw tweet text into structured predictions using multiple modeling approaches.

The system is designed to:
- Handle noisy social media text
- Compare feature engineering techniques
- Evaluate model performance across different architectures
- Analyze limitations of NLP models on real-world data

---

## 🧠 System Architecture

### 1. Data Processing
- Dynamic dataset loading (`train_data.csv`, `test_data.csv`)
- Handling missing values
- Cleaning noisy tweet data (URLs, mentions, special characters)
- Tokenization and normalization

### 2. Feature Engineering
- CountVectorizer
- TF-IDF Vectorizer
- Word embeddings for deep learning models

### 3. Modeling Approaches

#### Traditional ML
- Logistic Regression on CountVectorizer and TF-IDF

#### Deep Learning
- Feedforward Neural Network
- Convolutional Neural Network (CNN) using embeddings

### 4. Evaluation & Analysis
- Accuracy, confusion matrix, classification report
- Model comparison
- Error analysis

---

## 📊 Key Insights

- Frequency-based features performed competitively
- Deep learning improved contextual understanding
- Challenges:
  - Negation
  - Sarcasm
  - Informal language
  - Mixed sentiment

---

## 📈 Visualization

- Top frequent words analysis

---

## 📂 Project Structure

.
├── data/
│   ├── train_data.csv
│   ├── test_data.csv
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   ├── evaluation.py
│   ├── nn_model.py
│   ├── cnn_model.py
│
├── main.py
├── requirements.txt
├── top_words_visualization.png

---

## ⚙️ Setup

git clone https://github.com/your-username/your-repo.git
cd your-repo

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

---

## ▶️ Run

python main.py

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / PyTorch
- Matplotlib / Seaborn

---

## 🧩 Highlights

- Modular NLP pipeline
- Comparison of ML vs DL approaches
- Detailed evaluation and error analysis
- Reproducible and extensible system

