AI-Powered Fake News Detector

A web application built with Streamlit, leveraging machine learning and the Gemini API to detect whether a news article is likely real or fake.

Overview

This project provides a user-friendly interface where users can enter a topic of interest. It then uses the Gemini API to perform a web search, retrieve a recent news article, generate a concise summary, and analyze it using a pre-trained Logistic Regression model to classify the article as "Real" or "Fake".

Features

Real-time News Analysis: Fetches and analyzes the latest news articles from the web.

AI-Powered Summarization: Utilizes Google's Gemini API to generate concise summaries.

Machine Learning Classification: Uses a trained TF-IDF Vectorizer and Logistic Regression model to detect fake news.

Source Linking: Provides a direct link to the original article for verification.

Simple Web Interface: Easy-to-use UI built with Streamlit.

How It Works

User Input: Enter a news topic (e.g., "global economy") in the text field.

API Call: The app uses the Gemini API to search the web for the latest related news.

Summarization: The Gemini API returns:

A concise AI-generated summary.

The original article title and URL.

Text Transformation: The summary is transformed into a numerical format using a pre-trained TF-IDF Vectorizer (vectorizer.jb).

Prediction: The transformed data is fed into a pre-trained Logistic Regression model (lr_model.jb), producing a prediction:

1 = Real

0 = Fake

Display Results: The app shows:

Article title

Summary

Authenticity prediction (color-coded)

Link to the original source

Getting Started

Prerequisites

Python 3.7 or higher

Pre-trained model files:

vectorizer.jb

lr_model.jb

Installation

Clone the repository:

git clone https://your-repository-url.git
cd your-project-folder


Create requirements.txt with the following content:

streamlit
joblib
requests
scikit-learn


Install dependencies:

pip install -r requirements.txt


Place model files: Ensure vectorizer.jb and lr_model.jb are in the same directory as app.py.

API Key

The Gemini API key is already hardcoded into app.py. No additional configuration is needed.

Usage

Run the Streamlit app:

streamlit run app.py


Open in browser:
Streamlit will display a local URL (e.g., http://localhost:8501).
Open it in your browser.

Analyze news:
Enter a topic and click "Fetch and Check News" to get results.

Project Files

app.py — Main application logic and API calls.

vectorizer.jb — Saved TF-IDF vectorizer object.

lr_model.jb — Saved Logistic Regression model object.

README.md — This file.

Future Improvements

Add multi-language support.

Improve model accuracy using deep learning-based NLP models.

Integrate a confidence score with predictions.

Add user authentication for personalized analysis.
