📰 AI-Powered Fake News Detector

A web application built with Streamlit, leveraging machine learning and the Gemini API to detect whether a news article is likely real or fake.

🔍 Overview

This project provides a user-friendly interface for detecting fake news. Users can enter a topic of interest, and the app will:

Fetch the latest news article using the Gemini API.

Generate a concise AI summary.

Analyze the text with a pre-trained Logistic Regression model to classify it as Real or Fake.

✨ Features

Real-time News Analysis: Fetches and analyzes the latest news articles.

AI-Powered Summarization: Uses Gemini API to generate concise summaries.

Machine Learning Classification: TF-IDF + Logistic Regression for fake news detection.

Source Linking: Direct link to the original article for verification.

Simple Web Interface: Easy-to-use UI powered by Streamlit.

⚙️ How It Works

User Input: Enter a news topic (e.g., "global economy") in the text field.

API Call: App fetches the latest news related to the topic via Gemini API.

Summarization: Returns an AI-generated summary, title, and URL.

Text Transformation: Summary is converted using TF-IDF vectorizer (vectorizer.jb).

Prediction: Logistic Regression model (lr_model.jb) predicts:

1 → Real

0 → Fake

Display Results: Shows title, summary, authenticity prediction (color-coded), and original source link.

🚀 Getting Started
Prerequisites

Python 3.7+

Pre-trained model files:

vectorizer.jb

lr_model.jb

Installation
# Clone the repository
git clone https://your-repository-url.git
cd your-project-folder

# Install dependencies
pip install -r requirements.txt


requirements.txt:

streamlit
joblib
requests
scikit-learn

Usage

Place model files (vectorizer.jb, lr_model.jb) in the same directory as app.py.

Run the Streamlit app:

streamlit run app.py


Open the local URL (e.g., http://localhost:8501
) in your browser.

Enter a topic and click “Fetch and Check News” to get results.

🗂️ Project Files
File	Description
app.py	Main application logic & API calls
vectorizer.jb	Saved TF-IDF vectorizer object
lr_model.jb	Saved Logistic Regression model object
README.md	Project documentation
🔮 Future Improvements

Multi-language support.

Improve accuracy with deep learning-based NLP models.

Integrate confidence scores with predictions.

Add user authentication for personalized analysis.

📈 Screenshot / Demo

(You can insert a screenshot of the app UI here for extra style)

🛡️ License

This project is licensed under the MIT License — see the LICENSE
 file for details.

🧑‍💻 Author

Varsh Vishwakarma AI • ML • DL • Data Science • Cloud • Full-Stack ML Developer
