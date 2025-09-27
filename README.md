# fake-news-detector
An AI-powered Fake News Detector that helps students identify misinformation. Built with Python, scikit-learn, and Streamlit, it classifies news articles as Real or Fake using a trained ML model. Simple, fast, and extendable with advanced NLP for accurate, trustworthy results.

# Fake News Detector  

## Overview  
Misinformation spreads rapidly online, making it difficult for students to identify trustworthy content.  
This **AI-powered Fake News Detector** uses Machine Learning to classify news articles as **Real** or **Fake**.  
Built with **Python, scikit-learn, and Streamlit**, it provides a clean interface to quickly analyze news credibility.  

---

## Features  
- Detects **Real vs Fake** news using a trained ML model  
- Simple **Streamlit web app** for easy interaction  
- **Logistic Regression model** with TF-IDF vectorizer  
- Fast predictions with saved model and vectorizer  
- Extendable to advanced NLP models like **BERT/RoBERTa**  

---

## Tech Stack  
- **Python 3**  
- **scikit-learn** (ML model training)  
- **joblib** (Model & vectorizer storage)  
- **Streamlit** (Web app framework)  
- **pandas, numpy** (Data handling)  

---

## Project Structure  
Fake-News-Detector/
│
├── app.py # Streamlit web app
├── model.jb # Trained ML model
├── vectorizer.jb # Saved vectorizer
├── requirements.txt # Dependencies
└── README.md # Project description

yaml
Copy code

---

## How to Run  
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/Fake-News-Detector.git
   cd Fake-News-Detector
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run app.py
Enter a news article text and check whether it is Real or Fake.
