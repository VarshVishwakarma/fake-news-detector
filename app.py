import streamlit as st
import joblib
import requests
import os


try:
    vectorizer = joblib.load('vectorizer.jb')
    model = joblib.load('lr_model.jb')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'vectorizer.jb' and 'lr_model.jb' are present in the directory.")
    
    st.stop()



GEMINI_API_KEY = "AIzaSyBQHX3Ez610_q8TQi2Rm9-iIhP_BYNLspI"
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

def fetch_and_summarize_news_with_gemini(topic):
    """
    Uses Gemini with Google Search grounding to find and summarize a recent news article.
    """
    prompt = f"Find the most recent news article about '{topic}' and provide a concise summary of its content. The summary should be neutral and factual."
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}]
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(GEMINI_API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status() 
        data = response.json()
        
        candidate = data.get('candidates', [{}])[0]
        summary = candidate.get('content', {}).get('parts', [{}])[0].get('text')
        

        grounding_metadata = candidate.get('groundingMetadata', {})
        attribution = grounding_metadata.get('groundingAttributions', [{}])[0].get('web', {})
        title = attribution.get('title', 'No Title Found')
        url = attribution.get('uri', '#')

        if summary and title:
            return summary, title, url
        else:
            return None, None, None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch news via Gemini. Error: {e}")
        return None, None, None
    except (KeyError, IndexError):
        st.error("Could not parse the response from the Gemini API. The structure might have changed or the response was empty.")
        return None, None, None


st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 AI-Powered News Analyzer")
st.markdown("This tool uses the Gemini API to fetch, summarize, and analyze the latest news from the web to determine if it's likely real or fake.")

st.header("Analyze the Latest News")
st.write("Enter a topic to have Gemini find and check the latest article.")

topic_input = st.text_input("News Topic (e.g., 'global economy', 'advances in AI'):", "")

if st.button("Fetch and Check News"):
    if topic_input.strip():
        with st.spinner(f"Gemini is searching for news about '{topic_input}'..."):
            summary, article_title, article_url = fetch_and_summarize_news_with_gemini(topic_input)
            
            if summary:
                st.subheader("Fetched Article Details")
                st.markdown(f"**Title:** [{article_title}]({article_url})")
                st.info(f"**AI-Generated Summary being analyzed:**\n\n \"{summary}\"")
                
                transformed_input = vectorizer.transform([summary])
                prediction = model.predict(transformed_input)
                
                st.subheader("Analysis Result")
                if prediction[0] == 1:
                    st.success("✅ The news is likely REAL.")
                else:
                    st.error("❌ The news is likely FAKE.")
            else:
                st.warning("Could not find any news articles for that topic. Please try another one.")
    else:
        st.warning("Please enter a topic.")

<meta name="google-site-verification" content="1r5FxF8NU9p42aDKcS0B4HV-bUJ7atwq0AQ5bE-FIzg" />