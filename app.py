import streamlit as st
import joblib
import requests
import os

# --- STEP 1: SET PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="Free Fake News Detector | AI-Powered Analysis",
    page_icon="📰",
    layout="centered"
)

# --- STEP 2: GOOGLE VERIFICATION TAG ---
st.markdown("""
<meta name="google-site-verification" content="1r5FxF8NU9p42aDKcS0B4HV-bUJ7atwq0AQ5bE-FIzg" />
""", unsafe_allow_html=True)

# --- STEP 3: LOAD MODELS SAFELY ---
try:
    # Using 'rb' for joblib files is safer
    vectorizer = joblib.load('vectorizer.jb')
    model = joblib.load('lr_model.jb')
except Exception as e:
    st.error(f"CRITICAL BOOT ERROR: Failed to load model files.")
    st.error(f"Error details: {e}")
    st.stop()

# --- STEP 4: API CONFIG (STABLE MODELS) ---
GEMINI_API_KEY = "AIzaSyBQHX3Ez610_q8TQi2Rm9-iIhP_BYNLspI"
# gemini-1.5-flash is currently the most robust for tool usage across regions
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def fetch_and_summarize_news_with_gemini(topic):
    """
    Uses Gemini to find/summarize news. Includes a fallback if the search tool is forbidden (403).
    """
    prompt = f"Find the most recent news article about '{topic}' and provide a concise summary of its content. The summary should be neutral and factual."
    
    # Try with Google Search grounding first
    payload_with_search = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}]
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        # ATTEMPT 1: With Search Tool
        response = requests.post(GEMINI_API_ENDPOINT, json=payload_with_search, headers=headers)
        
        # If we get a 403, the API Key/Region likely doesn't support the Search Tool
        if response.status_code == 403:
            st.warning("⚠️ Google Search tool access is restricted for this API key. Falling back to internal AI knowledge...")
            payload_no_search = {
                "contents": [{"parts": [{"text": f"Summarize the latest known facts about the topic: {topic}. Be neutral and factual."}]}]
            }
            response = requests.post(GEMINI_API_ENDPOINT, json=payload_no_search, headers=headers)
            
        response.raise_for_status() 
        data = response.json()
        
        candidate = data.get('candidates', [{}])[0]
        summary = candidate.get('content', {}).get('parts', [{}])[0].get('text')
        
        # Metadata might be missing in fallback mode
        grounding_metadata = candidate.get('groundingMetadata', {})
        attribution = grounding_metadata.get('groundingAttributions', [{}])[0].get('web', {}) if grounding_metadata else {}
        title = attribution.get('title', f"News Report: {topic}")
        url = attribution.get('uri', 'https://news.google.com')

        if summary:
            return summary, title, url
        else:
            return None, None, None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Gemini API Error: {e}")
        return None, None, None
    except (KeyError, IndexError):
        st.error("Could not parse the response from the Gemini API.")
        return None, None, None

# --- STEP 5: APP UI ---
st.title("📰 AI-Powered News Analyzer")
st.markdown("This tool uses the Gemini API to fetch, summarize, and analyze the latest news.")

st.header("Analyze the Latest News")
topic_input = st.text_input("News Topic (e.g., 'global economy'):", "")

if st.button("Fetch and Check News"):
    if topic_input.strip():
        with st.spinner(f"Searching for news about '{topic_input}'..."):
            summary, article_title, article_url = fetch_and_summarize_news_with_gemini(topic_input)
            
            if summary:
                st.subheader("Fetched Article Details")
                st.markdown(f"**Source:** [{article_title}]({article_url})")
                st.info(f"**AI Summary:**\n\n \"{summary}\"")
                
                # Machine Learning Prediction
                try:
                    transformed_input = vectorizer.transform([summary])
                    prediction = model.predict(transformed_input)
                    
                    st.subheader("Analysis Result")
                    if prediction[0] == 1:
                        st.success("✅ The news is likely REAL.")
                    else:
                        st.error("❌ The news is likely FAKE.")
                except Exception as ml_err:
                    st.error(f"ML Model Error: {ml_err}")
            else:
                st.warning("Could not retrieve info for that topic.")
    else:
        st.warning("Please enter a topic.")
