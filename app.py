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

# --- STEP 4: API CONFIG (LATEST STABLE DECEMBER 2025 MODELS) ---
GEMINI_API_KEY = "AIzaSyBQHX3Ez610_q8TQi2Rm9-iIhP_BYNLspI"
# gemini-2.5-flash is the new stable standard as of late 2025
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

def fetch_and_summarize_news_with_gemini(topic):
    """
    Uses Gemini to find/summarize news. 
    Handles 403/404 by immediately switching to internal AI reasoning.
    """
    prompt = f"Summarize the most recent news about '{topic}'. Provide a concise, neutral, and factual summary."
    
    # Attempt with Google Search grounding (may trigger 403 on free tier)
    payload_with_search = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}]
    }
    
    # Fallback without search tools (works on all free keys)
    payload_no_search = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        # First attempt: Try with Search
        response = requests.post(GEMINI_API_ENDPOINT, json=payload_with_search, headers=headers)
        
        # If Forbidden (403) or Not Found (404), immediately use the fallback payload
        if response.status_code in [403, 404]:
            response = requests.post(GEMINI_API_ENDPOINT, json=payload_no_search, headers=headers)
        
        response.raise_for_status() 
        data = response.json()
        
        # Extract content
        candidate = data.get('candidates', [{}])[0]
        parts = candidate.get('content', {}).get('parts', [])
        summary = parts[0].get('text') if parts else None
        
        # Handle metadata for Title and URL
        grounding_metadata = candidate.get('groundingMetadata', {})
        attributions = grounding_metadata.get('groundingAttributions', [])
        
        if attributions:
            web_info = attributions[0].get('web', {})
            title = web_info.get('title', f"Recent News: {topic}")
            url = web_info.get('uri', 'https://news.google.com')
        else:
            title = f"AI News Report: {topic}"
            url = "https://news.google.com"

        return summary, title, url
            
    except Exception as e:
        st.error(f"News Retrieval Error: {str(e)}")
        return None, None, None

# --- STEP 5: APP UI ---
st.title("📰 AI-Powered News Analyzer")
st.markdown("Verify news authenticity using machine learning and Google's latest Gemini 2.5 AI.")

st.header("Search & Verify")
topic_input = st.text_input("Enter a news topic to check:", placeholder="e.g. global climate changes")

if st.button("Fetch and Check News", use_container_width=True):
    if topic_input.strip():
        with st.spinner(f"AI is researching '{topic_input}'..."):
            summary, article_title, article_url = fetch_and_summarize_news_with_gemini(topic_input)
            
            if summary:
                st.divider()
                st.subheader("Source Document")
                st.markdown(f"**Title:** [{article_title}]({article_url})")
                st.info(f"**Content Summary:**\n\n {summary}")
                
                # Machine Learning Prediction
                try:
                    transformed_input = vectorizer.transform([summary])
                    prediction = model.predict(transformed_input)
                    
                    st.subheader("Authenticity Rating")
                    if prediction[0] == 1:
                        st.success("✅ **Likely REAL News**")
                    else:
                        st.error("❌ **Likely FAKE/UNRELIABLE**")
                except Exception as ml_err:
                    st.error(f"Machine Learning Module Error: {ml_err}")
            else:
                st.warning("Could not retrieve information. Try a different topic.")
    else:
        st.warning("Please enter a topic first.")

st.sidebar.markdown("### Project Info")
st.sidebar.info("This tool combines a Logistic Regression model with Gemini 2.5 Flash for advanced news verification.")
