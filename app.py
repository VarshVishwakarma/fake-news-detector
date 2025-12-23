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

# --- STEP 4: API CONFIG (LATEST STABLE MODELS) ---
GEMINI_API_KEY = "AIzaSyBQHX3Ez610_q8TQi2Rm9-iIhP_BYNLspI"
# Primary Endpoint
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def fetch_and_summarize_news_with_gemini(topic):
    """
    Uses Gemini to find/summarize news. 
    Handles 403 Forbidden by falling back to non-grounded internal logic.
    """
    prompt = f"Find the most recent news article about '{topic}' and provide a concise summary of its content. The summary should be neutral and factual."
    
    # 1. Configuration for grounded search
    payload_with_search = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}]
    }
    
    # 2. Configuration for fallback (no search tool)
    payload_no_search = {
        "contents": [{"parts": [{"text": f"Provide a factual summary of the most recent significant news regarding: {topic}."}]}]
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        # ATTEMPT 1: Try with Google Search tool
        response = requests.post(GEMINI_API_ENDPOINT, json=payload_with_search, headers=headers)
        
        # PERMANENT FIX: Handle 403 Forbidden (restricted tool access)
        if response.status_code == 403:
            st.info("ℹ️ Google Search tool is restricted for this key. Using internal AI knowledge instead...")
            response = requests.post(GEMINI_API_ENDPOINT, json=payload_no_search, headers=headers)
        
        # Handle 404 (model name change/deprecated)
        if response.status_code == 404:
            st.warning("⚠️ Model deprecated. Switching to Pro model...")
            fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
            response = requests.post(fallback_url, json=payload_no_search, headers=headers)

        response.raise_for_status() 
        data = response.json()
        
        # Extract content
        try:
            candidate = data.get('candidates', [{}])[0]
            summary = candidate.get('content', {}).get('parts', [{}])[0].get('text')
            
            # Metadata handling (only exists if search worked)
            grounding_metadata = candidate.get('groundingMetadata', {})
            attributions = grounding_metadata.get('groundingAttributions', [])
            
            if attributions:
                web_info = attributions[0].get('web', {})
                title = web_info.get('title', f"News Report: {topic}")
                url = web_info.get('uri', 'https://news.google.com')
            else:
                title = f"AI Analysis: {topic}"
                url = "https://news.google.com"

            if summary:
                return summary, title, url
        except (KeyError, IndexError):
            return None, None, None
            
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        return None, None, None
    
    return None, None, None

# --- STEP 5: APP UI ---
st.title("📰 AI-Powered News Analyzer")
st.markdown("Analyze news authenticity using machine learning and Google's Gemini AI.")

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
                    # Pre-process text using our loaded vectorizer
                    transformed_input = vectorizer.transform([summary])
                    prediction = model.predict(transformed_input)
                    
                    st.subheader("Authenticity Rating")
                    if prediction[0] == 1:
                        st.success("✅ **Likely REAL News**")
                        st.write("Our model suggests this content aligns with typical factual reporting patterns.")
                    else:
                        st.error("❌ **Likely FAKE/UNRELIABLE**")
                        st.write("Our model detected patterns often associated with misinformation or clickbait.")
                except Exception as ml_err:
                    st.error(f"Machine Learning Module Error: {ml_err}")
            else:
                st.warning("No data found. Please try a more specific news topic.")
    else:
        st.warning("Please enter a topic first.")

st.sidebar.markdown("### About")
st.sidebar.info("This project combines a Logistic Regression ML model with Gemini's reasoning capabilities to help users identify potential misinformation.")
