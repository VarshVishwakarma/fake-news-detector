import streamlit as st
import joblib

vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('lr_model.jb')

st.title("Fake News Detector")
st.write("Enter the news text below to check if it's real or fake.")

news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if news_input.strip():
        transformed_input = vectorizer.transform([news_input])
        prediction = model.predict(transformed_input)
        
        if prediction[0] == 1:
            st.success("The news is Real.")
        else:
            st.error("The news is Fake.")
    else:
        st.warning("Please enter some text to analyze.")