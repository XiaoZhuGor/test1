import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import subprocess

# Import matplotlib after installation
import matplotlib.pyplot as plt

# Download the stopwords resource
nltk.download('stopwords')

# Define preprocessing functions (same as before)

# Load your pre-trained models (model1 and model2)
model1 = joblib.load("bnb_smote.pkl")  # Replace with your model file path
model2 = joblib.load("LinearSVC_smote.pkl")

# Load your CSV data into a DataFrame
data = pd.read_csv('Tweets.csv', encoding='latin1')

# Apply preprocessing to the 'text' column using .apply()
data['cleaned_data'] = data['text'].apply(preprocess_input_text)

# Create a Streamlit app
st.title("Sentiment Analysis Test")

# Create a form to encapsulate the input fields
with st.form("text_prediction_form"):
    # Create a text input field
    user_input = st.text_area("Enter Text Prediction Here:", "")

    # Create a selectbox to allow the user to choose the model
    selected_model = st.selectbox("Select a Model", ["BernoulliNB", "LinearSVC"])

    # Create a button to make predictions
    prediction_button = st.form_submit_button("Make Prediction")

# Check if the form is submitted
if prediction_button:
    if user_input:
        if selected_model == "BernoulliNB":
            # Preprocess the user input for TF-IDF and BoW features
            preprocessed_input = preprocess_input_text(user_input)

            # Transform the preprocessed input using the same TF-IDF vectorizer
            tfidf_input = tfidf_vectorizer.transform([preprocessed_input])

            # Make predictions using model1
            prediction = model1.predict(tfidf_input)

            # Display the preprocessed input
            st.write(f"Preprocessed text: {preprocessed_input}")

            # Display the prediction result
            st.write(f"Prediction: {prediction}")
        else:
            # Preprocess the user input for TF-IDF and BoW features
            preprocessed_input = preprocess_input_text(user_input)

            # Transform the preprocessed input using the same TF-IDF vectorizer
            tfidf_input = tfidf_vectorizer2.transform([preprocessed_input])

            # Make predictions using model2
            prediction = model2.predict(tfidf_input)

            # Display the preprocessed input
            st.write(f"Preprocessed text: {preprocessed_input}")

            # Display the prediction result
            st.write(f"Prediction: {prediction}")
