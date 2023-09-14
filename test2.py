import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import subprocess

# ... (the rest of your imports and data preprocessing functions)

# Create a Streamlit app
st.title("Deployment Test")

# Create a text input field
user_input = st.text_area("Enter some text:", "")

# Create a selectbox to allow the user to choose the model
selected_model = st.selectbox("Select a Model", ["BernoulliNB", "LinearSVC"])

# Create a placeholder to display the prediction
prediction_placeholder = st.empty()

# Create a button to make predictions
if st.button("Make Prediction"):
    if user_input:
        model_to_use = None  # Initialize model_to_use
        if selected_model == "BernoulliNB":
            model_to_use = model1
        else:
            model_to_use = model2

        if model_to_use is not None:
            # Preprocess the user input for TF-IDF and BoW features
            preprocessed_input = preprocess_input_text(user_input)

            # Transform the preprocessed input using the corresponding TF-IDF vectorizer
            if model_to_use == model1:
                tfidf_input = tfidf_vectorizer.transform([preprocessed_input])
            else:
                tfidf_input = tfidf_vectorizer2.transform([preprocessed_input])

            # Make predictions using the selected model
            prediction = model_to_use.predict(tfidf_input)

            # Display the preprocessed input
            st.write(f"Preprocessed text: {preprocessed_input}")

            # Display the prediction result
            prediction_placeholder.write(f"Prediction: {prediction}")
