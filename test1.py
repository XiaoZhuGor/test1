import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers and special characters using regular expressions
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    # Split the text into sentences
    sentences = re.split(r'[.!?]', text)

    # Process each sentence
    cleaned_sentences = []
    for sentence in sentences:
        words = sentence.split()

        # Remove stopwords and apply stemming
        words = [word for word in words if word not in stop_words]

        # Apply stemming
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in words]

        # Join stemmed words back into a sentence
        cleaned_sentence = ' '.join(stemmed_words)
        cleaned_sentences.append(cleaned_sentence)

    # Join the cleaned sentences back together
    cleaned_text = ' '.join(cleaned_sentences)

    return cleaned_text

# Load your pre-trained model (model1)
model1 = joblib.load("tolonglah.pkl")

# Create a Streamlit app
st.title("deployment test")

# Create a text input field
user_input = st.text_area("Enter some text:", "")

# Create a button to make predictions
if st.button("Make Prediction"):
    if user_input:
         # Preprocess the user input
        preprocessed_input = preprocess_text(user_input)

        # Make predictions using model1
        prediction = model1.predict([preprocessed_input])  # Pass a list containing the input

        # Reshape the prediction to 1D array
        prediction = prediction.reshape(-1)

        # Display the prediction result
        st.write(f"Prediction: {prediction[0]}")  # Access the first (and only) prediction in the list
