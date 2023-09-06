import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
import numpy as np

# Download the NLTK stopwords resource (only need to do this once)
nltk.download('stopwords')
# Load your pre-trained model (model1)
model1 = joblib.load("tolonglah.pkl")

# Define the text preprocessing function
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

        # Remove the first word in the sentence
        if len(words) > 1:
            words = words[1:]

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

# Create a file uploader to allow users to upload text data
user_input = st.text_area("Enter some text:", "")

# Preprocess the user input
if user_input:
    preprocessed_input = preprocess_text(user_input)

    # Vectorize the preprocessed input using TF-IDF and BoW
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    bow_vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2))
    
    # Transform the preprocessed input
    tfidf_features = tfidf_vectorizer.transform([preprocessed_input])
    bow_features = bow_vectorizer.transform([preprocessed_input])

    # Combine TF-IDF and BoW features
    combined_features = hstack([tfidf_features, bow_features])

    # Make predictions using model1
    prediction = model1.predict(combined_features)[0]

    # Display the prediction result
    st.write(f"Prediction: {prediction}")
