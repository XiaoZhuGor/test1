import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

# Create a Streamlit app
st.title("Text Classification App")
st.write("Enter some text and I'll predict the class.")

# Create a text input field for user input
user_input = st.text_area("Enter some text:", "")

# Preprocess the user input and make predictions
if user_input:
    preprocessed_input = preprocess_text(user_input)

    # Create a new TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    # Fit the vectorizer with your data (you need a corpus of documents to fit it)
    # tfidf_vectorizer.fit(your_corpus)  # Uncomment and replace 'your_corpus' with your actual data

    # Transform the preprocessed input
    tfidf_features = tfidf_vectorizer.transform([preprocessed_input])

    # Make predictions using model1
    prediction = model1.predict(tfidf_features)[0]

    # Display the prediction result
    st.write(f"Prediction: {prediction}")
