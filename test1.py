import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
import pandas as pd

# Create a function to preprocess text
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

# Load your pre-trained model (model1)
model1 = joblib.load("tolonglah.pkl")

# Load your CSV data into a DataFrame
data = pd.read_csv('Tweets.csv', encoding='latin1')
# Apply preprocessing to the 'text' column using .apply()
data['cleaned_data'] = data['text'].apply(preprocess_text)

# Create a Streamlit app
st.title("deployment test")

# Create a text input field
user_input = st.text_area("Enter some text:", "")


# Recreate the TF-IDF vectorizer with the same parameters used during training
tfidf_vectorizer = TfidfVectorizer(max_features=40064, ngram_range=(1, 2))
tfidf_features = tfidf_vectorizer.fit_transform(data['cleaned_data'])

# Recreate the BoW vectorizer with the same parameters used during training
bow_vectorizer = CountVectorizer(max_features=40064, ngram_range=(1, 2))
bow_features = bow_vectorizer.fit_transform(data['cleaned_data'])

if st.button("Make Prediction"):
    if user_input:
        # Preprocess the user input
        preprocessed_input = preprocess_text(user_input)

        # Transform the preprocessed input using the same TF-IDF vectorizer
        tfidf_input = tfidf_vectorizer.transform([preprocessed_input])

        # Transform the preprocessed input using the same BoW vectorizer
        bow_input = bow_vectorizer.transform([preprocessed_input])

        # Combine the TF-IDF and BoW features (if needed)
        # Example of concatenation (modify as needed):
        combined_input = hstack([tfidf_input, bow_input])

        # Make predictions using model1
        prediction = model1.predict(combined_input)

        # Display the prediction result
        st.write(f"Prediction: {prediction[0]}")
