import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
import pandas as pd

# Download NLTK stopwords and tokenizer data
nltk.download('punkt')
nltk.download('stopwords')

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
model1 = joblib.load("tolonglah.pkl")  # Replace with your model file path

# Load your CSV data into a DataFrame
data = pd.read_csv('Tweets.csv', encoding='latin1')

# Apply preprocessing to the 'text' column using .apply()
data['cleaned_data'] = data['text'].apply(preprocess_text)

# Recreate the TF-IDF vectorizer with the same parameters used during training
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
tfidf_features = tfidf_vectorizer.fit_transform(data['cleaned_data'])

# Recreate the BoW vectorizer with the same parameters used during training
bow_vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2))
bow_features = bow_vectorizer.fit_transform(data['cleaned_data'])

# Create a Streamlit app
st.title("Deployment Test")

# Create a text input field
user_input = st.text_area("Enter some text:", "")

# Create a POS vectorizer
pos_vectorizer = CountVectorizer()  # Moved this line here

# Create a button to make predictions
if st.button("Make Prediction"):
    if user_input:
        # Preprocess the user input for TF-IDF and BoW features
        preprocessed_input = preprocess_text(user_input)

        # Transform the preprocessed input using the same TF-IDF vectorizer
        tfidf_input = tfidf_vectorizer.transform([preprocessed_input])

        # Transform the preprocessed input using the same BoW vectorizer
        bow_input = bow_vectorizer.transform([preprocessed_input])

        # Combine the TF-IDF and BoW features
        combined_input = tfidf_input + bow_input

        # Process the user input for POS features
        tokens = nltk.word_tokenize(user_input)
        pos_tags = nltk.pos_tag(tokens)
        pos_tags_str = ' '.join([tag for _, tag in pos_tags])

        # Transform the preprocessed input using the POS vectorizer
        pos_input = pos_vectorizer.transform([pos_tags_str])

        # Combine the TF-IDF, BoW, and POS features for prediction
        all_features_input = hstack([combined_input, pos_input])

        # Make predictions using model1
        prediction = model1.predict(all_features_input)

        # Display the prediction result
        st.write(f"Prediction: {prediction}")
