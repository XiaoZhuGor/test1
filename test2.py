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
# Define preprocessing functions
def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
    return text

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_username(text):
    return re.sub('@[^\s]+', '', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def decontraction(text):
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"won\'t've", " will not have", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"don\'t", " do not", text)
    
    text = re.sub(r"can\'t've", " can not have", text)
    text = re.sub(r"ma\'am", " madam", text)
    text = re.sub(r"let\'s", " let us", text)
    text = re.sub(r"ain\'t", " am not", text)
    text = re.sub(r"shan\'t", " shall not", text)
    text = re.sub(r"sha\n't", " shall not", text)
    text = re.sub(r"o\'clock", " of the clock", text)
    text = re.sub(r"y\'all", " you all", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n\'t've", " not have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll've", " will have", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    return text  

def separate_alphanumeric(text):
    words = text
    words = re.findall(r"[^\W\d_]+|\d+", words)
    return " ".join(words)

def cont_rep_char(text):
    tchr = text.group(0) 
    
    if len(tchr) > 1:
        return tchr[0:2] 

def unique_char(rep, text):
    substitute = re.sub(r'(\w)\1+', rep, text)
    return substitute

def char(text):
    substitute = re.sub(r'[^a-zA-Z]',' ',text)
    return substitute

# Function to preprocess input text
def preprocess_input_text(input_text):
    text = input_text
    text = remove_username(text)
    text = remove_url(text)
    text = remove_emoji(text)
    text = decontraction(text)
    text = separate_alphanumeric(text)
    text = unique_char(cont_rep_char, text)
    text = char(text)
    text = text.lower()
    text = remove_stopwords(text)
    return text

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

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), max_df=0.5)
tfidf_features = tfidf_vectorizer.fit_transform(data['cleaned_data'])

tfidf_vectorizer2 = TfidfVectorizer(max_features=2500, ngram_range=(1, 3), max_df=0.25)
tfidf_features2 = tfidf_vectorizer2.fit_transform(data['cleaned_data'])

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

            # Map the predicted label to sentiment class
            predicted_sentiment_label = sentiment_mapping.get(prediction[0], 'unknown')
            
            # Make predictions using model2
            prediction = model2.predict(tfidf_input)

            # Display the preprocessed input
            st.write(f"Preprocessed text: {preprocessed_input}")

            # Display the prediction result
            st.write(f"Prediction: {prediction}")
