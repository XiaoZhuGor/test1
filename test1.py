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
    def remove_stopwords(text):
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
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
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
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
        substitute = re.sub(r'[^a-zA-Z]', ' ', text)
        return substitute

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
tfidf_vectorizer = TfidfVectorizer(max_features=10016, ngram_range=(1, 2))
tfidf_features = tfidf_vectorizer.fit_transform(data['cleaned_data'])

# Recreate the BoW vectorizer with the same parameters used during training
bow_vectorizer = CountVectorizer(max_features=10016, ngram_range=(1, 2))
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
        
        # Display the preprocessed input text
        st.write(f"Preprocessed text: {preprocessed_input}")
        
        # Display the prediction result
        st.write(f"Prediction: {prediction[0]}")
