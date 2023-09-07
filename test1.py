import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

# Download NLTK stopwords
nltk.download('stopwords')

# Load your pre-trained model (model1)
model1 = joblib.load("tolonglah.pkl")

# Create a Streamlit app
st.title("deployment test")

# Create a text input field
user_input = st.text_area("Enter some text:", "")

# Create a button to make predictions
if st.button("Make Prediction"):
    if user_input:
        # Make predictions using model1
        prediction = model1.predict(user_input)

        # Display the prediction result
        st.write(f"Prediction: {prediction}")
