import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np



# Load your pre-trained model (model1)
model1 = joblib.load("tolonglah.pkl")

# Create a Streamlit app
st.title("deployment test")

# Create a text input field
user_input = st.text_area("Enter some text:", "")

# Create a button to make predictions
if st.button("Make Prediction"):
    if user_input:
         # Initialize the CountVectorizer
        vectorizer = CountVectorizer(binary=True)

        # Fit the vectorizer on the training data and transform user input
        user_input_transformed = vectorizer.transform([user_input])

        # Make predictions using model1
        prediction = model1.predict(user_input_transformed)

        # Display the prediction result
        st.write(f"Prediction: {prediction[0]}")
