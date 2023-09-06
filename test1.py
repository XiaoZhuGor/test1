
import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Create a file uploader to allow users to upload the model file
model_file = st.file_uploader("Upload model file (.pkl)", type=["pkl"])

# Check if a model file was uploaded
if model_file:
    # Load the model from the uploaded file
    loaded_models = joblib.load(model_file)

    # Define the Streamlit app
    st.title("Streamlit App with Multiple Models")

    # Create a dropdown to select the model
    selected_model = st.selectbox("Select a Model", list(loaded_models.keys()))

    # Create an input field for user input
    user_input = st.text_input("Enter some text:", "")

    # Load the vectorizer used during training (replace 'vectorizer' with your actual vectorizer)
    vectorizer = joblib.load("your_vectorizer.pkl")  # Replace with the path to your vectorizer

    # Function to make predictions using the selected model
    def make_prediction(input_text, selected_model):
        model = loaded_models[selected_model]

        # Vectorize the input text using the same vectorizer used during training
        input_features = vectorizer.transform([input_text])

        prediction = model.predict(input_features)[0]
        return prediction

    # Make predictions when the user clicks a button
    if st.button("Make Prediction"):
        if user_input:
            prediction = make_prediction(user_input, selected_model)
            st.write(f"Prediction using {selected_model}: {prediction}")
