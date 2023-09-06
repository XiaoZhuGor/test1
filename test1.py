import streamlit as st
import joblib
import numpy as np  # Added this import

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

    # Create an input field for user input (you can modify this based on your model's input requirements)
    user_input = st.text_input("Enter some text:", "")

    # Function to make predictions using the selected model
    def make_prediction(input_text, selected_model):
        model = loaded_models[selected_model]

        # Preprocess the input text using the same vectorizer used during training
        # Replace 'vectorizer' with your actual vectorizer
        # Example: vectorizer = your_actual_vectorizer
        input_features = your_actual_vectorizer.transform([input_text])

        # Convert input_features to a NumPy array and reshape it
        input_features = np.array(input_features)
        input_features = input_features.reshape(1, -1)

        # Replace this with your actual prediction code for the selected model
        prediction = model.predict(input_features)[0]
        return prediction

    # Make predictions when the user clicks a button
    if st.button("Make Prediction"):
        if user_input:
            prediction = make_prediction(user_input, selected_model)
            st.write(f"Prediction using {selected_model}: {prediction}")

# Note: You should replace 'your_actual_vectorizer' with the actual variable name of your vectorizer.
# You may also need to preprocess the input text further to match the preprocessing done during training.
