import streamlit as st
import joblib

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

    # Function to make predictions using the selected model
    def make_prediction(input_text, selected_model):
        model = loaded_models[selected_model]
        prediction = model.predict([input_text])[0]
        return prediction

    # Make predictions when the user clicks a button
    if st.button("Make Prediction"):
        if user_input:
            prediction = make_prediction(user_input, selected_model)
            st.write(f"Prediction using {selected_model}: {prediction}")
