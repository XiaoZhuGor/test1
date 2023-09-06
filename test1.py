import streamlit as st
import joblib

# Load the saved models from the .pkl file
model_filename = "best_models.pkl"
loaded_models = joblib.load(model_filename)

# Define the Streamlit app
st.title("Streamlit App with Multiple Models")

# Create a dropdown to select the model
selected_model = st.selectbox("Select a Model", list(loaded_models.keys()))

# Create an input field for user input (you can modify this based on your model's input requirements)
user_input = st.text_input("Enter some text:", "")

# Function to make predictions using the selected model
def make_prediction(input_text, selected_model):
    model = loaded_models[selected_model]
    # Replace this with your actual prediction code for the selected model
    prediction = model.predict([input_text])[0]
    return prediction

# Make predictions when the user clicks a button
if st.button("Make Prediction"):
    if user_input:
        prediction = make_prediction(user_input, selected_model)
        st.write(f"Prediction using {selected_model}: {prediction}")

# Note: You should replace the prediction logic with your specific model's prediction code.
