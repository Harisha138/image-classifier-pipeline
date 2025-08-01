import streamlit as st
import requests
from PIL import Image
import os

# --- IMPORTANT ---
# Make sure this URL is correct
MODAL_API_URL = "https://harisha1382004--image-classifier-app-fastapi-app.modal.run/predict" 

# --- Streamlit UI Configuration ---

st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("Cat vs. Dog Image Classifier")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")
st.write("This UI sends the image to a model hosted on Modal.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    # The parameter 'use_column_width' is deprecated. Using 'use_container_width' instead.
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    
    # When the user clicks the button, send the request
    if st.button('Predict'):
        with st.spinner('Sending image to the model...'):
            try:
                # Prepare the file to be sent in the request
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send the POST request to the Modal endpoint
                response = requests.post(MODAL_API_URL, files=files)
                
                if response.status_code == 200:
                    # Display the prediction
                    prediction = response.json()['prediction']
                    st.success(f"Prediction: **{prediction}**")
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the API: {e}")

