import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Function to render the UI HTML content
def render_ui():
    # Read the HTML content from the ui.html file
    with open("ui.html", "r") as f:
        html_content = f.read()

    # Display the HTML content within Streamlit
    st.markdown(html_content, unsafe_allow_html=True)

# Load trained AI models
outfit_model = load_model('outfit_model.h5')
accessory_model = load_model('accessory_model.h5')

# Define class labels for the predictions
outfit_class_labels = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

accessory_class_labels = ['Shoes', 'Bag', 'Jewelry', 'Hat']  # Modify according to your dataset

# Function to preprocess the uploaded outfit image
def prepare_outfit_image(image):
    image = image.resize((28, 28))  # Resize to 28x28
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for CNN input
    return image

# Function to preprocess the accessory image
def prepare_accessory_image(image):
    image = image.resize((64, 64))  # Resize to 64x64 for accessories
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 64, 64, 3)  # Reshape for CNN input (RGB channels)
    return image

# Function to make outfit predictions
def predict_outfit(image):
    image = prepare_outfit_image(image)
    prediction = outfit_model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability
    return outfit_class_labels[predicted_class]

# Function to make accessory predictions
def recommend_accessories(image):
    image = prepare_accessory_image(image)
    prediction = accessory_model.predict(image)
    predicted_accessory = accessory_class_labels[np.argmax(prediction)]  # Accessory prediction
    return predicted_accessory

# Streamlit UI
def main():
    # Step 1: Display the UI HTML when the app starts
    render_ui()

    # Step 2: Add file uploader for the image and process predictions after interaction
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Predict and display the outfit category
        predicted_outfit = predict_outfit(img)
        st.success(f"Predicted Outfit Category: **{predicted_outfit}**")

        # Predict and display the recommended accessory
        recommended_accessory = recommend_accessories(img)
        st.success(f"Suggested Accessory: **{recommended_accessory}**")

if __name__ == '__main__':
    main()
