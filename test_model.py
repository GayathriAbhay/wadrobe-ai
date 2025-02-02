def prepare_image(image):
    image = image.resize((28, 28))  # Resize to 28x28
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for CNN input
    
    # Debugging - Print the processed image
    st.image(image.reshape(28, 28), caption="Processed Image", use_column_width=True)
    
    return image
