import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Load Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to include a single channel (grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Optionally, convert grayscale images to RGB by repeating the grayscale channel
x_train_rgb = np.repeat(x_train, 3, axis=-1)
x_test_rgb = np.repeat(x_test, 3, axis=-1)

# Step 2: Split the data into train and validation datasets (80% train, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(x_train_rgb, y_train, test_size=0.2, random_state=42)

# Define the base directory for the dataset
base_dir = 'fashion_mnist_data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# List of class names in Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create the necessary directories for train and validation data
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)

# Step 3: Function to save images to the appropriate class directory
def save_images_to_directory(images, labels, data_dir, class_names):
    for i, image in enumerate(images):
        class_name = class_names[labels[i]]
        class_dir = os.path.join(data_dir, class_name)
        image_path = os.path.join(class_dir, f'{i}.png')
        img = tf.keras.preprocessing.image.array_to_img(image)  # Convert array to image
        img.save(image_path)

# Step 4: Save the training and validation images into the respective directories
save_images_to_directory(x_train, y_train, train_dir, class_names)
save_images_to_directory(x_val, y_val, validation_dir, class_names)

# Step 5: Set up ImageDataGenerator for preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2,
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Step 6: Use ImageDataGenerator to load data
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(64, 64),  # Resize images to 64x64
    batch_size=32, 
    class_mode='categorical'  # For multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, 
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='categorical'
)

# Step 7: Build the CNN model for accessory classification
accessory_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  # Number of accessory classes
])

# Step 8: Compile the model
accessory_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 9: Train the model
history = accessory_model.fit(
    train_generator, 
    epochs=10, 
    validation_data=validation_generator
)

# Step 10: Save the model
try:
    accessory_model.save('accessory_model.h5')
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")
