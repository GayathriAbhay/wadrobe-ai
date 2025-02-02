import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define the accessory classes: Sandal (5), Sneaker (8), and Bag (9)
accessory_classes = [5, 8, 9]

# Filter the training data for accessory classes
train_mask = np.isin(y_train, accessory_classes)
x_train_accessories = x_train[train_mask]
y_train_accessories = y_train[train_mask]

# Filter the test data for accessory classes
test_mask = np.isin(y_test, accessory_classes)
x_test_accessories = x_test[test_mask]
y_test_accessories = y_test[test_mask]

# Display some of the images
fig, axes = plt.subplots(1, 5, figsize=(15, 15))
for i, ax in enumerate(axes):
    ax.imshow(x_train_accessories[i], cmap='gray')
    ax.set_title(f"Label: {y_train_accessories[i]}")
    ax.axis('off')

plt.show()
