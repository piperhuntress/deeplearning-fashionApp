import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(_, _), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for readable filenames
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Create directory and save first 5 images
save_dir = "fashion_mnist_test_images"
os.makedirs(save_dir, exist_ok=True)

for i in range(5):
    img_array = test_images[i]
    label = test_labels[i]
    img = Image.fromarray(img_array.astype(np.uint8))
    filename = os.path.join(
        save_dir, f"test_{i}_label_{label}_{class_names[label]}.png"
    )
    img.save(filename)
    print(f"Saved: {filename}")
