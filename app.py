import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# Class Names
# -------------------------------
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


# -------------------------------
# Load or Train Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("fashion_model.h5")
        return model
    except:
        st.warning("No saved model found. Training a new model...")

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = (
            fashion_mnist.load_data()
        )

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.fit(train_images, train_labels, epochs=5)

        model.save("fashion_model.h5")
        return model


# Load model
model = load_model()

# Add Softmax Probability Layer
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("👕 Fashion MNIST Clothing Classifier")
st.write(
    "Upload an image (28×28 grayscale) and the model will predict the clothing type."
)

st.sidebar.header("Options")
show_probs = st.sidebar.checkbox("Show probability chart", True)

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Load Image
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img = img.resize((28, 28))

    st.image(img, caption="Uploaded Image", width=200)

    # Convert Image to Array
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = probability_model.predict(img_array)
    predicted_label = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Display Result
    st.subheader("Prediction Result")
    st.success(f"Predicted Class: **{class_names[predicted_label]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    # Probability Chart
    if show_probs:
        st.subheader("Prediction Probabilities")

        fig, ax = plt.subplots()
        ax.bar(class_names, predictions[0])
        plt.xticks(rotation=45)
        st.pyplot(fig)

else:
    st.info("Please upload an image to classify.")
