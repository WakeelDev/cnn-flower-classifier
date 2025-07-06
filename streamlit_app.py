import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import gdown

# Google Drive file ID
drive_url = "https://drive.google.com/uc?id=1yu3dZ77n_rJShBRRcsg_kjMOKGJ67Sqj"
model_path = "cnn_model.h5"

# Download the model if not already present
if not os.path.exists(model_path):
    st.info("Downloading model...")
    gdown.download(drive_url, model_path, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Set class names (same as used during training)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Streamlit UI
st.title("🌼 CNN Flower Classifier")
st.write("Upload an image of a flower to classify it.")

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = tf.keras.utils.img_to_array(image.resize((180, 180)))
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    predictions = model.predict(img_array)
    prob = tf.nn.softmax(predictions[0])
    predicted = class_names[np.argmax(prob)]
    confidence = np.max(prob) * 100

    st.markdown(f"### 🌸 Prediction: **{predicted}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
