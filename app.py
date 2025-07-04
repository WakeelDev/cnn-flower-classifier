import streamlit as st
import os
import urllib.request
import tensorflow as tf
import numpy as np
from PIL import Image

# Download model from Google Drive if not present
model_path = "flower_model.h5"
model_url = "https://drive.google.com/uc?id=1OrFphQW3SbpBCyXYIQot8BctkdK2_2kM"

if not os.path.exists(model_path):
    with st.spinner("⏳ Downloading model..."):
        urllib.request.urlretrieve(model_url, model_path)
        st.success("✅ Model downloaded!")

# Load model
model = tf.keras.models.load_model(model_path)

# Class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Streamlit UI
st.title("🌼 Flower Classifier")
st.write("Upload a flower image and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    prob = tf.nn.softmax(predictions[0])
    class_idx = np.argmax(prob)
    confidence = np.max(prob) * 100

    st.markdown(f"### 🌸 Prediction: **{class_names[class_idx]}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
