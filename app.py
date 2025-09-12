import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications import MobileNetV2

# === CONFIGURATION ===
MODEL_NAME = "mobilenetv2_model.keras"
CLASS_FILE = "class_names.json"
IMG_SIZE = 180

# === Load Model & Classes ===

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_NAME,
        custom_objects={"MobileNetV2": MobileNetV2}
    )


model = load_model()

with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

# === Streamlit UI ===
st.title("🌸 Flower Classifier (MobileNetV2)")

uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocessing (same as training: [-1, 1])
    img_array = (img_array / 127.5) - 1.0  

    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = float(np.max(preds))

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
