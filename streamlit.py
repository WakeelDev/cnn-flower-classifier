import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# === Load Model and Class Names ===
MODEL_NAME = "mobilenetv2_model.keras"   # <-- switched to .keras format
CLASS_FILE = "class_names.json"

@st.cache_resource
def load_model():
    # Load without compile since we only need inference
    return tf.keras.models.load_model(MODEL_NAME, compile=False)

with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

model = load_model()

# === Streamlit UI ===
st.title("🌸 Flower Classifier (MobileNetV2)")

uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load & preprocess image
    img = Image.open(uploaded_file).convert("RGB").resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # SAME preprocessing as during training ([-1, 1] scaling for MobileNetV2)
    img_array = (img_array / 127.5) - 1.0  

    # Predict
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions, axis=1)[0]
    pred_class = class_names[pred_index]
    confidence = float(np.max(predictions))

    # Show results
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
