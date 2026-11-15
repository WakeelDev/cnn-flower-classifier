import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from groq import Groq

# === CONFIGURATION ===
MODEL_NAME = "mobilenetv2_model.keras"
CLASS_FILE = "class_names.json"
IMG_SIZE = 180

# === PAGE CONFIG ===
st.set_page_config(page_title="AI Botanical Agent", page_icon="🌸", layout="wide")

# === SIDEBAR: API CONFIGURATION ===
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    st.write("Enter your free Groq API key to power the botanical agent.")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get a free key at console.groq.com")
    st.divider()
    st.info("💡 **Project Tip:** Your MobileNetV2 CNN identifies flowers with ~93% accuracy, and the Groq AI agent provides expert botanical care instructions on the fly!")

# === LOAD MODEL & CLASSES ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_NAME,
        custom_objects={"MobileNetV2": MobileNetV2}
    )

try:
    model = load_model()
    with open(CLASS_FILE, "r") as f:
        class_names = json.load(f)
except Exception as e:
    st.error(f"⚠️ Model files not found! Please ensure `{MODEL_NAME}` and `{CLASS_FILE}` are in your directory.")
    st.stop()

# === STREAMLIT UI ===
st.title("🌸 AI Flower Classifier & Botanical Agent")
st.write("Upload a flower image (`daisy`, `dandelion`, `rose`, or `sunflower`). Our computer vision CNN will classify it, and our AI Agent will act as your personal horticulturist!")

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Flower", use_container_width=True)

    with col2:
        with st.spinner("Analyzing flower features with CNN..."):
            resized_img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.utils.img_to_array(resized_img)
            img_array = np.expand_dims(img_array, axis=0)

            # Preprocessing (matches training: [-1, 1])
            img_array = (img_array / 127.5) - 1.0  

            preds = model.predict(img_array)
            confidence = float(np.max(preds))
            pred_index = np.argmax(preds)
            
            # === UNCERTAINTY GRADIENT & 50% THRESHOLD LOGIC ===
            HARD_THRESHOLD = 0.50  # 50% cutoff for unknown species
            
            if confidence < HARD_THRESHOLD:
                pred_class = "Unknown Species"
                is_unknown = True
            else:
                pred_class = class_names[pred_index]
                is_unknown = False

        st.subheader("📊 CNN Classification Results")
        if is_unknown:
            st.warning("⚠️ Very low confidence (<50%). This image appears to be an unrecognized flower or non-flower object outside our trained categories.")
            st.metric(label="Status", value="Unknown Species")
        else:
            st.metric(label="Predicted Species", value=pred_class.capitalize())
            st.metric(label="Confidence Score", value=f"{confidence * 100:.1f}%")
            
            # Uncertainty Gradient Tiers
            if confidence >= 0.75:
                st.success("🟢 **High Certainty:** The model is highly confident in this classification.")
            else:
                st.info("🟡 **Moderate Certainty:** The model detected features of this flower, but background noise or complex lighting reduced its certainty.")

        # Top predictions breakdown
        top_3_indices = np.argsort(preds[0])[-3:][::-1]
        with st.expander("View Top Predictions Breakdown"):
            for idx in top_3_indices:
                st.text(f"• {class_names[idx].capitalize()}: {preds[0][idx]*100:.1f}%")

    # === PHASE 2: AGENTIC LLM CARE GUIDE ===
    st.divider()
    if is_unknown:
        st.subheader("🌱 AI Botanical Agent Analysis")
    else:
        st.subheader(f"🌱 AI Botanical Agent Care Guide for: {pred_class.capitalize()}")

    if not groq_api_key:
        st.warning("🔑 Please enter your **Groq API Key** in the sidebar to generate the AI agent's expert care report and enable interactive botanical chat.")
    else:
        try:
            client = Groq(api_key=groq_api_key)
            
            with st.spinner("Consulting the AI botanical database..."):
                if is_unknown:
                    prompt = (
                        f"You are an expert master horticulturist and botanical agent. "
                        f"A user scanned an image, but our computer vision CNN model flagged it as an unknown species "
                        f"with a low confidence of {confidence*100:.1f}% because it didn't match our trained classes (daisy, dandelion, rose, sunflower). "
                        f"Politely explain to the user that this appears to be a flower or object outside of the system's trained dataset, "
                        f"acknowledge that the model correctly flagged its uncertainty, and invite them to upload one of the 4 supported flowers."
                    )
                else:
                    prompt = (
                        f"You are an expert master horticulturist and botanical agent. "
                        f"A user has scanned a flower and our CNN model identified it as a '{pred_class}' with {confidence*100:.1f}% confidence. "
                        f"Provide a structured, engaging care report for this flower including: "
                        f"1. Optimal Watering Schedule 💧\n"
                        f"2. Sunlight Requirements ☀️\n"
                        f"3. Ideal Soil & Fertilizer Type 🌱\n"
                        f"4. A fun, lesser-known botanical fact 🧠"
                    )
                
                chat_completion = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                care_report = chat_completion.choices[0].message.content

            st.success("Report generated successfully!")
            st.markdown(care_report)

            # === INTERACTIVE FOLLOW-UP CHAT WITH THE AGENT ===
            st.divider()
            st.subheader(f"💬 Chat with your Botanical Expert")
            user_question = st.text_input(f"Ask any question about this plant:")
            
            if user_question:
                with st.spinner("Agent is thinking..."):
                    followup_prompt = (
                        f"You are a botanical agent. The user scanned an image classified as '{pred_class}' (Confidence: {confidence*100:.1f}%). "
                        f"The user is asking: '{user_question}'. Give a helpful, friendly, and concise expert answer."
                    )
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": followup_prompt}],
                        temperature=0.7,
                    )
                    st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"⚠️ Error communicating with Groq API: {e}")