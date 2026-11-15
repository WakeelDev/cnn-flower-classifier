import json
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import streamlit as st
from groq import Groq

# Page Configuration
st.set_page_config(
    page_title="Botanical AI & Flower Classifier", page_icon="🌸", layout="centered"
)

# Robust absolute path resolution for cloud deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mobilenetv2_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

# Explicitly initialize global variables to prevent any NameError scope issues
model = None
class_names = None
load_error = None


@st.cache_resource
def load_model_and_classes():
  try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
      return None, None, "Model or class files missing from directory."
    
    with open(CLASS_PATH, "r") as f:
      loaded_classes = json.load(f)
      
    num_classes = len(loaded_classes)

    # 1. Rebuild the Sequential architecture matching local training
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights=None
    )
    
    net = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    # 2. Build model and load weights
    net.build((None, 224, 224, 3))
    net.load_weights(MODEL_PATH)
    
    return net, loaded_classes, None
  except Exception as e:
    return None, None, str(e)


# Unpack loader results at the top global level
model, class_names, load_error = load_model_and_classes()

# Sidebar - Groq API Key Configuration
st.sidebar.title("🌿 Agent Configuration")
st.sidebar.markdown(
    "Enter your free Groq API key to power the botanical assistant."
)
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Project Tip:** Your MobileNetV2 CNN identifies flowers with ~93%"
    " accuracy, and the Groq AI agent provides expert botanical care"
    " instructions on the fly!"
)

# Main Application Layout
st.title("🌸 Botanical AI: Flower Classifier & Assistant")
st.write(
    "Upload an image of a flower (**daisy, dandelion, rose, or sunflower**) to"
    " classify it and consult our expert botanical assistant!"
)

if model is None or class_names is None:
  st.error(
      f"⚠️ **Model loading failed!** Details: `{load_error}`\n\n"
      f"Ensured files exist in: `{BASE_DIR}`"
  )
else:
  uploaded_file = st.file_uploader(
      "Choose a flower image...", type=["jpg", "jpeg", "png"]
  )

  if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Flower Image", use_column_width=True)

    # Preprocess image for MobileNetV2 (224x224)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(float))

    # Run Prediction
    with st.spinner("Analyzing flower features..."):
      preds = model.predict(img_array)
      confidence = float(np.max(preds))
      predicted_class_idx = int(np.argmax(preds))

      # Handle list or dict format for class_names
      if isinstance(class_names, list):
        predicted_class = class_names[predicted_class_idx]
      elif isinstance(class_names, dict):
        predicted_class = class_names.get(
            str(predicted_class_idx), class_names.get(predicted_class_idx)
        )
      else:
        predicted_class = str(predicted_class_idx)

    # 50% Confidence Threshold & Uncertainty Gradient UI
    CONFIDENCE_THRESHOLD = 0.50

    st.markdown("---")
    st.subheader("📊 Classification Results")

    if confidence < CONFIDENCE_THRESHOLD:
      st.warning(
          f"⚠️ **Uncertain Classification** (Confidence:"
          f" {confidence * 100:.2f}%)"
      )
      st.info(
          "The model's confidence is below the 50% threshold. This image may"
          " not belong to the supported classes or is out-of-distribution."
      )
      display_class = "Unknown / Out-of-Distribution"
    else:
      # Uncertainty Tiers
      if confidence >= 0.85:
        st.success(f"🟢 **High Confidence Tier** ({confidence * 100:.2f}%)")
      elif confidence >= 0.70:
        st.info(f"🟡 **Moderate Confidence Tier** ({confidence * 100:.2f}%)")
      else:
        st.warning(f"🟠 **Low Confidence Tier** ({confidence * 100:.2f}%)")

      display_class = predicted_class
      st.metric(
          label="Predicted Flower",
          value=display_class.capitalize(),
          delta=f"{confidence * 100:.1f}% Confidence",
      )

    # Groq Botanical Assistant Section
    st.markdown("---")
    st.subheader("🤖 Groq Botanical Assistant")

    if not groq_api_key:
      st.warning(
          "Please enter your Groq API Key in the sidebar to chat with the"
          " botanical assistant."
      )
    else:
      client = Groq(api_key=groq_api_key)
      default_prompt = (
          f"Provide expert care instructions, ideal growing conditions, soil"
          f" requirements, and interesting botanical facts for"
          f" {display_class}."
      )
      user_prompt = st.text_area(
          "Ask the botanical assistant anything about this flower:",
          value=default_prompt,
      )

      if st.button("Generate Botanical Advice"):
        with st.spinner("Consulting AI botanist..."):
          try:
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert botanist and AI assistant"
                            " providing clear, accurate, and engaging plant"
                            " care information."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            ai_response = chat_completion.choices[0].message.content
            st.markdown("### 📝 Expert Botanical Insights:")
            st.write(ai_response)
          except Exception as e:
            st.error(f"Error communicating with Groq API: {e}")