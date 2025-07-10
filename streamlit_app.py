import streamlit as st
import requests
from io import BytesIO
import time
import numpy as np
from PIL import Image
import os
import tempfile
import h5py

# Set page configuration
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ===== CUSTOM STYLES =====
st.markdown("""
<style>
.header {
    color: #e75480;
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 20px;
}
.troubleshoot {
    background-color: #fff0f5;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}
.step {
    font-weight: bold;
    color: #e75480;
}
.download-section {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
}
.prediction-card {
    background-color: #f0f8ff;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}
.error-section {
    background-color: #ffebee;
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
    border: 1px solid #ffcdd2;
}
</style>
""", unsafe_allow_html=True)

# ===== CONFIGURATION =====
# Your actual Google Drive file ID
FILE_ID = "1yu3dZ77n_rJShBRRcsg_kjMOKGJ67Sqj"

# Display file info
st.markdown('<h1 class="header">🌸 Flower Classification App</h1>', unsafe_allow_html=True)
st.markdown("---")
st.info(f"**Using model from Google Drive file ID:** `{FILE_ID}`")
st.markdown(f"**Download link:** [https://drive.google.com/uc?export=download&id={FILE_ID}](https://drive.google.com/uc?export=download&id={FILE_ID})")

# ===== ROBUST DOWNLOAD FUNCTION =====
def download_model():
    """Download model with proper Google Drive handling"""
    try:
        URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
        # First request to get confirmation token
        response = session.get(URL, stream=True, timeout=60)
        response.raise_for_status()
        
        # Find confirmation token in cookies
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # Second request with confirmation token
        if token:
            params = {'id': FILE_ID, 'confirm': token}
            response = session.get(URL, params=params, stream=True, timeout=60)
            response.raise_for_status()
        
        # Check if we actually got a file
        if 'content-length' not in response.headers:
            st.error("❌ No content length in response headers")
            return None
            
        return response.content  # Return the content directly
        
    except requests.exceptions.HTTPError as e:
        st.error(f"🚨 HTTP Error: {e.response.status_code if e.response else 'No response'}")
        if e.response:
            if e.response.status_code == 404:
                st.error(f"File not found. Verify your file ID: {FILE_ID}")
            elif e.response.status_code == 403:
                st.error("Permission denied. Ensure sharing is set to 'Anyone with the link'")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"🚨 Unexpected download error: {str(e)}")
        return None

# ===== MODEL LOADING =====
def load_model(model_bytes):
    """Load model from bytes with format detection"""
    try:
        # Try loading as Keras model
        try:
            st.info("Attempting to load as Keras model...")
            from tensorflow.keras.models import load_model as load_keras_model
            
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
                tmp_file.write(model_bytes)
                tmp_file_path = tmp_file.name
            
            model = load_keras_model(tmp_file_path)
            os.unlink(tmp_file_path)  # Delete temp file
            st.success("✅ Successfully loaded as Keras model!")
            return model
        except Exception as keras_error:
            st.warning(f"⚠️ Keras loading failed: {str(keras_error)[:100]}...")
        
        # Try loading as joblib model
        try:
            st.info("Attempting to load as joblib model...")
            import joblib
            model = joblib.load(BytesIO(model_bytes))
            st.success("✅ Successfully loaded as joblib model!")
            return model
        except Exception as joblib_error:
            st.warning(f"⚠️ Joblib loading failed: {str(joblib_error)[:100]}...")
        
        # Try loading as PyTorch model
        try:
            st.info("Attempting to load as PyTorch model...")
            import torch
            model = torch.load(BytesIO(model_bytes))
            st.success("✅ Successfully loaded as PyTorch model!")
            return model
        except Exception as torch_error:
            st.warning(f"⚠️ PyTorch loading failed: {str(torch_error)[:100]}...")
        
        # If all fail
        st.error("❌ Failed to load model with any supported format")
        st.markdown('<div class="error-section">', unsafe_allow_html=True)
        st.subheader("🛠️ Debug Information")
        st.markdown("""
        **Possible Solutions:**
        1. Verify the model file format:
           - Keras models should be .h5 or .keras files
           - Joblib models should be .pkl or .joblib files
           - PyTorch models should be .pt or .pth files
        2. Check the file signature:
           - HDF5 files start with `\x89HDF`
           - Joblib files start with `\\x80\\x04\\x95`
        3. Ensure all dependencies are installed:
           ```python
           pip install tensorflow joblib torch
           ```
        """)
        
        # Display file signature
        if len(model_bytes) > 8:
            st.code(f"File signature: {model_bytes[:8]}", language="python")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return None
            
    except Exception as e:
        st.error(f"❌ Critical error during model loading: {str(e)}")
        return None

# ===== MODEL DOWNLOAD AND LOADING PROCESS =====
def download_and_load_model():
    """Handle the complete download and loading process"""
    try:
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.subheader("📥 Downloading Model")
        
        # Start download
        model_bytes = download_model()
        if model_bytes is None:
            return None
            
        # Display download info
        total_size = len(model_bytes)
        total_mb = total_size / (1024 * 1024)
        st.success(f"✅ Download complete! Size: {total_mb:.1f}MB")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load model
        with st.spinner("🔍 Detecting model format..."):
            model = load_model(model_bytes)
            
        if model:
            st.success("✅ Model loaded successfully!")
            return model
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error in download/load process: {str(e)}")
        return None

# ===== IMAGE PROCESSING =====
def preprocess_image(uploaded_file):
    """Convert uploaded file to model input format with error handling"""
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Basic preprocessing - MODIFY FOR YOUR MODEL
        image = image.resize((224, 224))  # Standard size for many models
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
        
    except Exception as e:
        st.error(f"❌ Image processing error: {str(e)}")
        return None

# ===== MAIN APP =====
# Initialize model
if 'model' not in st.session_state:
    with st.spinner("⚙️ Initializing model..."):
        model = download_and_load_model()
        if model:
            st.session_state.model = model
        else:
            st.error("❌ Model failed to load. Please check the errors above.")
            st.markdown("""
            <div class="troubleshoot">
            <h3>🔧 Advanced Troubleshooting</h3>
            <ol>
                <li><span class="step">Test the download link</span> - <a href="https://drive.google.com/uc?export=download&id={FILE_ID}" target="_blank">Click here</a> to test download in browser</li>
                <li><span class="step">Check file format</span> - The file should be a Keras (.h5), joblib, or PyTorch model</li>
                <li><span class="step">Library compatibility</span> - Install required packages: 
                    <code>pip install tensorflow joblib torch</code>
                </li>
                <li><span class="step">Local testing</span> - Try loading the model in a local Python environment</li>
                <li><span class="step">File inspection</span> - Check the first few bytes of the file for format signature</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

st.markdown("---")
st.subheader("🌼 Flower Classification")

# File uploader
uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Process image
    processed_image = preprocess_image(uploaded_file)
    
    if processed_image is not None:
        # Make prediction
        with st.spinner("🔍 Analyzing flower..."):
            try:
                # Try different prediction methods based on model type
                model = st.session_state.model
                
                # If Keras model
                if hasattr(model, "predict"):
                    prediction = model.predict(processed_image)
                    class_idx = np.argmax(prediction[0])
                    confidence = prediction[0][class_idx] * 100
                
                # If scikit-learn model
                elif hasattr(model, "predict_proba"):
                    # Reshape for sklearn models
                    processed_image = processed_image.reshape(1, -1)
                    prediction = model.predict_proba(processed_image)
                    class_idx = np.argmax(prediction[0])
                    confidence = prediction[0][class_idx] * 100
                
                # If PyTorch model
                elif hasattr(model, "forward"):
                    import torch
                    input_tensor = torch.from_numpy(processed_image).float()
                    with torch.no_grad():
                        output = model(input_tensor)
                    prediction = torch.nn.functional.softmax(output[0], dim=0)
                    class_idx = torch.argmax(prediction).item()
                    confidence = prediction[class_idx].item() * 100
                
                else:
                    st.error("❌ Unsupported model type for prediction")
                    st.stop()
                
                # Sample flower classes - REPLACE WITH YOUR ACTUAL CLASSES
                FLOWER_CLASSES = [
                    "Rose", "Tulip", "Daisy", "Sunflower", "Lily",
                    "Orchid", "Peony", "Hydrangea", "Daffodil", "Carnation"
                ]
                
                # Create prediction card
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.subheader(f"Prediction: **{FLOWER_CLASSES[class_idx]}**")
                
                # Confidence meter
                st.metric("Confidence", f"{confidence:.2f}%")
                
                # Confidence bar
                st.progress(int(confidence))
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Ensure your model expects the same input format used in preprocessing")

# Add footer
st.markdown("---")
st.markdown("### Requirements")
st.code("pip install streamlit requests numpy Pillow tensorflow joblib torch", language="bash")

st.markdown("### Troubleshooting Tips")
st.markdown("""
1. **Install all dependencies** - The app requires TensorFlow for Keras models
2. **Verify model format** - Your model is likely a Keras .h5 file
3. **Test locally** - Try loading the model with:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('your_model.h5')
