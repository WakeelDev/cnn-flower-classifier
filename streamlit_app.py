import streamlit as st
import requests
from io import BytesIO
import joblib  # or torch/tensorflow/keras depending on your model
import time

# ===== CONFIGURATION =====
FILE_ID = "https://drive.google.com/file/d/1yu3dZ77n_rJShBRRcsg_kjMOKGJ67Sqj/view?usp=sharing"  # Replace with your actual file ID
MODEL_TYPE = "joblib"  # Change to "torch", "tensorflow", or "keras" if needed
CHUNK_SIZE = 32768  # Download chunk size (32KB)
# =========================

def safe_download_gdrive(file_id):
    """Robust Google Drive downloader with cookie handling"""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    try:
        # Initial request
        response = session.get(URL, params={'id': file_id}, stream=True)
        response.raise_for_status()
        
        # Check for security token in cookies
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # Confirm download if token exists
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
            response.raise_for_status()
        
        return response
        
    except requests.exceptions.RequestException as e:
        st.error(f"Download failed: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model with robust error handling and caching"""
    try:
        # Display download progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.subheader("🚚 Downloading model (this may take several minutes)...")
        
        # Start download
        response = safe_download_gdrive(FILE_ID)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        file_bytes = BytesIO()
        
        # Stream download with progress
        start_time = time.time()
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:  # Filter out keep-alive chunks
                file_bytes.write(chunk)
                downloaded += len(chunk)
                
                # Update progress every 0.5 seconds
                if (time.time() - start_time) > 0.5:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress)
                    mb_downloaded = downloaded / (1024 * 1024)
                    status_text.text(f"📥 Downloaded: {mb_downloaded:.1f}MB/{total_size/(1024*1024):.1f}MB")
                    start_time = time.time()
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text(f"✅ Download complete! Total size: {total_size/(1024*1024):.1f}MB")
        
        # Load model from memory
        file_bytes.seek(0)  # Reset pointer to start
        
        if MODEL_TYPE == "joblib":
            return joblib.load(file_bytes)
        elif MODEL_TYPE == "torch":
            import torch
            return torch.load(file_bytes)
        elif MODEL_TYPE == "tensorflow":
            import tensorflow as tf
            return tf.keras.models.load_model(file_bytes)
        elif MODEL_TYPE == "keras":
            from tensorflow import keras
            return keras.models.load_model(file_bytes)
        else:
            raise ValueError(f"Unsupported model type: {MODEL_TYPE}")
            
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# ===== MAIN APP =====
st.title("Flower Classification App")
st.subheader("Powered by Streamlit")

# Initialize session state
if 'model' not in st.session_state:
    with st.spinner("⚙️ Initializing model for the first time..."):
        st.session_state.model = load_model()

st.success("Model loaded successfully! Ready for predictions.")

# ===== ADD YOUR PREDICTION CODE BELOW =====
# Example:
# input = st.file_uploader("Upload flower image")
# if input:
#     prediction = st.session_state.model.predict(...)
#     st.write(f"Prediction: {prediction}")
