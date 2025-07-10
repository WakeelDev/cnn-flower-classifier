import streamlit as st
import sys
import subprocess
import os
from io import BytesIO
import time
import requests

# Set page configuration
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== DEPENDENCY INSTALLATION =====
def install_missing_package(package_name):
    """Install missing packages with user feedback"""
    with st.spinner(f"Installing {package_name}..."):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success(f"Successfully installed {package_name}")
            st.rerun()  # Restart app to load new dependencies
        else:
            st.error(f"Failed to install {package_name}:")
            st.code(result.stderr)

# Check and install required packages
REQUIRED_PACKAGES = [
    ("requests", "requests"),
    ("joblib", "joblib"),
    ("PIL", "Pillow"),  # For image processing
    ("numpy", "numpy")   # For numerical operations
]

for import_name, package_name in REQUIRED_PACKAGES:
    try:
        __import__(import_name)
    except ImportError:
        st.warning(f"⚠️ Required package '{package_name}' is missing")
        if st.button(f"Install {package_name}", key=f"install_{package_name}"):
            install_missing_package(package_name)
        st.stop()

# Now safely import the required packages
import joblib
import numpy as np
from PIL import Image
import requests

# ===== CONFIGURATION =====
FILE_ID = "1-8vIDkq0z7wXqJvQdZ7QYd7q8w7XqJvQ"  # Example ID - REPLACE WITH YOURS
MODEL_TYPE = "joblib"  # Change to "torch" for PyTorch models
CHUNK_SIZE = 32768
# =========================

# ===== MODEL DOWNLOADER =====
@st.cache_resource(show_spinner=False)
def download_model():
    """Robust model downloader with progress tracking"""
    try:
        # Create direct download URL
        URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        session = requests.Session()
        
        # Initial request
        response = session.get(URL, stream=True)
        response.raise_for_status()
        
        # Check for security token
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # Confirm download if token exists
        if token:
            params = {'id': FILE_ID, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
            response.raise_for_status()
        
        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        file_bytes = BytesIO()
        
        # Create progress elements
        progress_bar = st.progress(0)
        status = st.empty()
        status.subheader("⏳ Downloading model...")
        
        # Stream download with progress updates
        start_time = time.time()
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                file_bytes.write(chunk)
                downloaded += len(chunk)
                
                # Update progress every 0.3 seconds
                if (time.time() - start_time) > 0.3:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress)
                    mb_downloaded = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    status.text(f"📥 Downloaded: {mb_downloaded:.1f}MB / {total_mb:.1f}MB")
                    start_time = time.time()
        
        # Final update
        progress_bar.progress(1.0)
        status.text(f"✅ Download complete! Size: {total_mb:.1f}MB")
        
        return file_bytes
        
    except Exception as e:
        st.error(f"🚨 Download failed: {str(e)}")
        st.info("""
        **Troubleshooting Tips:**
        1. Verify your Google Drive file ID
        2. Ensure file sharing is set to "Anyone with the link"
        3. Check your internet connection
        4. If file is large (>500MB), consider compressing it
        """)
        st.stop()

# ===== MODEL LOADER =====
@st.cache_resource(show_spinner=False)
def load_model(file_bytes):
    """Load model from bytes with error handling"""
    try:
        file_bytes.seek(0)  # Reset pointer
        
        if MODEL_TYPE == "joblib":
            return joblib.load(file_bytes)
        elif MODEL_TYPE == "torch":
            import torch
            return torch.load(file_bytes)
        else:
            raise ValueError(f"Unsupported model type: {MODEL_TYPE}")
            
    except Exception as e:
        st.error(f"🚨 Model loading failed: {str(e)}")
        st.info("""
        **Possible Solutions:**
        1. Verify MODEL_TYPE matches your model format
        2. Ensure training and inference libraries match
        3. Check model file integrity
        4. Update your ML libraries
        """)
        st.stop()

# ===== IMAGE PROCESSING =====
def preprocess_image(uploaded_file):
    """Convert uploaded file to model input format"""
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Basic preprocessing - MODIFY FOR YOUR MODEL
        image = image.resize((224, 224))  # Standard size for many models
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
        
    except Exception as e:
        st.error(f"❌ Image processing error: {str(e)}")
        return None

# ===== MAIN APPLICATION =====
def main():
    st.title("🌸 Flower Classification App")
    st.markdown("""
    <style>
    .reportview-container {
        background: url('https://images.unsplash.com/photo-1490750967868-88aa4486c946');
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.9);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        with st.spinner("⚙️ Initializing model..."):
            file_bytes = download_model()
            st.session_state.model = load_model(file_bytes)
    
    st.success("✅ Model loaded successfully! Ready for predictions.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a flower image", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        # Process image
        processed_image = preprocess_image(uploaded_file)
        
        if processed_image is not None:
            # Make prediction
            with st.spinner("🔍 Analyzing flower..."):
                try:
                    # This will vary based on your model - example for classification
                    prediction = st.session_state.model.predict(processed_image)
                    class_idx = np.argmax(prediction)
                    
                    # Sample flower classes - REPLACE WITH YOUR CLASSES
                    FLOWER_CLASSES = [
                        "Rose", "Tulip", "Daisy", "Sunflower", "Lily",
                        "Orchid", "Peony", "Hydrangea", "Daffodil", "Carnation"
                    ]
                    
                    st.subheader(f"Prediction: **{FLOWER_CLASSES[class_idx]}**")
                    st.metric("Confidence", f"{prediction[0][class_idx]*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Ensure your model expects the same input format used in preprocessing")

# Run the app
if __name__ == "__main__":
    main()
