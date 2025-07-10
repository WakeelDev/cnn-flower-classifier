import streamlit as st
import requests
from io import BytesIO
import joblib
import time
import numpy as np
from PIL import Image
import sys

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
        response = session.get(URL, stream=True, timeout=30)
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
            response = session.get(URL, params=params, stream=True, timeout=30)
            response.raise_for_status()
        
        # Check if we actually got a file
        if 'content-length' not in response.headers:
            st.error("❌ No content length in response headers")
            return None
            
        return response
        
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
def load_model():
    """Load model with robust progress tracking and error handling"""
    try:
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.subheader("📥 Downloading Model")
        
        # Start download
        response = download_model()
        if response is None:
            return None
            
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        file_bytes = BytesIO()
        
        # Initialize variables to prevent UnboundLocalError
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        progress = 0
        
        # Create progress elements
        progress_bar = st.progress(0)
        status = st.empty()
        status.text("Starting download...")
        
        # Stream download with progress
        start_time = time.time()
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                file_bytes.write(chunk)
                downloaded += len(chunk)
                
                # Update progress
                if total_size > 0:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress)
                    
                    # Update status every 0.3 seconds
                    if (time.time() - start_time) > 0.3:
                        mb_downloaded = downloaded / (1024 * 1024)
                        status.text(f"Downloaded: {mb_downloaded:.1f}MB / {total_mb:.1f}MB")
                        start_time = time.time()
        
        # Final update - ensure variables are defined
        progress_bar.progress(1.0)
        status.text(f"✅ Download complete! Size: {total_mb:.1f}MB")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load model
        file_bytes.seek(0)
        try:
            model = joblib.load(file_bytes)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Model loading failed: {str(e)}")
            st.info("This usually means the file is corrupted, in the wrong format, or requires specific dependencies")
            # Display detailed error info for debugging
            st.markdown('<div class="error-section">', unsafe_allow_html=True)
            st.subheader("🛠️ Debug Information")
            st.code(f"Error type: {type(e).__name__}\nError details: {str(e)}", language="python")
            st.markdown("""
            **Possible Solutions:**
            1. Verify the model file format matches what joblib expects
            2. Check if all required dependencies are installed
            3. Test loading the model locally with the same environment
            4. Re-export the model with the correct serialization method
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            return None
            
    except Exception as e:
        st.error(f"❌ Critical error during model loading: {str(e)}")
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
        model = load_model()
        if model:
            st.session_state.model = model
        else:
            st.error("❌ Model failed to load. Please check the errors above.")
            st.markdown("""
            <div class="troubleshoot">
            <h3>🔧 Advanced Troubleshooting</h3>
            <ol>
                <li><span class="step">Test the download link</span> - <a href="https://drive.google.com/uc?export=download&id={FILE_ID}" target="_blank">Click here</a> to test download in browser</li>
                <li><span class="step">Check file size</span> - The file should be larger than 0 bytes</li>
                <li><span class="step">Verify file format</span> - Ensure it's a valid joblib file</li>
                <li><span class="step">Library compatibility</span> - Match library versions between training and deployment</li>
                <li><span class="step">Local testing</span> - Try loading the model in a local Python environment</li>
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
                # This will vary based on your model - example for classification
                prediction = st.session_state.model.predict(processed_image)
                class_idx = np.argmax(prediction)
                
                # Sample flower classes - REPLACE WITH YOUR ACTUAL CLASSES
                FLOWER_CLASSES = [
                    "Rose", "Tulip", "Daisy", "Sunflower", "Lily",
                    "Orchid", "Peony", "Hydrangea", "Daffodil", "Carnation"
                ]
                
                # Create prediction card
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.subheader(f"Prediction: **{FLOWER_CLASSES[class_idx]}**")
                
                # Confidence meter
                confidence = prediction[0][class_idx] * 100
                st.metric("Confidence", f"{confidence:.2f}%")
                
                # Confidence bar
                st.progress(int(confidence))
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Ensure your model expects the same input format used in preprocessing")

# Add footer
st.markdown("---")
st.markdown("### Troubleshooting Tips")
st.markdown("""
1. **Download issues** - [Test this download link](https://drive.google.com/uc?export=download&id=1yu3dZ77n_rJShBRRcsg_kjMOKGJ67Sqj) directly in your browser
2. **Sharing settings** - Ensure file is set to "Anyone with the link can view"
3. **File format** - Confirm your model is in .pkl, .joblib, or compatible format
4. **File size** - The file should be >1MB (if it's too small, it might be corrupted)
5. **Model compatibility** - Verify your model was trained with joblib-compatible libraries
""")
