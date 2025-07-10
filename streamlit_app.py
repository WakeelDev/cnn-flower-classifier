import streamlit as st
import requests
from io import BytesIO
import joblib
import time
import re

# Set page configuration
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ===== CONFIGURATION =====
# REPLACE THIS WITH YOUR ACTUAL GOOGLE DRIVE SHAREABLE LINK
DRIVE_LINK = "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing" 

# Automatically extract file ID from any Google Drive URL format
def extract_file_id(url):
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"open\?id=([a-zA-Z0-9_-]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    st.error(f"❌ Could not extract file ID from URL: {url}")
    st.info("Make sure your Google Drive link is in one of these formats:")
    st.info("1. https://drive.google.com/file/d/FILE_ID/view?usp=sharing")
    st.info("2. https://drive.google.com/open?id=FILE_ID")
    st.stop()
    return None

# Get file ID from the drive link
FILE_ID = extract_file_id(DRIVE_LINK)
# =========================

# ===== ROBUST DOWNLOAD FUNCTION =====
def download_model():
    """Improved downloader with proper Google Drive URL handling"""
    try:
        # Correct download URL format
        URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        session = requests.Session()
        
        # First request to get confirmation token
        response = session.get(URL, stream=True)
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
            response = session.get(URL, params=params, stream=True)
            response.raise_for_status()
        
        # Verify we got a valid response
        if 'content-disposition' not in response.headers:
            st.error("⚠️ Google Drive blocked the download")
            st.info("""
            **Solution:**
            1. Open this link in your browser: 
            https://drive.google.com/uc?export=download&id={FILE_ID}
            2. If you see a security warning, click "Download anyway"
            3. Copy the new URL after completing the security check
            """)
            st.stop()
            
        return response
        
    except requests.exceptions.HTTPError as e:
        st.error(f"🚨 HTTP Error: {e.response.status_code}")
        if e.response.status_code == 404:
            st.error(f"File not found. Verify your file ID: {FILE_ID}")
        elif e.response.status_code == 403:
            st.error("Permission denied. Ensure sharing is set to 'Anyone with the link'")
        return None
    except Exception as e:
        st.error(f"🚨 Download failed: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model with progress tracking"""
    try:
        st.subheader("📥 Downloading Model")
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Start download
        response = download_model()
        if response is None:
            return None
            
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        file_bytes = BytesIO()
        
        # Stream download with progress
        start_time = time.time()
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                file_bytes.write(chunk)
                downloaded += len(chunk)
                
                # Update progress
                if (time.time() - start_time) > 0.3:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress)
                    mb_downloaded = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    status.text(f"Downloaded: {mb_downloaded:.1f}MB / {total_mb:.1f}MB")
                    start_time = time.time()
        
        # Final update
        progress_bar.progress(1.0)
        status.text(f"✅ Download complete! Size: {total_mb:.1f}MB")
        st.balloons()
        
        # Load model
        file_bytes.seek(0)
        return joblib.load(file_bytes)
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

# ===== MAIN APP =====
def main():
    st.title("🌸 Flower Classification App")
    st.markdown("---")
    
    # Display file info
    st.info(f"**Using model from:** [Google Drive File](https://drive.google.com/file/d/{FILE_ID}/view)")
    st.info(f"**File ID:** `{FILE_ID}`")
    
    # Initialize model
    if 'model' not in st.session_state:
        with st.spinner("Initializing model..."):
            st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("Model failed to load. Please check the errors above.")
        return
    
    st.success("✅ Model loaded successfully! Ready for predictions.")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            # Display image
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
            # Add your image processing and prediction code here
            # For example:
            # processed_image = preprocess(uploaded_file)
            # prediction = st.session_state.model.predict(processed_image)
            
            st.subheader("Prediction Results")
            st.write("Replace this with your actual prediction code")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
