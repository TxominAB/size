import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Shrimp Analyzer",
    page_icon="🦐",
    layout="wide"
)

def main():
    st.title("GROBEST Shrimp Measurement Web App")
    st.markdown("""
    Upload an image containing shrimp and reference circle (2cm diameter) for automated:
    - Length measurement (cm)
    - Weight estimation (g)
    - Size variation analysis (CV%)
    """)
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read file content ONCE and reuse
        file_bytes = uploaded_file.read()
        
        # Temporary file handling
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(file_bytes)  # Use already read bytes
            img_path = tfile.name

        # Load image from bytes
        original_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if original_img is None:
            st.error("Error loading image - please try another file")
            return

        # Rest of the code remains the same...
        # Load model
        model = YOLO('models/yolov8n-seg-custom.pt')
        
        # Processing pipeline
        with st.spinner('Analyzing shrimp...'):
            # Perform inference
            results = model(img_path)
            
            # ... (remaining code unchanged up to metrics calculation)

            # Calculate metrics
            if lengths:
                avg_length = np.mean(lengths)
                cv = (np.std(lengths) / avg_length) * 100
                weights = [calculate_weight(l) for l in lengths]  # Using polynomial formula
                avg_weight = np.mean(weights)
                
                # ... (remaining visualization code unchanged)

        # Cleanup temp file
        os.unlink(img_path)
        st.session_state.processed = True

if __name__ == "__main__":
    main()