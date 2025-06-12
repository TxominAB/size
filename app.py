import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Shrimp Sampling Model",
    page_icon="🦐",
    layout="wide"
)

def main():
    st.title("Shrimp Measurement Web App")
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
        # Read file content ONCE and store in variable
        file_bytes = uploaded_file.read()  # <--- THIS IS CRUCIAL
        
        # Temporary file handling using the stored bytes
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
            tfile.write(file_bytes)  # Use pre-read bytes
            img_path = tfile.name

        # Load image from the stored bytes
        original_img = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),  # Use existing variable
            cv2.IMREAD_COLOR
        )

        # Rest of the code remains the same...
        # Load model
        model = YOLO('yolov8n-seg-custom.pt')
        
        # In the processing pipeline section:
        with st.spinner('Analyzing shrimp...'):
            # Initialize lengths here
            lengths = []  # <-- ADD THIS LINE
            
            # Perform inference
            results = model(img_path)
            
            # Reference circle detection
            # ... (existing reference detection code)
            
            # Process detections
            annotated_img = original_img.copy()

            for result in results:
                if result.masks is not None:
                    for mask in result.masks.xy:
                        # ... (existing measurement logic)
                        lengths.append(length_cm)  # Now appending to initialized list

            # Calculate metrics
            if lengths:  # Now properly scoped and defined
                avg_length = np.mean(lengths)
                cv = (np.std(lengths) / avg_length) * 100
                # ... rest of metric calculations
            else:
                st.warning("No shrimp detected in the image")

        # Cleanup temp file
        os.unlink(img_path)
        st.session_state.processed = True

if __name__ == "__main__":
    main()