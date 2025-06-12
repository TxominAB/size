import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Shrimp Sampler",
    page_icon="🦐",
    layout="wide"
)

def main():
    st.title("Shrimp Sampling Web App")
    st.markdown("""
    Upload an image containing shrimp and reference circle (2cm diameter) for automated:
    - Length measurement (cm)
    - Weight estimation (g)
    - Size variation analysis (CV%)
    """)
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    def calculate_weight(length_cm):
        return (0.002 * (length_cm**4) 
                - 0.0578 * (length_cm**3) 
                + 0.7526 * (length_cm**2) 
                - 3.5356 * length_cm 
                + 5.8716)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Temporary file handling
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_file.read())
            img_path = tfile.name

        # Load and verify image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original_img is None:
            st.error("Error loading image - please try another file")
            return

        # Load model
        model = YOLO('yolov8n-seg-custom.pt')  # Update path as needed
        
        # Processing pipeline
        with st.spinner('Analyzing shrimp...'):
            # Perform inference
            results = model(img_path)
            
            # Reference circle detection
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                ref_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(ref_contour)
                px_to_cm = 2.0 / (radius * 2)  # Reference diameter 2cm
            else:
                st.warning("Reference circle not found - using default conversion")
                px_to_cm = 0.1  # Fallback value

            # Process detections
            lengths = []
            annotated_img = original_img.copy()

            for result in results:
                if result.masks is not None:
                    for mask in result.masks.xy:
                        x_min, y_min = np.min(mask, axis=0)
                        x_max, y_max = np.max(mask, axis=0)
                        
                        width = x_max - x_min
                        height = y_max - y_min
                        long_side = max(width, height)
                        short_side = min(width, height)
                        
                        # Measurement logic
                        if long_side <= 2 * short_side:
                            diagonal = np.sqrt(width**2 + height**2)
                            length_px = diagonal * 0.85
                            color = (0, 0, 255)  # Red
                        else:
                            length_px = long_side * 0.95
                            color = (0, 255, 0)  # Green
                        
                        length_cm = length_px * px_to_cm
                        lengths.append(length_cm)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_img, 
                                    (int(x_min), int(y_min)),
                                    (int(x_max), int(y_max)),
                                    color, 2)

            # Calculate metrics
            if lengths:
                avg_length = np.mean(lengths)
                cv = (np.std(lengths) / avg_length) * 100
                weights = [calculate_weight(l) for l in lengths]
                avg_weight = np.mean(weights)
                
                # Convert images for display
                annotated_display = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                original_display = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(original_display, caption="Original Image", use_column_width=True)
                
                with col2:
                    st.image(annotated_display, caption="Analyzed Image", use_column_width=True)
                
                # Metrics display
                st.subheader("Analysis Results")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Average Length", f"{avg_length:.2f} cm")
                
                with metric_col2:
                    st.metric("Weight Estimate", f"{avg_weight:.2f} g")
                
                with metric_col3:
                    st.metric("Size Variation", f"{cv:.2f}%")
                
                # Histogram
                fig, ax = plt.subplots(figsize=(8,4))
                ax.hist(lengths, bins=np.arange(0, max(lengths)+1, 0.5), 
                       color='teal', edgecolor='black')
                ax.set_xlabel('Length (cm)')
                ax.set_ylabel('Count')
                ax.set_title('Size Distribution')
                st.pyplot(fig)
                
            else:
                st.warning("No shrimp detected in the image")

        # Cleanup temp file
        os.unlink(img_path)
        st.session_state.processed = True

if __name__ == "__main__":
    main()