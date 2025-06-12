import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
import os

def calculate_weight(length_cm):
    """Calculate weight using 4th-degree polynomial formula"""
    return (0.002 * (length_cm**4) 
            - 0.0578 * (length_cm**3) 
            + 0.7526 * (length_cm**2) 
            - 3.5356 * length_cm 
            + 5.8716)

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

    if 'processed' not in st.session_state:
        st.session_state.processed = False

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            file_ext = os.path.splitext(uploaded_file.name)[1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
                tfile.write(file_bytes)
                img_path = tfile.name

            original_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if original_img is None:
                st.error("Error loading image - please try another file")
                return

            model = YOLO('yolov8n-seg-custom.pt')
            
            with st.spinner('Analyzing shrimp...'):
                lengths = []  # Proper initialization
                results = model(img_path)
                
                # Reference detection
                gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                px_to_cm = 0.1  # Default fallback
                if contours:
                    ref_contour = max(contours, key=cv2.contourArea)
                    (_, _), radius = cv2.minEnclosingCircle(ref_contour)
                    px_to_cm = 2.0 / (radius * 2)

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
                            
                            if long_side <= 2 * short_side:
                                diagonal = np.sqrt(width**2 + height**2)
                                length_px = diagonal * 0.9
                                color = (0, 0, 255)
                            else:
                                length_px = long_side * 0.9
                                color = (0, 255, 0)
                            
                            length_cm = length_px * px_to_cm
                            lengths.append(length_cm)
                            
                            cv2.rectangle(annotated_img, 
                                        (int(x_min), int(y_min)),
                                        (int(x_max), int(y_max)),
                                        color, 2)

                if lengths:
                    avg_length = np.mean(lengths)
                    cv = (np.std(lengths) / avg_length) * 100
                    weights = [calculate_weight(l) for l in lengths]
                    avg_weight = np.mean(weights)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 
                               caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), 
                               caption="Analyzed Image", use_container_width=True)
                    
                    st.subheader("Analysis Results")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Average Length", f"{avg_length:.2f} cm")
                    metric_col2.metric("Weight Estimate", f"{avg_weight:.2f} g")
                    metric_col3.metric("Size Variation", f"{cv:.2f}%")
                    
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.hist(lengths, bins=np.arange(0, max(lengths)+1, 0.5), 
                           color='teal', edgecolor='black')
                    ax.set_xlabel('Length (cm)')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                else:
                    st.warning("No shrimp detected in the image")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
        finally:
            if 'img_path' in locals():
                os.unlink(img_path)
            st.session_state.processed = True

if __name__ == "__main__":
    main()