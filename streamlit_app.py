import os
import sys
import tempfile
import streamlit as st
from pathlib import Path

# Initialize Streamlit interface first
st.set_page_config(
    page_title="AI Face Swap App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try importing OpenCV with error handling
try:
    import cv2
    import numpy as np
except ImportError as e:
    st.error(f"Failed to import OpenCV: {e}")
    st.error("This might be due to missing system libraries. Please check the logs.")
    st.info("Try installing required system dependencies with: apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6")
    st.stop()

# Add the project directory to the path so we can import the roop modules
sys.path.append(os.path.abspath("."))

# Import core functionality without UI dependencies
try:
    from roop.predictor import predict_image, predict_video
    from roop.processors.frame.core import get_frame_processors_modules
    from roop.utilities import has_image_extension, is_image, is_video, detect_fps
except ImportError as e:
    st.error(f"Failed to import roop modules: {e}")
    st.stop()

st.title("🎭 AI Face Swap")
st.subheader("Upload a source face and target video/image to swap faces")

with st.sidebar:
    st.header("Settings")
    frame_processors = st.multiselect(
        "Frame Processors",
        options=["face_swapper", "face_enhancer"],
        default=["face_swapper"]
    )
    
    keep_fps = st.checkbox("Keep Original FPS", value=True)
    keep_frames = st.checkbox("Keep Temporary Frames", value=False)
    skip_audio = st.checkbox("Skip Audio", value=False)
    many_faces = st.checkbox("Many Faces", value=False)
    
    selected_enhancer = st.selectbox(
        "Face Enhancer",
        options=["GFPGAN"],
        index=0
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Face")
    source_image = st.file_uploader("Upload source face image", type=["jpg", "jpeg", "png"])
    if source_image is not None:
        st.image(source_image, caption="Source Face", use_column_width=True)

with col2:
    st.subheader("Target")
    target_file = st.file_uploader("Upload target video or image", type=["jpg", "jpeg", "png", "mp4"])
    if target_file is not None and is_video(target_file.name):
        st.video(target_file, format="video/mp4")
    elif target_file is not None and is_image(target_file.name):
        st.image(target_file, caption="Target Image", use_column_width=True)

if st.button("Process", type="primary"):
    if source_image is None:
        st.error("Please upload a source face image.")
    elif target_file is None:
        st.error("Please upload a target video or image.")
    else:
        progress_text = "Processing... Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        # Save uploaded files to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_source = os.path.join(temp_dir, "source" + os.path.splitext(source_image.name)[1])
            with open(temp_source, "wb") as f:
                f.write(source_image.getbuffer())
            
            temp_target = os.path.join(temp_dir, target_file.name)
            with open(temp_target, "wb") as f:
                f.write(target_file.getbuffer())
            
            # Output path
            output_path = os.path.join(temp_dir, "output" + os.path.splitext(target_file.name)[1])
            
            try:
                # Process either image or video
                if is_image(target_file.name):
                    result = predict_image(
                        source_path=temp_source,
                        target_path=temp_target,
                        output_path=output_path,
                        frame_processors=frame_processors,
                        many_faces=many_faces,
                    )
                    my_bar.progress(100, text="Processing complete!")
                    result_image = cv2.imread(output_path)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_image, caption="Result", use_column_width=True)
                    
                    # Provide download button for the result
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Download Result",
                            data=file,
                            file_name="face_swap_result" + os.path.splitext(target_file.name)[1],
                            mime=f"image/{os.path.splitext(target_file.name)[1][1:]}"
                        )
                
                elif is_video(target_file.name):
                    result = predict_video(
                        source_path=temp_source,
                        target_path=temp_target,
                        output_path=output_path,
                        frame_processors=frame_processors,
                        keep_fps=keep_fps,
                        keep_frames=keep_frames,
                        skip_audio=skip_audio,
                        many_faces=many_faces,
                    )
                    my_bar.progress(100, text="Processing complete!")
                    
                    # Display the video result
                    st.video(output_path)
                    
                    # Provide download button for the result
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Download Result",
                            data=file,
                            file_name="face_swap_result.mp4",
                            mime="video/mp4"
                        )
            
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                raise e

st.markdown("---")
st.caption("Powered by Roop AI • Created for educational purposes only")
