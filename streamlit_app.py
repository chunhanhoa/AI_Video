import streamlit as st
import os
import sys
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="AI Face Swap",
    page_icon="🎭",
    layout="wide"
)

# Initialize these variables for safety
has_core_modules = False
has_cv2 = False

# Import dependencies with error handling
try:
    import cv2
    import numpy as np
    has_cv2 = True
except ImportError as e:
    st.error(f"Error importing OpenCV: {e}")

# Add the project directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try to import roop modules
try:
    # These are the key modules from your roop library
    from roop.processors.frame.face_swapper import FaceSwapper
    from roop.processors.frame.face_enhancer import FaceEnhancer
    from roop.core import extract_face_images, find_similar_face
    from roop.utilities import has_image_extension, is_image, is_video, detect_fps
    
    # Initialize processors
    face_swapper = FaceSwapper()
    face_enhancer = FaceEnhancer()
    has_core_modules = True
except ImportError as e:
    st.warning(f"Could not import roop modules: {e}")
    st.info("Running in limited functionality mode.")

# App title and description
st.title("🎭 AI Face Swap")
st.subheader("Upload a source face and a target image/video to swap faces")

# File upload interface
source_file = st.file_uploader("Upload source face image", type=["jpg", "jpeg", "png"])
target_file = st.file_uploader("Upload target video/image", type=["jpg", "jpeg", "png", "mp4"])

# Display uploaded files
col1, col2 = st.columns(2)
if source_file is not None:
    with col1:
        st.image(source_file, caption="Source Face", use_column_width=True)
        
if target_file is not None:
    with col2:
        file_type = target_file.name.split(".")[-1].lower()
        if file_type in ["jpg", "jpeg", "png"]:
            st.image(target_file, caption="Target Image", use_column_width=True)
        elif file_type == "mp4":
            st.video(target_file, format="video/mp4")

# Processing settings
with st.sidebar:
    st.header("Settings")
    enhance_face = st.checkbox("Enhance Face Quality", value=True)
    keep_fps = st.checkbox("Keep Original FPS", value=True)
    keep_frames = st.checkbox("Keep Temp Frames", value=False)
    skip_audio = st.checkbox("Skip Audio", value=False)
    many_faces = st.checkbox("Process Multiple Faces", value=False)

# Define processing function
def process_files(source_path, target_path, output_path):
    if not has_core_modules:
        st.error("Required modules not available. Cannot process files.")
        return False
    
    try:
        if is_image(target_path):
            # Process image
            source_face = extract_face_images(source_path)[0]
            target_image = cv2.imread(target_path)
            
            # Swap face
            result_image = face_swapper.process_frame(target_image, source_face)
            
            # Enhance if selected
            if enhance_face:
                result_image = face_enhancer.process_frame(result_image)
                
            # Save result
            cv2.imwrite(output_path, result_image)
            return True
            
        elif is_video(target_path):
            # For video, we need to process frame by frame
            from roop.predictor import predict_video
            result = predict_video(
                source_path=source_path,
                target_path=target_path,
                output_path=output_path,
                frame_processors=["face_swapper", "face_enhancer"] if enhance_face else ["face_swapper"],
                keep_fps=keep_fps,
                keep_frames=keep_frames,
                skip_audio=skip_audio,
                many_faces=many_faces
            )
            return True
    except Exception as e:
        st.error(f"Processing error: {e}")
        return False

# Process button
if st.button("Start Face Swap", disabled=(source_file is None or target_file is None)):
    if has_cv2 and source_file and target_file:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Preparing files...")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to disk
            progress_bar.progress(10)
            status_text.text("Saving uploaded files...")
            
            # Save source file
            source_path = os.path.join(temp_dir, "source" + os.path.splitext(source_file.name)[1])
            with open(source_path, "wb") as f:
                f.write(source_file.getbuffer())
            
            # Save target file
            target_path = os.path.join(temp_dir, "target" + os.path.splitext(target_file.name)[1])
            with open(target_path, "wb") as f:
                f.write(target_file.getbuffer())
            
            # Set output path
            output_path = os.path.join(temp_dir, "output" + os.path.splitext(target_file.name)[1])
            
            # Process files
            progress_bar.progress(30)
            status_text.text("Processing... This may take a while.")
            
            # Actual processing
            success = process_files(source_path, target_path, output_path)
            
            if success:
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Display result
                st.subheader("Result")
                if is_image(target_path):
                    # For image results
                    result_image = cv2.imread(output_path)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_image, caption="Processed Result", use_column_width=True)
                    
                    # Provide download button
                    with open(output_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Result",
                            data=file,
                            file_name="faceswap_result" + os.path.splitext(target_file.name)[1],
                            mime=f"image/{os.path.splitext(target_file.name)[1][1:]}"
                        )
                else:
                    # For video results
                    st.video(output_path)
                    
                    # Provide download button
                    with open(output_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Result",
                            data=file,
                            file_name="faceswap_result.mp4",
                            mime="video/mp4"
                        )
            else:
                progress_bar.progress(100)
                status_text.text("Processing failed.")
                st.error("Failed to process files. See error message above.")

# System information
with st.expander("System Information"):
    st.code(f"Python version: {sys.version}")
    if has_cv2:
        st.success(f"OpenCV version: {cv2.__version__}")
    else:
        st.error("OpenCV is not available")
    
    if has_core_modules:
        st.success("Roop face swap modules loaded successfully")
    else:
        st.error("Roop face swap modules could not be loaded")

# Footer
st.markdown("---")
st.caption("Created for educational purposes only")
