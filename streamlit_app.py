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

# Create a simplified version of the functionality without requiring customtkinter
class SimpleFaceProcessor:
    def __init__(self):
        self.loaded = False
        try:
            # Try to import insightface for face processing
            import insightface
            import onnxruntime
            from insightface.app import FaceAnalysis
            
            # Initialize face analyzer
            self.face_analyzer = FaceAnalysis(
                name="buffalo_l", 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self.loaded = True
        except ImportError as e:
            st.warning(f"Could not initialize face processor: {e}")
    
    def swap_face(self, source_img, target_img):
        """Simple face swap using insightface directly"""
        if not self.loaded:
            return None
            
        try:
            import insightface
            from insightface.app import FaceSwap
            
            # Initialize face swapper
            face_swapper = FaceSwap()
            
            # Get faces from source image
            source_faces = self.face_analyzer.get(source_img)
            if len(source_faces) == 0:
                st.error("No face detected in source image")
                return None
                
            # Get faces from target image
            target_faces = self.face_analyzer.get(target_img)
            if len(target_faces) == 0:
                st.error("No face detected in target image")
                return None
                
            # Swap face
            result = face_swapper.get(target_img, target_faces[0], source_faces[0])
            return result
        except Exception as e:
            st.error(f"Error in face swapping: {e}")
            return None

# Create our simplified processor
face_processor = SimpleFaceProcessor()

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
        st.image(source_file, caption="Source Face", use_container_width=True)
        
if target_file is not None:
    with col2:
        file_type = target_file.name.split(".")[-1].lower()
        if file_type in ["jpg", "jpeg", "png"]:
            st.image(target_file, caption="Target Image", use_container_width=True)
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
            
            # Check if target is an image
            if target_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Read images
                    source_img = cv2.imread(source_path)
                    target_img = cv2.imread(target_path)
                    
                    # Convert BGR to RGB for display
                    source_img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                    
                    # Process using our simplified processor
                    progress_bar.progress(50)
                    result_img = face_processor.swap_face(source_img, target_img)
                    
                    if result_img is not None:
                        # Save result
                        cv2.imwrite(output_path, result_img)
                        
                        # Convert to RGB for display
                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        
                        # Show result
                        progress_bar.progress(100)
                        status_text.text("Processing complete!")
                        
                        st.subheader("Result")
                        st.image(result_img_rgb, caption="Processed Result", use_container_width=True)
                        
                        # Provide download button
                        with open(output_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Result",
                                data=file,
                                file_name="faceswap_result" + os.path.splitext(target_file.name)[1],
                                mime=f"image/{os.path.splitext(target_file.name)[1][1:]}"
                            )
                    else:
                        progress_bar.progress(100)
                        status_text.text("Processing failed.")
                        st.error("Failed to process the face swap. See error message above.")
                        
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.text("Processing failed.")
                    st.error(f"Error processing images: {e}")
            else:
                # For video
                progress_bar.progress(100)
                status_text.text("Video processing not implemented in this simplified version.")
                st.warning("Video processing requires additional dependencies that aren't available in this environment.")

# System information
with st.expander("System Information"):
    st.code(f"Python version: {sys.version}")
    if has_cv2:
        st.success(f"OpenCV version: {cv2.__version__}")
    else:
        st.error("OpenCV is not available")
    
    # Check for insightface
    try:
        import insightface
        st.success(f"InsightFace version: {insightface.__version__}")
    except ImportError:
        st.error("InsightFace not available")
    
    # Try to check for GPU
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            st.success("GPU is available for processing")
        else:
            st.warning("GPU not available, using CPU only")
    except:
        st.info("Unable to check GPU status")

# Footer
st.markdown("---")
st.caption("Created for educational purposes only")
