import streamlit as st
import os
import sys
import tempfile

# Safer imports with error handling
try:
    import numpy as np
    import cv2
    from PIL import Image
    HAS_DEPENDENCIES = True
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    HAS_DEPENDENCIES = False

# Basic app setup
st.set_page_config(
    page_title="AI Face Swap",
    page_icon="🎭",
    layout="wide"
)

# Add face swapping implementation
def swap_face(source_path, target_path, output_path):
    try:
        # Import insightface for face swapping
        import insightface
        from insightface.app import FaceAnalysis
        from insightface.app import FaceSwap
        
        # Initialize face analyzer
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize face swapper
        swapper = FaceSwap()
        
        # Load source and target images
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)
        
        if source_img is None or target_img is None:
            st.error("Failed to load images")
            return False
            
        # Get faces from source image
        source_faces = app.get(source_img)
        if len(source_faces) == 0:
            st.error("No face found in source image")
            return False
            
        # Get faces from target image
        target_faces = app.get(target_img)
        if len(target_faces) == 0:
            st.error("No face found in target image")
            return False
            
        # Perform face swap
        result = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)
        
        # Save the result
        cv2.imwrite(output_path, result)
        return True
    except Exception as e:
        st.error(f"Face swap error: {e}")
        return False

# Add video face swapping
def swap_video_faces(source_path, video_path, output_path):
    try:
        # Import required modules
        import insightface
        from insightface.app import FaceAnalysis
        from insightface.app import FaceSwap
        
        # Initialize face analyzer and swapper
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        swapper = FaceSwap()
        
        # Load source face
        source_img = cv2.imread(source_path)
        source_faces = app.get(source_img)
        if len(source_faces) == 0:
            st.error("No face found in source image")
            return False
        source_face = source_faces[0]
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video file")
            return False
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every frame
            faces = app.get(frame)
            if len(faces) > 0:
                # Process the first face found (or loop through all faces if needed)
                for face in faces:
                    frame = swapper.get(frame, face, source_face, paste_back=True)
            
            # Write the frame
            out.write(frame)
            
            # Update progress 
            frame_count += 1
            progress = min(99, int(frame_count / total_frames * 100))
            yield progress, frame_count, total_frames
            
        # Release resources
        cap.release()
        out.release()
        return True
    except Exception as e:
        st.error(f"Video processing error: {e}")
        return False

st.title("🎭 AI Face Swap")
st.subheader("Upload a source face and a target image/video to swap faces")

# File upload widgets
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
    if not HAS_DEPENDENCIES:
        st.error("Required dependencies are missing. Cannot perform face swap.")
    else:
        # Try to import insightface first to check if it's available
        try:
            import insightface
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing... Please wait")
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files
                source_path = os.path.join(temp_dir, f"source{os.path.splitext(source_file.name)[1]}")
                with open(source_path, "wb") as f:
                    f.write(source_file.getbuffer())
                    
                target_path = os.path.join(temp_dir, f"target{os.path.splitext(target_file.name)[1]}")
                with open(target_path, "wb") as f:
                    f.write(target_file.getbuffer())
                    
                output_path = os.path.join(temp_dir, f"output{os.path.splitext(target_file.name)[1]}")
                
                # Progress updates
                progress_bar.progress(10)
                
                # Process based on file type
                is_video = target_file.name.lower().endswith('.mp4')
                
                if is_video:
                    status_text.text("Processing video... This may take several minutes")
                    
                    # Process video frame by frame with progress updates
                    for progress, frame_idx, total in swap_video_faces(source_path, target_path, output_path):
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_idx}/{total}...")
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    # Show result
                    st.subheader("Result")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Download Processed Video",
                            data=file,
                            file_name="faceswap_result.mp4",
                            mime="video/mp4"
                        )
                else:
                    status_text.text("Processing image...")
                    
                    # Process image
                    if swap_face(source_path, target_path, output_path):
                        progress_bar.progress(100)
                        status_text.text("Processing complete!")
                        
                        # Read and display the result
                        result_img = cv2.imread(output_path)
                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        
                        st.subheader("Result")
                        st.image(result_img_rgb, caption="Face Swap Result", use_container_width=True)
                        
                        # Download button
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Processed Image",
                                data=file,
                                file_name=f"faceswap_result{os.path.splitext(target_file.name)[1]}",
                                mime=f"image/{os.path.splitext(target_file.name)[1][1:]}"
                            )
                    else:
                        progress_bar.progress(100)
                        status_text.text("Processing failed")
                        
        except ImportError as e:
            st.error(f"Missing insightface library: {e}")
            st.info("Install with: pip install insightface onnxruntime")
        except Exception as e:
            st.error(f"Error during processing: {e}")

# Add footer
st.markdown("---")
st.caption("Created for educational purposes only")

# System info in an expander
with st.expander("System Information"):
    st.write(f"Python version: {sys.version}")
    
    if HAS_DEPENDENCIES:
        st.success("✅ Basic dependencies installed")
        st.write(f"OpenCV version: {cv2.__version__}")
        st.write(f"NumPy version: {np.__version__}")
    else:
        st.error("❌ Some dependencies are missing")
        
    # Check for insightface
    try:
        import insightface
        st.success(f"✅ InsightFace version: {insightface.__version__}")
    except ImportError:
        st.error("❌ InsightFace not installed")
