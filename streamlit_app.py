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

# Add face swapping implementation using InsightFace model directly
def swap_face(source_path, target_path, output_path):
    try:
        # Import insightface for face swapping
        import insightface
        from insightface.app import FaceAnalysis
        
        # Initialize face analyzer
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
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
        
        # Try a different approach for face swapping since downloading is failing
        try:
            import insightface.model_zoo
            try:
                # Try first with local URL
                model = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)
            except:
                # Try with alternative URL
                alt_url = "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
                local_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
                
                # Make sure directory exists
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Download directly
                st.info(f"Downloading model from alternative source: {alt_url}")
                import urllib.request
                urllib.request.urlretrieve(alt_url, local_path)
                
                # Load the model
                model = insightface.model_zoo.get_model(local_path, download=False)
            
            # Perform face swap
            result = target_img.copy()
            for target_face in target_faces:
                result = model.get(result, target_face, source_faces[0], paste_back=True)
            
            # Save the result
            cv2.imwrite(output_path, result)
            return True
        except Exception as e:
            st.error(f"Error with model: {e}")
            
            # Simple alternative method - blend the faces
            st.warning("Using simplified face blending as fallback")
            
            # Get face landmarks
            source_lm = source_faces[0].landmark_2d_106
            target_lm = target_faces[0].landmark_2d_106
            
            # Create a simple mask for the face region
            mask = np.zeros_like(target_img[:,:,0], dtype=np.uint8)
            hull = cv2.convexHull(target_lm.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Warp source face to match target face shape
            h, w = target_img.shape[:2]
            warped = cv2.warpAffine(
                source_img,
                cv2.estimateAffinePartial2D(source_lm, target_lm, method=cv2.LMEDS)[0],
                (w, h)
            )
            
            # Blend faces
            mask_3d = np.stack([mask]*3, axis=2) / 255.0
            result = target_img * (1-mask_3d) + warped * mask_3d
            result = result.astype(np.uint8)
            
            # Save result
            cv2.imwrite(output_path, result)
            return True
    except Exception as e:
        st.error(f"Face swap error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

# Safer video face swapping
def process_video(source_path, video_path, output_path):
    try:
        # Import dependencies
        import cv2
        
        # Use OpenCV for video processing instead of moviepy (more reliable)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video file")
            return False
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize face analyzer
        import insightface
        from insightface.app import FaceAnalysis
        
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Read source face
        source_img = cv2.imread(source_path)
        source_faces = app.get(source_img)
        if len(source_faces) == 0:
            st.error("No face found in source image")
            cap.release()
            return False
        source_face = source_faces[0]
        
        # Try to get the model
        try:
            import insightface.model_zoo
            try:
                # Try first with local URL
                model = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)
            except:
                # Try with alternative URL
                alt_url = "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
                local_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
                
                # Make sure directory exists
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Download directly
                st.info(f"Downloading model from alternative source: {alt_url}")
                import urllib.request
                urllib.request.urlretrieve(alt_url, local_path)
                
                # Load the model
                model = insightface.model_zoo.get_model(local_path, download=False)
        except Exception as e:
            st.error(f"Failed to load face swap model: {e}")
            cap.release()
            return False
        
        # Prepare output video
        temp_dir = os.path.dirname(output_path)
        temp_output = os.path.join(temp_dir, "temp_output.mp4") 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every 2nd frame for speed (adjust as needed)
            if frame_count % 2 == 0:
                # Detect faces
                faces = app.get(frame)
                
                # Swap faces if any are found
                if len(faces) > 0:
                    # Apply to all faces or just the first one based on setting
                    face_to_process = faces if many_faces else [faces[0]]
                    for face in face_to_process:
                        try:
                            frame = model.get(frame, face, source_face, paste_back=True)
                        except:
                            pass
            
            # Write the frame
            out.write(frame)
            
            # Update progress
            frame_count += 1
            if frame_count % 10 == 0:  # Update progress every 10 frames
                progress = min(95, int((frame_count / total_frames) * 100))
                yield progress
        
        # Release resources
        cap.release()
        out.release()
        
        # Copy output with compatible codec (convert if needed)
        import subprocess
        try:
            subprocess.run([
                'ffmpeg', '-i', temp_output, 
                '-c:v', 'libx264', '-preset', 'fast',
                '-pix_fmt', 'yuv420p', output_path
            ], check=True, capture_output=True)
        except:
            # If ffmpeg fails, just use the original output
            import shutil
            shutil.copy2(temp_output, output_path)
        
        yield 100
        return True
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
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
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Processing... Please wait")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save uploaded files
                source_path = os.path.join(temp_dir, f"source{os.path.splitext(source_file.name)[1]}")
                with open(source_path, "wb") as f:
                    f.write(source_file.getbuffer())
                    
                target_path = os.path.join(temp_dir, f"target{os.path.splitext(target_file.name)[1]}")
                with open(target_path, "wb") as f:
                    f.write(target_file.getbuffer())
                
                # Set output path
                output_path = os.path.join(temp_dir, f"output{os.path.splitext(target_file.name)[1]}")
                
                # Progress updates
                progress_bar.progress(10)
                
                # Process based on file type
                is_video = target_file.name.lower().endswith('.mp4')
                
                if is_video:
                    status_text.text("Processing video... This may take several minutes")
                    
                    # Process video frame by frame with progress updates
                    success = False
                    for progress in process_video(source_path, target_path, output_path):
                        progress_bar.progress(progress)
                        status_text.text(f"Video processing: {progress}%")
                        if progress == 100:
                            success = True
                    
                    if success and os.path.exists(output_path):
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
                        st.error("Failed to process video. See errors above.")
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
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

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
