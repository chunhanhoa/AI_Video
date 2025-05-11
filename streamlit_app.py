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
        
        # Since FaceSwap is not available in the cloud environment,
        # we'll implement a simpler version using the model directly
        import insightface.model_zoo
        
        # Load face swapping model
        model = insightface.model_zoo.get_model('inswapper_128.onnx', 
                                              download=True, 
                                              download_zip=True)
        
        # Perform face swap
        result = target_img.copy()
        for target_face in target_faces:
            result = model.get(result, target_face, source_faces[0], paste_back=True)
        
        # Save the result
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
        # Use moviepy instead of OpenCV for video processing
        from moviepy.editor import VideoFileClip, ImageSequenceClip
        import insightface
        from insightface.app import FaceAnalysis
        import insightface.model_zoo
        import glob
        
        # Create temp directory for frames
        temp_frames_dir = os.path.join(os.path.dirname(output_path), "frames")
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        # Load video
        video_clip = VideoFileClip(video_path)
        
        # Initialize face analyzer
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load face swapping model
        model = insightface.model_zoo.get_model('inswapper_128.onnx', 
                                              download=True, 
                                              download_zip=True)
        
        # Load source image and detect face
        source_img = cv2.imread(source_path)
        source_faces = app.get(source_img)
        if len(source_faces) == 0:
            st.error("No face found in source image")
            return False
        source_face = source_faces[0]
        
        # Process the video
        total_frames = int(video_clip.fps * video_clip.duration)
        processed_frames = []
        
        # Process in batches to save memory
        frame_batch_size = 30
        for i, frame in enumerate(video_clip.iter_frames()):
            # Convert RGB to BGR (for insightface)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = app.get(frame_bgr)
            
            # Apply face swap if faces are detected
            if len(faces) > 0:
                for face in faces:
                    frame_bgr = model.get(frame_bgr, face, source_face, paste_back=True)
            
            # Convert back to RGB for moviepy
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            processed_frames.append(frame_rgb)
            
            # Save batch of frames
            if len(processed_frames) >= frame_batch_size or i == total_frames - 1:
                batch_clip = ImageSequenceClip(processed_frames, fps=video_clip.fps)
                
                # Use unique temp filename for each batch
                temp_output = os.path.join(temp_frames_dir, f"batch_{i//frame_batch_size}.mp4")
                batch_clip.write_videofile(temp_output, codec='libx264', audio=False, verbose=False, logger=None)
                
                # Clear memory
                processed_frames = []
                batch_clip = None
                
            # Update progress
            yield min(95, int((i + 1) / total_frames * 100))
            
        # Combine all batches
        from moviepy.editor import concatenate_videoclips
        batch_files = sorted(glob.glob(os.path.join(temp_frames_dir, "batch_*.mp4")))
        batch_clips = [VideoFileClip(f) for f in batch_files]
        
        # Create final clip
        if batch_clips:
            final_clip = concatenate_videoclips(batch_clips, method="compose") 
            
            # Add audio if available
            if video_clip.audio is not None:
                final_clip = final_clip.set_audio(video_clip.audio)
                
            # Write final video
            final_clip.write_videofile(output_path, codec='libx264')
            
            # Clean up
            for clip in batch_clips:
                clip.close()
            final_clip.close()
            
        yield 100
        return True
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False
    finally:
        # Close video clip
        try:
            video_clip.close()
        except:
            pass

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
