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

# Add face swapping implementation using face landmarks and blending
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
        
        # Since we're having issues with model download, use facial landmarks-based approach
        st.info("Using face landmark-based blending for face swapping")
        
        # Advanced face swapping using face landmarks and blending
        result = face_swap_using_landmarks(source_img, target_img, source_faces[0], target_faces)
        
        # Save the result
        cv2.imwrite(output_path, result)
        return True
        
    except Exception as e:
        st.error(f"Face swap error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

# Face swapping using face landmarks
def face_swap_using_landmarks(source_img, target_img, source_face, target_faces):
    # Process each target face
    result = target_img.copy()
    
    for target_face in target_faces:
        # Get facial landmarks
        source_landmarks = source_face.landmark_2d_106
        target_landmarks = target_face.landmark_2d_106
        
        # Get face bounding box with some margin
        bbox = target_face.bbox.astype(np.int32)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Add margin
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.2)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(target_img.shape[1], x2 + margin_x)
        y2 = min(target_img.shape[0], y2 + margin_y)
        
        # Find convex hull of face landmarks
        hull_points = cv2.convexHull(target_landmarks.astype(np.int32))
        
        # Create mask of the face
        mask = np.zeros_like(target_img[:,:,0], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull_points, 255)
        
        # Create a tight mask around eyes, nose, mouth for better blending
        feature_mask = np.zeros_like(target_img[:,:,0], dtype=np.uint8)
        
        # Eyes, nose, mouth points (approximate indices for buffalo_l model)
        left_eye_pts = target_landmarks[60:68].astype(np.int32)
        right_eye_pts = target_landmarks[68:76].astype(np.int32)
        nose_pts = target_landmarks[51:55].astype(np.int32)
        mouth_pts = target_landmarks[76:88].astype(np.int32)
        
        # Fill features
        cv2.fillConvexPoly(feature_mask, cv2.convexHull(left_eye_pts), 255)
        cv2.fillConvexPoly(feature_mask, cv2.convexHull(right_eye_pts), 255)
        cv2.fillConvexPoly(feature_mask, cv2.convexHull(nose_pts), 255)
        cv2.fillConvexPoly(feature_mask, cv2.convexHull(mouth_pts), 255)
        
        # Dilate feature mask for better coverage
        feature_mask = cv2.dilate(feature_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        # Estimate affine transformation
        tform = cv2.estimateAffinePartial2D(source_landmarks, target_landmarks, method=cv2.LMEDS)[0]
        
        # Apply the transformation to the source image
        warped = cv2.warpAffine(source_img, tform, (target_img.shape[1], target_img.shape[0]))
        
        # Alpha blending masks
        mask_3d = np.stack([mask] * 3, axis=2) / 255.0
        feature_mask_3d = np.stack([feature_mask] * 3, axis=2) / 255.0
        
        # Create final blend mask with feathering
        blend_mask = cv2.GaussianBlur(mask, (51, 51), 30) / 255.0
        blend_mask = np.stack([blend_mask] * 3, axis=2)
        
        # Perform color correction to match skin tones
        warped_face = warped * mask_3d
        target_face_region = target_img * mask_3d
        
        # Simple color correction using mean and std
        for i in range(3):  # For each color channel
            if np.sum(mask) > 0:  # Avoid division by zero
                mu_s = np.sum(warped_face[:,:,i]) / np.sum(mask_3d[:,:,i])
                mu_t = np.sum(target_face_region[:,:,i]) / np.sum(mask_3d[:,:,i])
                
                std_s = np.sqrt(np.sum(((warped_face[:,:,i] - mu_s) * mask_3d[:,:,i]) ** 2) / np.sum(mask_3d[:,:,i]))
                std_t = np.sqrt(np.sum(((target_face_region[:,:,i] - mu_t) * mask_3d[:,:,i]) ** 2) / np.sum(mask_3d[:,:,i]))
                
                if std_s > 0:
                    warped[:,:,i] = ((warped[:,:,i] - mu_s) * (std_t / std_s)) + mu_t
        
        # Create blended result
        blended = target_img * (1 - blend_mask) + warped * blend_mask
        
        # Enhanced feature blending with feature mask
        feature_blend = target_img * (1 - feature_mask_3d) + warped * feature_mask_3d
        
        # Combine feature blend with main blend
        result = blended.copy()
        result = np.where(feature_mask_3d > 0, feature_blend, result)
        
        # Apply some final filtering to make it look more natural
        face_region = result[y1:y2, x1:x2]
        face_region = cv2.GaussianBlur(face_region, (3, 3), 0)
        result[y1:y2, x1:x2] = face_region
    
    return result.astype(np.uint8)

# Process video frames with the landmark-based approach
def process_video(source_path, target_path, output_path):
    try:
        # Import dependencies
        import cv2
        import insightface
        from insightface.app import FaceAnalysis
        
        # Process fewer frames for speed (e.g., every 3rd frame)
        process_every_n_frames = 3
        
        # Use OpenCV for video processing
        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            st.error("Failed to open video file")
            return False
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize face analyzer
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Read source image and detect face
        source_img = cv2.imread(source_path)
        source_faces = app.get(source_img)
        if len(source_faces) == 0:
            st.error("No face found in source image")
            cap.release()
            return False
        source_face = source_faces[0]
        
        # Set up video writer
        temp_output = os.path.join(os.path.dirname(output_path), "temp_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps/process_every_n_frames, (width, height))
        
        # Process frames
        frame_idx = 0
        last_processed_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every nth frame
            if frame_idx % process_every_n_frames == 0:
                # Detect faces
                faces = app.get(frame)
                
                # Apply face swap if faces detected
                if len(faces) > 0:
                    frame = face_swap_using_landmarks(source_img, frame, source_face, 
                                                    faces[:1] if not many_faces else faces)
                
                # Store this processed frame
                last_processed_frame = frame.copy()
            elif last_processed_frame is not None:
                # Use the last processed frame for in-between frames
                frame = last_processed_frame
            
            # Write the frame
            out.write(frame)
            
            # Update progress
            frame_idx += 1
            if frame_idx % 10 == 0:
                progress = min(95, int(frame_idx / total_frames * 100))
                yield progress
        
        # Release video resources
        cap.release()
        out.release()
        
        # Convert video to web-compatible format
        try:
            import subprocess
            subprocess.run([
                'ffmpeg', '-i', temp_output, 
                '-c:v', 'libx264', '-preset', 'fast',
                '-pix_fmt', 'yuv420p', output_path
            ], check=True, capture_output=True)
        except Exception as e:
            st.warning(f"Video conversion warning: {e}")
            # If ffmpeg conversion fails, use the original output
            import shutil
            shutil.copy2(temp_output, output_path)
        
        yield 100
        return True
    except Exception as e:
        st.error(f"Video processing error: {e}")
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
