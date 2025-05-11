import streamlit as st

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Face Swap",
    page_icon="🎭",
    layout="wide"
)

# Only import other libraries after st.set_page_config
import os
import sys
import tempfile
import numpy as np
import cv2
from PIL import Image
import subprocess

# Check for torch AFTER set_page_config
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Safer imports with error handling
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

# Check for basic dependencies
HAS_DEPENDENCIES = all([np, cv2, Image])

# Display dependency errors 
if not HAS_DEPENDENCIES:
    st.error("Missing core dependencies: numpy, opencv-python, or pillow.")
if not HAS_INSIGHTFACE:
    st.error("InsightFace not installed. Falling back to basic face detection.")

# Create face mask function
def create_face_mask(img, landmarks, dilation=5):
    hull = cv2.convexHull(landmarks.astype(np.int32))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = np.ones((dilation, dilation), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask / 255.0

# Advanced color correction
def color_correct_face(source_face, target_face, mask):
    source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
    for i in range(3):
        if np.sum(mask) > 0:
            mu_s = np.sum(source_lab[:, :, i] * mask) / np.sum(mask)
            mu_t = np.sum(target_lab[:, :, i] * mask) / np.sum(mask)
            std_s = np.sqrt(np.sum(((source_lab[:, :, i] - mu_s) * mask) ** 2) / np.sum(mask))
            std_t = np.sqrt(np.sum(((target_lab[:, :, i] - mu_t) * mask) ** 2) / np.sum(mask))
            if std_s > 0:
                source_lab[:, :, i] = ((source_lab[:, :, i] - mu_s) * (std_t / std_s)) + mu_t
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

# Landmark-based face swapping
def face_swap_using_landmarks(source_img, target_img, source_face, target_faces):
    result = target_img.copy()
    
    for target_face in target_faces:
        source_landmarks = source_face.landmark_2d_106
        target_landmarks = target_face.landmark_2d_106
        
        bbox = target_face.bbox.astype(np.int32)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * 0.15)
        margin_y = int(height * 0.15)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(target_img.shape[1], x2 + margin_x)
        y2 = min(target_img.shape[0], y2 + margin_y)
        
        mask = create_face_mask(target_img, target_landmarks)
        mask_3d = np.stack([mask] * 3, axis=2)
        
        tform = cv2.estimateAffinePartial2D(source_landmarks, target_landmarks, method=cv2.LMEDS)[0]
        warped = cv2.warpAffine(source_img, tform, (target_img.shape[1], target_img.shape[0]))
        warped = color_correct_face(warped, target_img, mask)
        
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        try:
            result = cv2.seamlessClone(warped, result, (mask * 255).astype(np.uint8), center, cv2.NORMAL_CLONE)
        except:
            result = result * (1 - mask_3d) + warped * mask_3d
        
        face_region = result[y1:y2, x1:x2]
        face_region = cv2.GaussianBlur(face_region, (3, 3), 0)
        result[y1:y2, x1:x2] = face_region
    
    return result.astype(np.uint8)

# Process video
def process_video(source_path, target_path, output_path, many_faces=False):
    try:
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        source_img = cv2.imread(source_path)
        source_faces = app.get(source_img)
        if len(source_faces) == 0:
            st.error("No face found in source image")
            return False
        source_face = source_faces[0]
        
        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            st.error("Failed to open video file")
            return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process every 3rd frame for speed
        process_frame_interval = 3
        
        temp_output = os.path.join(os.path.dirname(output_path), "temp_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps/process_frame_interval, (width, height))
        
        frame_idx = 0
        last_processed = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every nth frame
            if frame_idx % process_frame_interval == 0:
                faces = app.get(frame)
                if len(faces) > 0:
                    frame = face_swap_using_landmarks(
                        source_img, frame, source_face, 
                        faces[:1] if not many_faces else faces
                    )
                last_processed = frame.copy()
            elif last_processed is not None:
                # Use last processed frame for in-between frames
                frame = last_processed
            
            out.write(frame)
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                progress = min(95, int(frame_idx / total_frames * 100))
                yield progress
        
        cap.release()
        out.release()
        
        try:
            subprocess.run([
                'ffmpeg', '-i', temp_output, '-i', target_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0?',
                '-pix_fmt', 'yuv420p', output_path
            ], check=True, capture_output=True)
        except Exception as e:
            st.warning(f"Video conversion warning: {e}")
            import shutil
            shutil.copy2(temp_output, output_path)
        
        yield 100
        return True
    except Exception as e:
        st.error(f"Video processing error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

# Streamlit UI
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

# Device check
device = "cpu"
if HAS_TORCH and torch.cuda.is_available():
    device = "cuda"
st.sidebar.write(f"Running on: {device}")

# Process button
if st.button("Start Face Swap", disabled=(source_file is None or target_file is None)):
    if not HAS_DEPENDENCIES or not HAS_INSIGHTFACE:
        st.error("Required dependencies are missing. Please install insightface and other dependencies.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Processing... Please wait")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                source_path = os.path.join(temp_dir, f"source{os.path.splitext(source_file.name)[1]}")
                with open(source_path, "wb") as f:
                    f.write(source_file.getbuffer())
                    
                target_path = os.path.join(temp_dir, f"target{os.path.splitext(target_file.name)[1]}")
                with open(target_path, "wb") as f:
                    f.write(target_file.getbuffer())
                
                output_path = os.path.join(temp_dir, f"output{os.path.splitext(target_file.name)[1]}")
                
                progress_bar.progress(10)
                
                is_video = target_file.name.lower().endswith('.mp4')
                
                if is_video:
                    status_text.text("Processing video... This may take several minutes")
                    
                    success = False
                    for progress in process_video(source_path, target_path, output_path, many_faces):
                        progress_bar.progress(progress)
                        status_text.text(f"Video processing: {progress}%")
                        if progress == 100:
                            success = True
                    
                    if success and os.path.exists(output_path):
                        st.subheader("Result")
                        st.video(output_path)
                        
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
                    
                    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
                    app.prepare(ctx_id=0, det_size=(640, 640))
                    
                    source_img = cv2.imread(source_path)
                    target_img = cv2.imread(target_path)
                    source_faces = app.get(source_img)
                    target_faces = app.get(target_img)
                    
                    if len(source_faces) == 0 or len(target_faces) == 0:
                        st.error("No face found in source or target image")
                    else:
                        result_img = face_swap_using_landmarks(
                            source_img, target_img, source_faces[0], 
                            target_faces[:1] if not many_faces else target_faces
                        )
                        cv2.imwrite(output_path, result_img)
                        
                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        st.subheader("Result")
                        st.image(result_img_rgb, caption="Face Swap Result", use_container_width=True)
                        
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Processed Image",
                                data=file,
                                file_name=f"faceswap_result{os.path.splitext(target_file.name)[1]}",
                                mime=f"image/{os.path.splitext(target_file.name)[1][1:]}"
                            )
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Add footer
st.markdown("---")
st.caption("Created for educational purposes only")

# System info
with st.expander("System Information"):
    st.write(f"Python version: {sys.version}")
    
    if HAS_DEPENDENCIES:
        st.success("✅ Basic dependencies installed")
        st.write(f"OpenCV version: {cv2.__version__}")
        st.write(f"NumPy version: {np.__version__}")
    else:
        st.error("❌ Some dependencies are missing")
        
    if HAS_INSIGHTFACE:
        st.success(f"✅ InsightFace installed")
    else:
        st.error("❌ InsightFace not installed")
        
    if HAS_TORCH:
        st.success(f"✅ PyTorch {torch.__version__}")
    else:
        st.warning("⚠️ PyTorch not installed")