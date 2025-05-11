import streamlit as st
import os
import sys

st.set_page_config(
    page_title="AI Face Swap",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 AI Face Swap")

# Add a try-except block for better error handling
try:
    # First check for system libraries
    if not os.path.exists("/usr/lib/x86_64-linux-gnu/libGL.so.1"):
        st.error("Missing libGL.so.1 - Required by OpenCV")
        with st.expander("System Diagnostics"):
            st.code("Required libraries: libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6")
            st.info("These libraries should be in packages.txt")
    
    # Try importing core dependencies with better error handling
    try:
        import numpy as np
        st.success(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        st.error(f"❌ NumPy import error: {e}")
        
    try:
        # For OpenCV, let's add special handling
        os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"  # Fix for jpeg2000 support
        import cv2
        st.success(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        st.error(f"❌ OpenCV import error: {e}")
        st.info("This might be due to missing system libraries.")
    
    # Check other important dependencies
    dependencies = {
        "onnx": "ONNX model format",
        "tensorflow": "TensorFlow framework",
        "insightface": "Face recognition",
        "gfpgan": "Face enhancement",
        "streamlit": "Web interface"
    }
    
    missing_deps = []
    for module, description in dependencies.items():
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "unknown")
            st.success(f"✅ {module} ({description}): {version}")
        except (ImportError, AttributeError) as e:
            st.error(f"❌ {module} ({description}): {str(e)}")
            missing_deps.append(module)
    
    # If all core dependencies are available, try to import the app
    if not missing_deps:
        try:
            # Add the current directory to path to ensure imports work
            sys.path.insert(0, os.path.abspath("."))
            
            # Try to load the main application
            from streamlit_app import *
            st.success("Application loaded successfully!")
        except ImportError as e:
            st.error(f"Failed to import streamlit_app: {e}")
            st.warning("Check if streamlit_app.py exists and has no syntax errors")
    else:
        st.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        st.info("The application can't run until all dependencies are available.")
        
    # Display Python version info
    st.info(f"Python version: {sys.version}")
    
except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.code(f"Python version: {sys.version}")
