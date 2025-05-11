import streamlit as st

# Show a simple message if the app is not ready yet
try:
    # Try to import and run the streamlit app
    from streamlit_app import *
except ImportError as e:
    st.error(f"Error importing dependencies: {e}")
    st.warning("Setting up the environment. This might take a few minutes on the first run.")
    st.info("If this message persists, please check the logs for more information.")
    
    # Show a more detailed diagnostic
    import sys
    st.code(f"Python version: {sys.version}")
    
    # Check OpenCV
    try:
        import cv2
        st.success(f"OpenCV version: {cv2.__version__}")
    except ImportError as e:
        st.error(f"OpenCV import error: {e}")
    
    # Check numpy
    try:
        import numpy as np
        st.success(f"NumPy version: {np.__version__}")
    except ImportError as e:
        st.error(f"NumPy import error: {e}")
    
    # Check other key dependencies
    for module in ["onnx", "insightface", "tensorflow"]:
        try:
            imported = __import__(module)
            st.success(f"{module} version: {imported.__version__}")
        except (ImportError, AttributeError) as e:
            st.error(f"{module} import error: {e}")
