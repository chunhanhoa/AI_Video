import streamlit as st

st.set_page_config(
    page_title="AI Face Swap",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 AI Face Swap")
st.subheader("Upload a source face and a target image/video to swap faces")

# Simple file upload interface
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

# App status info
st.info("This is the minimal version of the app that should work in all environments")
