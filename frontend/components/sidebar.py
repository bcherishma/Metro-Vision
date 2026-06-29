from __future__ import annotations
from typing import Optional, Tuple
import streamlit as st
 
 
def render_sidebar() -> Tuple:
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60/1a1f36/ffffff?text=MetroVision", use_column_width=True)
        st.markdown("---")
        st.markdown("### Navigation")
        st.page_link("frontend/app.py",                       label="Home",             icon="🏠")
        st.page_link("frontend/pages/1_dashboard.py",         label="Dashboard",        icon="📊")
        st.page_link("frontend/pages/2_heatmap.py",           label="Heatmap",          icon="🌡️")
        st.page_link("frontend/pages/3_anomaly.py",           label="Anomaly Detection",icon="⚠️")
        st.page_link("frontend/pages/4_bird_eye_view.py",     label="Bird's Eye View",  icon="🗺️")
 
        st.markdown("---")
 
        st.markdown("### Input")
        uploaded_file = st.file_uploader(
            "Upload traffic video",
            type=["mp4", "avi", "mov"],
            help="Supports MP4, AVI, MOV. For best results use dashcam or traffic camera footage.",
        )
 
        st.markdown("---")
 
        st.markdown("### Model Settings")
 
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.10, max_value=0.90, value=0.45, step=0.05,
            help="Detections below this score are ignored. Lower = more detections but more noise.",
        )
 
        frame_skip = st.slider(
            "Frame skip",
            min_value=1, max_value=5, value=2, step=1,
            help="Process every Nth frame. Higher = faster but less smooth. Recommended: 2 on Mac.",
        )
 
        st.markdown("---")
 
        st.markdown("### Display Options")
 
        show_trails = st.checkbox(
            "Show trajectory trails",
            value=True,
            help="Draws motion history trails behind each tracked vehicle.",
        )
 
        show_bev = st.checkbox(
            "Bird's Eye View",
            value=True,
            help="Compute overhead perspective transform (adds ~5ms per frame).",
        )
 
        st.markdown("---")
 
        with st.expander("About MetroVision"):
            st.markdown("""
            **MetroVision** is a real-time urban scene perception engine
            built with YOLOv8, ByteTrack, and Streamlit.
 
            Designed for intelligent mobility research — relevant to
            ADAS, autonomous driving, and smart city systems.
 
            **Stack:** YOLOv8 · ByteTrack · OpenCV · Streamlit · Plotly
            """)
 
    return uploaded_file, confidence, frame_skip, show_trails, show_bev
 