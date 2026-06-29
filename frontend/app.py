import sys
from pathlib import Path
import streamlit as st
import yaml
from frontend.components.sidebar import render_sidebar
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="MetroVision — Urban Perception Engine",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Remove default Streamlit top padding */
    .block-container { padding-top: 1.5rem; }
 
    /* MetroVision brand font weight */
    h1, h2, h3 { font-weight: 600; }
 
    /* Metric card styling */
    [data-testid="metric-container"] {
        background: #1a1f36;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #2d3561;
    }
 
    /* Subtle separator */
    hr { border-color: #2d3561; }
 
    /* Alert / info box */
    .stAlert { border-radius: 8px; }
</style>
""", 
unsafe_allow_html=True)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
 
if "processing" not in st.session_state:
    st.session_state.processing = False
 
if "count_history" not in st.session_state:
    st.session_state.count_history = []
 
if "anomaly_log" not in st.session_state:
    st.session_state.anomaly_log = []

uploaded_file, confidence, frame_skip, show_trails, show_bev = render_sidebar()

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 🏙️")
with col_title:
    st.markdown("## MetroVision")
    st.caption("Urban Scene Perception Engine — A real-time system for intelligent mobility")
 
st.divider()

if uploaded_file is None:
    st.info("Upload a traffic video in the sidebar to begin analysis.")
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modules", "5", help="Detection, Tracking, Flow, Heatmap, Anomaly")
    with col2:
        st.metric("Model", "YOLOv8n", help="Optimised for Apple Silicon MPS")
    with col3:
        st.metric("Tracker", "ByteTrack", help="Multi-object tracking with persistent IDs")
    with col4:
        st.metric("Output", "Real-time", help="Live annotated video + analytics charts")
 
    st.markdown("""
    ### How to use MetroVision
    1. Upload a traffic video using the **sidebar**
    2. Adjust the confidence threshold and frame skip as needed
    3. Click **Run Analysis** to start processing
    4. Navigate between pages using the sidebar menu:
       - **Dashboard** — live video + vehicle counts
       - **Heatmap** — spatial activity density
       - **Anomaly Detection** — pothole and road hazard detection
       - **Bird's Eye View** — top-down scene map
    """)
else:
    st.success(f"Video uploaded: `{uploaded_file.name}` — head to the **Dashboard** page to run analysis.")
 
 
 
