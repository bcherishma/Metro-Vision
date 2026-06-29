import sys
import tempfile
from pathlib import Path 
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from frontend.components.sidebar import render_sidebar
from frontend.components.metrics_card import render_metrics_row
from frontend.components.charts import render_count_timeseries
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Dashboard — MetroVision",
    layout="wide",
    initial_sidebar_state="expanded",
)
uploaded_file, confidence, frame_skip, show_trails, show_bev = render_sidebar()

st.markdown("## Dashboard")
st.caption("Live vehicle detection, tracking, and traffic flow analytics")
st.divider()
 
if "count_history" not in st.session_state:
    st.session_state.count_history = []
if "anomaly_log" not in st.session_state:
    st.session_state.anomaly_log = []
if "peak_count" not in st.session_state:
    st.session_state.peak_count = 0
if "unique_count" not in st.session_state:
    st.session_state.unique_count = 0
 
col_video, col_bev = st.columns([3, 2])
col_metrics = st.container()
col_chart = st.container()
 
with col_video:
    st.markdown("#### Live Detection Feed")
    video_placeholder = st.empty()
 
with col_bev:
    st.markdown("#### Bird's Eye View")
    bev_placeholder = st.empty()
 
with col_metrics:
    st.markdown("#### Real-time Metrics")
    metrics_placeholder = st.empty()
 
with col_chart:
    st.markdown("#### Vehicle Count — Time Series")
    chart_placeholder = st.empty()
 
if uploaded_file is None:
    st.info("Upload a video in the sidebar to start live analysis.")
    st.stop()
 
run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)
 
if run_btn:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
 
    import yaml
    cfg_path = "configs/settings.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    cfg["model"]["confidence_threshold"] = confidence
    cfg["video"]["frame_skip"] = frame_skip
 
    updated_cfg_path = "/tmp/metrovision_runtime.yaml"
    with open(updated_cfg_path, "w") as f:
        yaml.dump(cfg, f)
 
    from backend.pipeline.video_pipeline import VideoPipeline
 
    with st.spinner("Loading MetroVision pipeline..."):
        pipeline = VideoPipeline(config_path=updated_cfg_path)
 
    st.session_state.count_history = []
    stop_btn = st.button("⏹  Stop", key="stop_processing")
 
    for result in pipeline.process_video(
        tmp_path,
        show_trails=show_trails,
        show_bev=show_bev,
    ):
        if stop_btn:
            break
 
        flow = result["flow_stats"]
 
        video_placeholder.image(
            result["annotated_frame_rgb"],
            caption=f"Frame {result['frame_number']} | FPS: {result['fps']}",
            use_column_width=True,
        )
 
        if result["bev_frame_rgb"] is not None:
            bev_placeholder.image(
                result["bev_frame_rgb"],
                caption="Top-down perspective",
                use_column_width=True,
            )
 
        st.session_state.peak_count  = flow["peak_count"]
        st.session_state.unique_count = result["unique_count"]
 
        with metrics_placeholder.container():
            render_metrics_row(
                vehicle_count    = flow["vehicle_count"],
                congestion_score = flow["congestion_score"],
                is_congested     = flow["is_congested"],
                unique_count     = result["unique_count"],
                fps              = result["fps"],
                peak_count       = flow["peak_count"],
            )
 
        st.session_state.count_history.append({
            "time":  result["timestamp_sec"],
            "count": flow["vehicle_count"],
        })
 
        chart_placeholder.plotly_chart(
            render_count_timeseries(st.session_state.count_history),
            use_container_width=True,
            key=f"chart_{result['frame_number']}",
        )
 
    st.success("Analysis complete.")
    heatmap_img = pipeline.get_heatmap_image()
    if heatmap_img is not None:
        import cv2
        st.markdown("#### Accumulated Activity Heatmap")
        st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB),
                 caption="Full-video spatial density",
                 use_column_width=True)
 