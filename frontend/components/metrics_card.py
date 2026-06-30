import streamlit as st
 
def render_metrics_row(
    vehicle_count: int,
    congestion_score: float,
    is_congested: bool,
    unique_count: int,
    fps: float,
    peak_count: int,
) -> None:
    """
    Renders a row of KPI metric cards.
    Uses Streamlit's st.metric() which renders a clean number + delta.
    """
    c1, c2, c3, c4, c5, c6 = st.columns(6)
 
    with c1:
        st.metric(
            label="Vehicles now",
            value=vehicle_count,
            delta=None,
            help="Number of vehicles detected in the current frame",
        )
 
    with c2:
        congestion_label = "🔴 CONGESTED" if is_congested else "🟢 CLEAR"
        st.metric(
            label="Congestion",
            value=f"{congestion_score:.0f}%",
            delta=congestion_label,
            delta_color="inverse",
            help="Congestion score: 0 = free flow, 100 = fully congested",
        )
 
    with c3:
        st.metric(
            label="Unique vehicles",
            value=unique_count,
            help="Total distinct vehicles tracked since analysis started",
        )
 
    with c4:
        st.metric(
            label="Peak count",
            value=peak_count,
            help="Maximum vehicles seen in a single frame this session",
        )
 
    with c5:
        st.metric(
            label="Processing FPS",
            value=f"{fps:.1f}",
            help="Frames processed per second (target: 10+ on Mac MPS)",
        )
 
    with c6:
        status = "Active" if fps > 0 else "Idle"
        st.metric(
            label="Pipeline status",
            value=status,
        )
 