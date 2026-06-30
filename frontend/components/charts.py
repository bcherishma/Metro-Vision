from __future__ import annotations
from typing import List
import plotly.graph_objects as go
 
 
# MetroVision brand colors
PRIMARY   = "#4F8EF7"
DANGER    = "#FF4B4B"
SUCCESS   = "#21C55D"
NEUTRAL   = "#8892B0"
BORDER    = "#30363D"
 
 
def render_count_timeseries(count_history: List[dict]) -> go.Figure:
    if not count_history:
        return _empty_figure("No data yet")
 
    times  = [d["time"]  for d in count_history]
    counts = [d["count"] for d in count_history]
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=counts,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(79, 142, 247, 0.15)",
        line=dict(color=PRIMARY, width=2),
        name="Vehicles",
        hovertemplate="Time: %{x:.1f}s<br>Count: %{y}<extra></extra>",
    ))
 
    fig.update_layout(
        **_base_layout(),
        xaxis_title="Time (seconds)",
        yaxis_title="Vehicle count",
        height=220,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig
 
 
def render_class_breakdown_bar(class_breakdown: dict) -> go.Figure:
    """
    Horizontal bar chart showing vehicle class distribution.
    """
    if not class_breakdown:
        return _empty_figure("No detections")
 
    classes = list(class_breakdown.keys())
    counts  = list(class_breakdown.values())
 
    class_colors = {
        "car":        PRIMARY,
        "truck":      DANGER,
        "bus":        "#4BC0C0",
        "person":     SUCCESS,
        "bicycle":    "#9B59B6",
        "motorcycle": "#F39C12",
    }
    colors = [class_colors.get(c, NEUTRAL) for c in classes]
 
    fig = go.Figure(go.Bar(
        x=counts, y=classes,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
 
    fig.update_layout(
        **_base_layout(),
        xaxis_title="Count",
        height=200,
        margin=dict(l=80, r=20, t=10, b=30),
    )
    return fig
 
 
def render_congestion_gauge(congestion_score: float) -> go.Figure:
    """
    Speedometer-style gauge showing congestion level 0–100.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=congestion_score,
        number={"suffix": "%", "font": {"color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": NEUTRAL},
            "bar":  {"color": PRIMARY},
            "steps": [
                {"range": [0,  40],  "color": "#1a2a1a"},   # green zone
                {"range": [40, 70],  "color": "#2a2a1a"},   # amber zone
                {"range": [70, 100], "color": "#2a1a1a"},   # red zone
            ],
            "threshold": {
                "line": {"color": DANGER, "width": 3},
                "thickness": 0.8,
                "value": 70,
            },
        },
    ))
 
    fig.update_layout(
        **_base_layout(),
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig
 
def _base_layout() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CDD9E5", family="Inter, sans-serif", size=12),
        xaxis=dict(
            gridcolor=BORDER, linecolor=BORDER,
            showgrid=True, zeroline=False,
        ),
        yaxis=dict(
            gridcolor=BORDER, linecolor=BORDER,
            showgrid=True, zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BORDER,
        ),
    )
 
 
def _empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=NEUTRAL, size=14),
    )
    fig.update_layout(**_base_layout(), height=200)
    return fig
 