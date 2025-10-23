#!/usr/bin/env python3
"""
Test script to verify the frame change visualization works
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sample data to test visualization
frame_times = np.linspace(0, 100, 500)  # 0 to 100 seconds
frame_changes = np.random.normal(2, 1, 500)  # Random changes around 2%
frame_changes = np.clip(frame_changes, 0, 10)  # Clip to reasonable range

# Add some peaks to simulate actual capture points
for i in [50, 150, 300, 450]:
    if i < len(frame_changes):
        frame_changes[i] = np.random.uniform(4.5, 8.0)  # Above threshold

# Screenshot times (simulated)
screenshot_times = [10, 25, 45, 60, 75, 90]
screenshot_changes = [frame_changes[int(t*5)] for t in screenshot_times]  # Approximate values

# Create test visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Frame Change Analysis (every 0.2s)",
        "Test 1: Pixel Similarity",
        "Test 2: Row Similarity", 
        "Boolean Duplicate Map"
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
    specs=[[{"type": "scatter"}, {"type": "heatmap"}],
           [{"type": "heatmap"}, {"type": "heatmap"}]]
)

# Frame change plot
fig.add_trace(
    go.Scatter(
        x=frame_times,
        y=frame_changes,
        mode='lines+markers',
        name='Frame Change %',
        line=dict(width=2, color='blue'),
        marker=dict(size=4),
        hovertemplate='Time: %{x:.1f}s<br>Change: %{y:.2f}%<extra></extra>'
    ),
    row=1, col=1
)

# Add threshold lines
fig.add_hline(y=4.0, line_dash="dash", line_color="orange", 
              annotation_text="Capture Threshold (4.0%)")
fig.add_hline(y=40.0, line_dash="dash", line_color="red",
              annotation_text="Major Change Threshold (40.0%)")

# Add screenshot markers
fig.add_trace(
    go.Scatter(
        x=screenshot_times,
        y=screenshot_changes,
        mode='markers',
        name='Screenshots Taken',
        marker=dict(size=8, color='red', symbol='diamond'),
        hovertemplate='Screenshot at %{x:.1f}s<br>Change: %{y:.2f}%<extra></extra>'
    ),
    row=1, col=1
)

# Dummy heatmaps for layout testing
n = 6
test_matrix = np.random.rand(n, n)
labels = [f"{i:02d}_test" for i in range(n)]

for row, col, title in [(1, 2, "Test 1"), (2, 1, "Test 2"), (2, 2, "Duplicates")]:
    fig.add_trace(
        go.Heatmap(
            z=test_matrix,
            x=labels,
            y=labels,
            colorscale='RdYlBu_r' if row != 2 or col != 2 else [[0, 'white'], [1, 'red']],
            showscale=True
        ),
        row=row, col=col
    )

# Update layout
fig.update_layout(
    title="Test Comprehensive Video Analysis Dashboard",
    height=1000,
    width=1400,
    font=dict(size=10),
    showlegend=True,
    margin=dict(l=80, r=80, t=120, b=100)
)

# Improve label spacing
fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
fig.update_yaxes(tickfont=dict(size=8))

# Add statistics annotation
stats_text = (
    f"<b>Test Analysis Statistics:</b><br>"
    f"• Frame change points: {len(frame_times)}<br>"
    f"• Average change: {np.mean(frame_changes):.2f}%<br>"
    f"• Maximum change: {np.max(frame_changes):.2f}%<br>"
    f"• Screenshots captured: {len(screenshot_times)}<br>"
    f"• Threshold exceeded: {np.sum(frame_changes >= 4.0)} times"
)

fig.add_annotation(
    text=stats_text,
    xref="paper", yref="paper",
    x=0.02, y=0.98,
    xanchor="left", yanchor="top",
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="black",
    borderwidth=1,
    font=dict(size=10)
)

# Save test visualization
fig.write_html("test_dashboard.html")
print("✓ Test dashboard saved: test_dashboard.html")
print("✓ Visualization components are working correctly!")