# streamlit_turbine_selection_overlay.py

import math
import streamlit as st
import plotly.graph_objects as go

# -------------------------------
# Constants
# -------------------------------
g = 9.81
rho = 1000.0

st.title("🌊 Turbine Selection with Interactive Overlay")
st.markdown("Move the sliders to place your operating point on the turbine selection chart (Q–H).")

# ---------------------------------------------------
# STEP 1: User Inputs
# ---------------------------------------------------
st.header("1. Define Inputs")

P_target_MW = st.number_input("Power per Turbine (MW)", value=125.0, step=5.0)
H_effective = st.slider("Effective Head H (m)", 1, 2000, 218)
Q_design = st.slider("Design Discharge Q (m³/s)", 1, 1000, 240)

# efficiencies
eta_turbine = st.slider("Turbine Efficiency (η_turbine)", 0.70, 0.98, 0.90)
eta_generator = st.slider("Generator Efficiency (η_generator)", 0.90, 0.99, 0.96)
eta_transformer = st.slider("Transformer Efficiency (η_transformer)", 0.98, 0.995, 0.99)

# ---------------------------------------------------
# STEP 2: Overall Efficiency & Power
# ---------------------------------------------------
eta_total = eta_turbine * eta_generator * eta_transformer
P = rho * g * Q_design * H_effective * eta_total
P_MW = P / 1e6

# ---------------------------------------------------
# STEP 3: Turbine Selection Chart Overlay
# ---------------------------------------------------
st.header("2. Turbine Selection Chart Overlay")

# Background image (uploaded slide)
img_path = "turbine_chart.png"  # <-- rename your uploaded file to this

fig = go.Figure()

# Add background image
fig.add_layout_image(
    dict(
        source="turbine_chart.png",   # will be served from local file
        xref="x", yref="y",
        x=0.1, y=2000,  # bottom-left (Q min, H max)
        sizex=1000, sizey=2000,  # axis span
        sizing="stretch",
        opacity=1,
        layer="below"
    )
)

# Set log-log axes same as the chart
fig.update_xaxes(type="log", range=[-1, 3], title="Discharge Q (m³/s)")
fig.update_yaxes(type="log", range=[0, 3.3], title="Head h (m)")

# Plot student's operating point
fig.add_trace(go.Scatter(
    x=[Q_design], y=[H_effective],
    mode="markers+text",
    text=[f"{P_MW:.1f} MW"],
    textposition="top center",
    marker=dict(size=14, color="red", symbol="circle"),
    name="Operating Point"
))

fig.update_layout(
    width=800, height=600,
    title="Interactive Turbine Selection Map (Overlay on Chart)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# STEP 4: Summary
# ---------------------------------------------------
st.header("3. Summary")
st.markdown(f"""
- **Effective Head (H):** {H_effective:.1f} m  
- **Discharge Q:** {Q_design:.1f} m³/s  
- **Net Power Output:** {P_MW:.1f} MW  
- **Overall Efficiency η_total:** {eta_total:.3f}  
""")
