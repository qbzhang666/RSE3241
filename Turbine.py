# streamlit_turbine_selection_overlay_embedded.py

import math
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# -------------------------------
# Constants
# -------------------------------
g = 9.81       # m/s²
rho = 1000.0   # kg/m³

st.title("🌊 Turbine Selection & Energy Generation")
st.markdown("This app overlays the turbine selection chart with your design operating point.")

# ---------------------------------------------------
# STEP 1: Load Embedded Chart
# ---------------------------------------------------
# Make sure the PNG file is inside your repo (e.g., "turbine_chart.png")
# Replace with the correct filename in your /src folder or working directory
chart_path = "66a71bbc-7083-4c5a-b5f8-1885a40896f4.png"
img = Image.open(chart_path)

# ---------------------------------------------------
# STEP 2: User Inputs
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
# STEP 3: Efficiency & Power
# ---------------------------------------------------
st.header("2. Efficiency and Power")
eta_total = eta_turbine * eta_generator * eta_transformer
P = rho * g * Q_design * H_effective * eta_total
P_MW = P / 1e6
st.write(f"⚙️ **Overall Efficiency η_total = {eta_total:.3f}**")
st.success(f"Net Power Output = {P_MW:.1f} MW")

# ---------------------------------------------------
# STEP 4: Turbine Selection Chart Overlay
# ---------------------------------------------------
st.header("3. Turbine Selection Chart (Overlay)")

fig = go.Figure()

# Add chart image as background
fig.add_layout_image(
    dict(
        source=img,
        xref="x", yref="y",
        x=0.1, y=2000,    # Align bottom-left of image
        sizex=1000, sizey=2000,
        sizing="stretch",
        opacity=1,
        layer="below"
    )
)

# Log scale axes to match chart
fig.update_xaxes(type="log", range=[-1, 3], title="Discharge Q (m³/s)")
fig.update_yaxes(type="log", range=[0, 3.3], title="Head h (m)")

# Add operating point
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
    title="Interactive Turbine Selection Map",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# STEP 5: Summary
# ---------------------------------------------------
st.header("4. Summary")
st.markdown(f"""
- **Effective Head (H):** {H_effective:.1f} m  
- **Discharge Q:** {Q_design:.1f} m³/s  
- **Net Power Output:** {P_MW:.1f} MW  
- **Overall Efficiency η_total:** {eta_total:.3f}  
""")
