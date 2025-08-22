# streamlit_turbine_selection.py

import os
import math
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# -------------------------------
# Constants
# -------------------------------
g = 9.81       # m/s²
rho = 1000.0   # kg/m³

st.title("🌊 Turbine Selection & Specific Speed Calculator")
st.markdown("""
This app overlays the classical Francis turbine sizing chart with your design operating point  
and calculates the **specific speed (Ns)**.  
We also use **Kidston PHES** as a case study.
""")

# ---------------------------------------------------
# STEP 1: Load Chart Image
# ---------------------------------------------------
chart_path = os.path.join("assets", "francis_chart.png")  # <-- save uploaded chart here

if not os.path.exists(chart_path):
    st.error(f"❌ Could not find {chart_path}. Please check that the file exists in your repo.")
else:
    img = Image.open(chart_path)

# ---------------------------------------------------
# STEP 2: User Inputs
# ---------------------------------------------------
st.header("1. Define Inputs")

P_target_MW = st.number_input("Power per Turbine (MW)", value=125.0, step=5.0)
H_effective = st.slider("Effective Head H (m)", 10, 2000, 218)
Q_design = st.slider("Design Discharge Q (m³/s)", 1, 1000, 240)
N_rpm = st.slider("Rotational Speed N (rpm)", 50, 1500, 300)

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
P_kW = P / 1e3
st.write(f"⚙️ **Overall Efficiency η_total = {eta_total:.3f}**")
st.success(f"Net Power Output = {P_MW:.1f} MW")

# ---------------------------------------------------
# STEP 4: Specific Speed Calculation
# ---------------------------------------------------
st.header("3. Specific Speed Ns")

# convert rpm → rps
N_rps = N_rpm / 60.0

if H_effective > 0:
    Ns = (N_rpm * math.sqrt(P_kW)) / (H_effective ** 1.25)
    st.info(f"🔹 Specific Speed Ns = {Ns:.1f}")
else:
    st.error("Head must be greater than zero to compute Ns.")

# ---------------------------------------------------
# STEP 5: Turbine Selection Chart Overlay
# ---------------------------------------------------
st.header("4. Turbine Selection Chart (Overlay)")

if os.path.exists(chart_path):
    fig = go.Figure()

    # Add chart image as background aligned with log-log axes
    fig.add_layout_image(
        dict(
            source=img,
            xref="x", yref="y",
            x=1, y=1000,          # bottom-left of chart
            sizex=1000, sizey=1000,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Log scale axes
    fig.update_xaxes(type="log", range=[0, 3], title="Discharge Q (m³/s)")
    fig.update_yaxes(type="log", range=[1, 3], title="Head H (m)")

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
        title="Francis Turbine Sizing Chart with Operating Point",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# STEP 6: Summary
# ---------------------------------------------------
st.header("5. Summary")
st.markdown(f"""
- **Effective Head (H):** {H_effective:.1f} m  
- **Discharge Q:** {Q_design:.1f} m³/s  
- **Net Power Output:** {P_MW:.1f} MW  
- **Overall Efficiency η_total:** {eta_total:.3f}  
- **Rotational Speed N:** {N_rpm} rpm  
- **Specific Speed Ns:** {Ns:.1f}  
""")
