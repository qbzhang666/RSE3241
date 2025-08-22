# streamlit_turbine_selection_kidston.py

import os
import math
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# -------------------------------
# Constants
# -------------------------------
g = 9.81
rho = 1000.0

st.title("🌊 Turbine Selection Guide (RSE3241 Week 8)")
st.markdown("""
This app overlays the classical turbine selection chart with your design point.  
We also use **Kidston PHES** as a case study.
""")

# ---------------------------------------------------
# STEP 1: Load Embedded Chart
# ---------------------------------------------------
chart_path = os.path.join("assets", "turbine_chart.png")  # store the image in /assets/

if not os.path.exists(chart_path):
    st.error("❌ Could not find turbine chart image. Please add it to /assets/ in your repo.")
else:
    img = Image.open(chart_path)

# ---------------------------------------------------
# STEP 2: User Inputs
# ---------------------------------------------------
st.header("1. Define Inputs")

P_target_MW = st.number_input("Power per Turbine (MW)", value=125.0, step=5.0)
H = st.slider("Effective Head H (m)", 1, 2000, 218)  # Kidston default
Q = st.slider("Design Discharge Q (m³/s)", 1, 1000, 240)  # Kidston default

# efficiencies
eta_turbine = st.slider("Turbine Efficiency (η_turbine)", 0.70, 0.98, 0.90)
eta_generator = st.slider("Generator Efficiency (η_generator)", 0.90, 0.99, 0.96)
eta_transformer = st.slider("Transformer Efficiency (η_transformer)", 0.98, 0.995, 0.99)

# ---------------------------------------------------
# STEP 3: Efficiency & Power
# ---------------------------------------------------
st.header("2. Efficiency and Power")

eta_total = eta_turbine * eta_generator * eta_transformer
P = rho * g * Q * H * eta_total
P_MW = P / 1e6

st.write(f"⚙️ **Overall Efficiency η_total = {eta_total:.3f}**")
st.success(f"Net Power Output = {P_MW:.1f} MW")

# ---------------------------------------------------
# STEP 4: Turbine Recommendation
# ---------------------------------------------------
st.header("3. Turbine Type Recommendation")

if H > 300 and Q < 200:
    turbine = "Pelton"
    rationale = "High head, low discharge → Pelton (impulse)."
elif 30 < H <= 300 and 20 <= Q <= 700:
    turbine = "Francis"
    rationale = "Medium head and medium discharge → Francis (reaction)."
elif H <= 30 and Q >= 50:
    turbine = "Kaplan"
    rationale = "Low head, high discharge → Kaplan (propeller)."
else:
    turbine = "Check ranges"
    rationale = "Outside textbook ranges. Needs detailed study."

st.success(f"Recommended Turbine: **{turbine}**")
st.write(rationale)

# ---------------------------------------------------
# STEP 5: Overlay Turbine Chart
# ---------------------------------------------------
st.header("4. Turbine Selection Chart (Overlay)")

if os.path.exists(chart_path):
    fig = go.Figure()

    # Add chart image aligned with log-log axes
    fig.add_layout_image(
        dict(
            source=img,
            xref="x", yref="y",
            x=0.1, y=2000,
            sizex=1000, sizey=2000,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Log-log axes
    fig.update_xaxes(type="log", range=[-1, 3], title="Discharge Q (m³/s)")
    fig.update_yaxes(type="log", range=[0, 3.3], title="Head H (m)")

    # Add operating point
    fig.add_trace(go.Scatter(
        x=[Q], y=[H],
        mode="markers+text",
        text=[f"{P_MW:.1f} MW\n({turbine})"],
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
# STEP 6: Summary
# ---------------------------------------------------
st.header("5. Summary")
st.markdown(f"""
- **Effective Head (H):** {H:.1f} m  
- **Discharge Q:** {Q:.1f} m³/s  
- **Net Power Output:** {P_MW:.1f} MW  
- **Overall Efficiency η_total:** {eta_total:.3f}  
- **Recommended Turbine:** {turbine}  
""")

# ---------------------------------------------------
# STEP 7: Kidston Case Study
# ---------------------------------------------------
st.header("6. 📍 Case Study: Kidston PHES")
st.markdown("""
- **Head (H):** ~218 m  
- **Discharge (Q):** ~240 m³/s  
- **Installed Capacity:** ~250 MW  
- **Turbine Type:** **Francis (reaction)**, suitable for medium head & medium flow.  

Kidston is an ideal example where the theoretical chart directly matches real-world design.
""")
