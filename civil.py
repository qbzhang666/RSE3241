# PHES / Hydropower Design Teaching App (Snowy 2.0 & Kidston)
# Enhanced version with improved UI, calculations, and visualization

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from io import BytesIO
import base64

# ======================== CONSTANTS & PHYSICS ========================
G = 9.80665  # Standard gravity (m/s²)
RHO = 997     # Water density at 25°C (kg/m³)
ATM_PRESSURE = 101.325  # kPa
VAP_PRESSURE = 3.169    # kPa at 25°C

# ======================== ENGINEERING FUNCTIONS ========================
def power_output(Q, h_net, eta):
    """Calculate power output in MW"""
    return RHO * G * Q * h_net * eta / 1e6

def discharge_required(P, h_net, eta):
    """Calculate required discharge for power output"""
    return P * 1e6 / (RHO * G * h_net * eta)

def head_loss(L, D, f, K, Q):
    """Calculate head loss using Darcy-Weisbach equation"""
    A = math.pi * (D/2)**2
    v = Q / A
    return (f * L/D + K) * (v**2) / (2 * G)

def thoma_sigma(H_atm, H_vap, submergence, h_loss, H_net):
    """Calculate Thoma cavitation coefficient"""
    return (H_atm - H_vap + submergence - h_loss) / H_net

# ======================== STREAMLIT CONFIG ========================
st.set_page_config(
    page_title="Advanced PHES Design Simulator",
    layout="wide",
    page_icon="💧"
)

st.title("🏔️ Pumped Hydro Energy Storage Design Simulator")
st.markdown("""
**Explore PHES design principles using Snowy 2.0 and Kidston as case studies**  
*Sections 5-9: Hydraulic Design, Pressure Tunnels, Cavitation Analysis*
""")

# ======================== SIDEBAR CONTROLS ========================
with st.sidebar:
    st.header("⚙️ Project Configuration")
    project = st.selectbox("Select Project", ["Snowy 2.0", "Kidston PHES", "Custom Design"])
    
    # Project presets
    if project == "Snowy 2.0":
        params = {
            "H_upper": 1200, "H_lower": 500, "head_diff": 700,
            "tunnel_length": 27000, "penstock_dia": 10.5,
            "power_capacity": 2000, "units": 6
        }
    elif project == "Kidston PHES":
        params = {
            "H_upper": 495, "H_lower": 320, "head_diff": 175,
            "tunnel_length": 2300, "penstock_dia": 4.2,
            "power_capacity": 250, "units": 2
        }
    else:  # Custom
        params = {
            "H_upper": 800, "H_lower": 400, "head_diff": 400,
            "tunnel_length": 5000, "penstock_dia": 6.0,
            "power_capacity": 500, "units": 4
        }
    
    # Adjustable parameters
    st.subheader("Design Parameters")
    h_gross = st.slider("Gross Head (m)", 50, 1000, params["head_diff"], 10)
    tunnel_length = st.number_input("Tunnel Length (m)", 500, 50000, params["tunnel_length"])
    penstock_dia = st.number_input("Penstock Diameter (m)", 1.0, 15.0, params["penstock_dia"], 0.1)
    power_capacity = st.number_input("Rated Power (MW)", 10, 3000, params["power_capacity"], 10)
    num_units = st.number_input("Number of Units", 1, 12, params["units"])
    efficiency = st.slider("Turbine Efficiency", 0.75, 0.95, 0.90, 0.01)

# ======================== MAIN CALCULATIONS ========================
# Hydraulic calculations
Q_design = discharge_required(power_capacity, h_gross, efficiency) / num_units
flow_velocity = Q_design / (math.pi * (penstock_dia/2)**2)

# Head loss calculations
friction_factor = 0.015  # Concrete-lined tunnel
local_losses = 1.2       # Entrance + bends + valves
h_loss = head_loss(tunnel_length, penstock_dia, friction_factor, local_losses, Q_design)
h_net = h_gross - h_loss

# Cavitation analysis
submergence = 10  # Runner below tailwater (m)
thoma = thoma_sigma(ATM_PRESSURE/9.80665, VAP_PRESSURE/9.80665, submergence, 2.0, h_net)

# ======================== VISUALIZATION ========================
st.header("📊 System Performance")
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "scatter"}]],
                    subplot_titles=("Energy Balance", "Head vs. Flow"))

# Energy balance chart
fig.add_trace(go.Bar(
    name="Gross Head",
    x=["Head Components"],
    y=[h_gross],
    marker_color="#1f77b4"
), row=1, col=1)

fig.add_trace(go.Bar(
    name="Head Loss",
    x=["Head Components"],
    y=[h_loss],
    marker_color="#ff7f0e"
), row=1, col=1)

fig.add_trace(go.Bar(
    name="Net Head",
    x=["Head Components"],
    y=[h_net],
    marker_color="#2ca02c"
), row=1, col=1)

# Head-flow curve
flow_range = np.linspace(0, Q_design*1.5, 50)
head_net_range = h_gross - head_loss(tunnel_length, penstock_dia, friction_factor, local_losses, flow_range)

fig.add_trace(go.Scatter(
    x=flow_range,
    y=head_net_range,
    mode="lines",
    name="Net Head",
    line=dict(color="#9467bd", width=3)
), row=1, col=2)

fig.add_trace(go.Scatter(
    x=[Q_design],
    y=[h_net],
    mode="markers",
    name="Design Point",
    marker=dict(color="red", size=10)
), row=1, col=2)

# Update layout
fig.update_layout(
    height=500,
    showlegend=True,
    barmode="stack",
    xaxis_title="Flow Rate (m³/s)",
    yaxis_title="Head (m)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ======================== ENGINEERING ANALYSIS ========================
st.header("⚙️ Engineering Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Hydraulics")
    st.metric("Design Flow per Unit", f"{Q_design:.1f} m³/s")
    st.metric("Flow Velocity", f"{flow_velocity:.1f} m/s", 
              "Good" if 4 <= flow_velocity <= 6 else "High" if flow_velocity > 6 else "Low")
    st.metric("Head Loss", f"{h_loss:.1f} m ({h_loss/h_gross:.1%})")

with col2:
    st.subheader("Structural")
    pressure_head = h_gross * 0.8  # 80% of max head
    st.metric("Max Internal Pressure", f"{pressure_head/100:.1f} MPa")
    
    # Steel lining stress calculation
    thickness = 0.05 * penstock_dia
    hoop_stress = (pressure_head * RHO * G * penstock_dia) / (2 * thickness)
    st.metric("Hoop Stress in Lining", f"{hoop_stress/1e6:.1f} MPa", 
              "Within limits" if hoop_stress/1e6 < 200 else "High")

with col3:
    st.subheader("Cavitation")
    st.metric("Thoma Coefficient", f"{thoma:.3f}")
    st.metric("Recommended Safety Margin", "0.10", 
              "Adequate" if thoma > 0.10 else "Inadequate")
    st.metric("Runner Submergence", f"{submergence} m")

# ======================== PRESSURE TUNNEL ANALYSIS ========================
st.header("🏔️ Pressure Tunnel Design")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Rock Cover Analysis")
    overburden = st.slider("Overburden Thickness (m)", 10, 500, 100, 10)
    rock_density = st.slider("Rock Density (kg/m³)", 2000, 3000, 2650, 50)
    
    # Minimum cover calculation
    min_cover = (1.5 * pressure_head * RHO) / rock_density
    st.metric("Minimum Rock Cover Required", f"{min_cover:.1f} m")
    st.metric("Safety Factor", f"{overburden/min_cover:.2f}",
              "Adequate" if overburden/min_cover > 1.2 else "Inadequate")
    
    st.subheader("Lining Design")
    lining_type = st.selectbox("Lining Type", ["Concrete", "Steel", "Shotcrete"])
    thickness = st.slider("Lining Thickness (mm)", 100, 1000, 300, 50)

with col2:
    # Tunnel cross-section visualization
    fig = go.Figure()
    
    # Rock boundary
    fig.add_shape(type="circle", x0=-5, y0=-5, x1=5, y1=5, 
                 line=dict(color="saddlebrown", width=2), fillcolor="peru", opacity=0.3)
    
    # Tunnel
    tunnel_radius = penstock_dia/2
    fig.add_shape(type="circle", x0=-tunnel_radius, y0=-tunnel_radius, 
                 x1=tunnel_radius, y1=tunnel_radius,
                 line=dict(color="royalblue", width=3), fillcolor="lightblue", opacity=0.5)
    
    # Lining
    lining_radius = tunnel_radius - thickness/1000
    fig.add_shape(type="circle", x0=-lining_radius, y0=-lining_radius, 
                 x1=lining_radius, y1=lining_radius,
                 line=dict(color="darkred", width=2, dash="dot"), 
                 fillcolor="rgba(205,92,92,0.2)")
    
    # Overburden
    fig.add_shape(type="rect", x0=-6, y0=5, x1=6, y1=10, 
                 line=dict(width=0), fillcolor="forestgreen", opacity=0.4)
    
    fig.update_layout(
        title="Tunnel Cross-Section",
        xaxis=dict(visible=False, range=[-10, 10]),
        yaxis=dict(visible=False, range=[-10, 10]),
        height=450,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================== RESULTS SUMMARY ========================
st.header("📝 Design Summary")

results = {
    "Parameter": [
        "Gross Head", "Net Head", "Head Loss", 
        "Total Discharge", "Discharge per Unit", "Flow Velocity",
        "Power Output", "Efficiency",
        "Internal Pressure", "Hoop Stress",
        "Thoma Coefficient", "Safety Margin"
    ],
    "Value": [
        f"{h_gross} m", f"{h_net:.1f} m", f"{h_loss:.1f} m",
        f"{Q_design * num_units:.1f} m³/s", f"{Q_design:.1f} m³/s", f"{flow_velocity:.1f} m/s",
        f"{power_capacity} MW", f"{efficiency*100:.1f}%",
        f"{pressure_head/100:.2f} MPa", f"{hoop_stress/1e6:.1f} MPa",
        f"{thoma:.3f}", f"{(thoma - 0.10):.3f}" if thoma > 0.10 else "Insufficient"
    ],
    "Status": [
        "Input", "Calculated", "Good" if h_loss/h_gross < 0.15 else "High",
        "Calculated", "Calculated", "✅ Optimal" if 4 <= flow_velocity <= 6 else "⚠️ Check",
        "Input", "Input",
        "Calculated", "✅ Safe" if hoop_stress/1e6 < 200 else "⚠️ Review",
        "Calculated", "✅ Adequate" if thoma > 0.10 else "⚠️ Insufficient"
    ]
}

df_results = pd.DataFrame(results)
st.dataframe(df_results, hide_index=True, use_container_width=True)

# ======================== DOWNLOAD REPORT ========================
st.header("📤 Export Results")

# Create PDF report (simulated)
report = f"""
PHES DESIGN REPORT
==================

Project: {project}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

DESIGN PARAMETERS
-----------------
Gross Head: {h_gross} m
Tunnel Length: {tunnel_length} m
Penstock Diameter: {penstock_dia} m
Rated Power: {power_capacity} MW
Number of Units: {num_units}
Turbine Efficiency: {efficiency*100:.1f}%

HYDRAULIC RESULTS
-----------------
Design Flow per Unit: {Q_design:.1f} m³/s
Flow Velocity: {flow_velocity:.1f} m/s
Head Loss: {h_loss:.1f} m ({h_loss/h_gross:.1%})
Net Head: {h_net:.1f} m

ENGINEERING ANALYSIS
-------------------
Max Internal Pressure: {pressure_head/100:.2f} MPa
Hoop Stress: {hoop_stress/1e6:.1f} MPa
Thoma Cavitation Coefficient: {thoma:.3f}
Rock Cover Safety Factor: {overburden/min_cover:.2f}
"""

# Download buttons
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="Download Report (TXT)",
        data=report,
        file_name="phes_design_report.txt",
        mime="text/plain"
    )
    
with col2:
    # Excel download
    excel_file = BytesIO()
    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="Design Summary", index=False)
        
        # Add calculations sheet
        calc_data = pd.DataFrame({
            "Parameter": ["Water Density", "Gravity", "Friction Factor", "Local Losses"],
            "Value": [RHO, G, friction_factor, local_losses],
            "Unit": ["kg/m³", "m/s²", "-", "-"]
        })
        calc_data.to_excel(writer, sheet_name="Constants", index=False)
    
    st.download_button(
        label="Download Data (Excel)",
        data=excel_file.getvalue(),
        file_name="phes_design_data.xlsx",
        mime="application/vnd.ms-excel"
    )

# ======================== FOOTER ========================
st.markdown("---")
st.caption("""
**PHES Design Teaching App** | Developed for Renewable Energy Engineering  
*Based on Snowy 2.0 (NSW, Australia) and Kidston PHES (QLD, Australia) case studies*
""")
