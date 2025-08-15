import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
g = 9.8  # m/s²
rho = 1000  # kg/m³

st.set_page_config(layout="wide")
st.title("Pumped Hydro Storage Penstock Design")

# =====================================
# Section 1: Input Parameters
# =====================================
st.header("1. System Parameters")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Hydraulic Parameters")
    HWL_u = st.number_input("Upper Reservoir HWL (m)", value=1100.0, key="hwl_u")
    LWL_u = st.number_input("Upper Reservoir LWL (m)", value=1080.0, key="lwl_u")
    HWL_l = st.number_input("Lower Reservoir HWL (m)", value=450.0, key="hwl_l")
    TWL_l = st.number_input("Lower Reservoir TWL (m)", value=420.0, key="twl_l")

with col2:
    st.subheader("Penstock Design")
    N_penstocks = st.number_input("Number of Penstocks", min_value=1, max_value=8, value=2, key="n_penstocks")
    eta_t = st.number_input("Turbine Efficiency", value=0.90, min_value=0.7, max_value=1.0, key="eta_t")
    D_pen = st.number_input("Penstock Diameter (m)", value=3.5, key="d_pen")
    design_power = st.number_input("Design Power (MW)", value=500.0, key="p_design")
    max_power = st.number_input("Maximum Power (MW)", value=600.0, key="p_max")

# =====================================
# Head Loss Parameters (Improved with references)
# =====================================
st.subheader("Head Loss Parameters")
col1, col2 = st.columns(2)
with col1:
    L_penstock = st.number_input("Penstock Length (m)", value=500.0, key="l_penstock")
    
    # Improved friction factor input
    f_options = {
        "New steel (welded)": 0.012,
        "New steel (riveted)": 0.017,
        "Concrete (smooth)": 0.015,
        "Concrete (rough)": 0.022,
        "PVC/Plastic": 0.009
    }
    f_material = st.selectbox(
        "Penstock Material",
        options=list(f_options.keys()),
        index=2,  # Default to concrete
        key="f_material"
    )
    f = st.slider(
        "Friction Factor (f)",
        min_value=0.005,
        max_value=0.03,
        value=f_options[f_material],
        step=0.001,
        key="friction"
    )
    st.caption(f"Typical for {f_material}: {f_options[f_material]:.3f}")
    
with col2:
    # Improved K_sum input with reference values
    st.markdown("**Local Loss Coefficients (ΣK)**")
    st.caption("Sum of individual loss coefficients (K-values)")
    
    # Component loss calculator
    components = {
        "Entrance (bellmouth)": 0.15,
        "Entrance (square)": 0.50,
        "90° bend": 0.25,
        "45° bend": 0.15,
        "Gate valve": 0.20,
        "Butterfly valve": 0.30,
        "T-junction": 0.40,
        "Exit": 1.00
    }
    
    K_sum = 0.0
    st.markdown("**Add Components:**")
    for comp, k_val in components.items():
        if st.checkbox(comp, value=(comp in ["Entrance (bellmouth)", "90° bend", "Exit"])):
            K_sum += k_val
            st.caption(f"+ {k_val} for {comp} (current ΣK = {K_sum:.2f})")
    
    # Display total K_sum
    st.markdown(f"**Total ΣK = {K_sum:.2f}**")
    st.caption("Typical range: 2.0-5.0 for well-designed systems")
    
    auto_hf = st.checkbox("Calculate losses automatically", value=True, key="auto_hf")

# Initialize variables with default values
hf_design = 25.0  
hf_max = 40.0

if auto_hf:
    # First calculate with default head losses to get initial Q estimates
    h_net_design_temp = (HWL_u - TWL_l) - hf_design
    h_net_min_temp = (LWL_u - HWL_l) - hf_max
    
    Q_design_total_temp = (design_power * 1e6) / (rho * g * h_net_design_temp * eta_t)
    Q_max_total_temp = (max_power * 1e6) / (rho * g * h_net_min_temp * eta_t)
    Q_design_temp = Q_design_total_temp / N_penstocks
    Q_max_temp = Q_max_total_temp / N_penstocks
    
    # Now calculate velocities
    v_design = Q_design_temp / (math.pi*(D_pen/2)**2)
    v_max = Q_max_temp / (math.pi*(D_pen/2)**2)
    
    # Finally calculate head losses
    hf_design = (f * L_penstock/D_pen + K_sum) * (v_design**2)/(2*g)
    hf_max = (f * L_penstock/D_pen + K_sum) * (v_max**2)/(2*g)
    
    st.write(f"Calculated Design Head Loss: {hf_design:.2f} m")
    st.write(f"Calculated Max Head Loss: {hf_max:.2f} m")
else:
    hf_design = st.number_input("Design Head Loss (m)", value=25.0, key="hf_design")
    hf_max = st.number_input("Max Head Loss (m)", value=40.0, key="hf_max")

# =====================================
# Section 2: Design Equations (as LaTeX in Streamlit)
# =====================================
st.header("Design Equations")

with st.expander("Show Design Equations"):
    st.markdown("""
    ### Total Discharge Calculation
    $$
    Q_{\\text{total}} = \\frac{P \\times 10^6}{\\rho \\cdot g \\cdot h_{\\text{net}} \\cdot \\eta_t}
    $$
    **Where:**
    - $P$: Power (MW)
    - $h_{\\text{net}} = \\Delta H - h_f$: Net head (Gross head $\\Delta H$ minus head loss $h_f$) (m)
    - $\\rho$: Water density (1000 kg/m³)
    - $g$: Gravitational acceleration (9.8 m/s²)
    - $\\eta_t$: Turbine efficiency (e.g., 0.85 for 85%)
    
    ### Per-Penstock Discharge
    $$
    Q_{\\text{penstock}} = \\frac{Q_{\\text{total}}}{N_{\\text{penstocks}}}
    $$
    **Where:**
    - $N_{\\text{penstocks}}$: Number of penstocks
    
    ### Flow Velocity Validation
    $$
    v = \\frac{Q_{\\text{penstock}}}{A} = \\frac{4 Q_{\\text{penstock}}}{\\pi D^2}
    $$
    **Industry Standard (USBR):**
    - Recommended velocity range: **4–6 m/s** (balances efficiency and material erosion)
    
    ### Head Loss (Darcy-Weisbach)
    $$
    h_f = \\frac{f L v^2}{D \\cdot 2g}
    $$
    **Where:**
    - $f$: Friction factor (0.015 for concrete)
    - $L$: Penstock length (m)
    - $D$: Penstock diameter (m)
    - $v$: Flow velocity (m/s)
    """)

# =====================================
# Section 8: Friction & Loss Coefficients Reference
# =====================================
st.header("Reference Tables")

with st.expander("📚 Friction Factors & Loss Coefficients"):
    st.subheader("Friction Factors (f) - Darcy-Weisbach Equation")
    friction_data = {
        "Material": ["New steel (welded)", 
                    "New steel (riveted)", 
                    "Concrete (smooth)", 
                    "Concrete (rough)", 
                    "PVC/Plastic"],
        "Friction Factor Range": ["0.010 - 0.015", 
                                 "0.015 - 0.020", 
                                 "0.012 - 0.018", 
                                 "0.018 - 0.025", 
                                 "0.007 - 0.012"],
        "Source": ["ASCE (2017)", 
                   "USBR (1987)", 
                   "ACI 351.3R (2018)", 
                   "USACE (2008)", 
                   "AWWA (2012)"]
    }
    st.table(pd.DataFrame(friction_data))

    st.subheader("Typical Local Loss Coefficients (K)")
    loss_data = {
        "Component": ["Entrance (bellmouth)", 
                     "Entrance (square)", 
                     "90° bend", 
                     "45° bend", 
                     "Gate valve (open)", 
                     "Butterfly valve (open)", 
                     "T-junction", 
                     "Exit"],
        "K-value Range": ["0.1 - 0.2", 
                         "0.4 - 0.5", 
                         "0.2 - 0.3", 
                         "0.1 - 0.2", 
                         "0.1 - 0.3", 
                         "0.2 - 0.4", 
                         "0.3 - 0.5", 
                         "0.8 - 1.0"],
        "Notes": ["Best case entrance", 
                 "Worst case entrance", 
                 "Depends on radius/diameter ratio", 
                 "Gentler than 90°", 
                 "Varies with design", 
                 "Position dependent", 
                 "Flow splitting", 
                 "All kinetic energy lost"]
    }
    st.table(pd.DataFrame(loss_data))

    st.markdown("""
    **Design Recommendations:**
    - For concrete penstocks: Use f = 0.015-0.020
    - ΣK typically ranges 2.0-5.0 for well-designed systems
    - Add 10-20% safety margin for aging effects
    - Reduce bends and use bellmouth entrances to minimize losses
    """)
# =====================================
# Section 3: Calculations
# =====================================
st.header("2. Calculation Results")

# Net heads
h_net_design = (HWL_u - TWL_l) - hf_design
h_net_min = (LWL_u - HWL_l) - hf_max

# Discharges
Q_design_total = (design_power * 1e6) / (rho * g * h_net_design * eta_t)
Q_max_total = (max_power * 1e6) / (rho * g * h_net_min * eta_t)
Q_design = Q_design_total / N_penstocks
Q_max = Q_max_total / N_penstocks

# Velocities
A_pen = math.pi * (D_pen/2)**2
v_design = Q_design / A_pen
v_max = Q_max / A_pen

# =====================================
# Section 4: Results Display
# =====================================
results = pd.DataFrame({
    "Parameter": ["Total System", "Per Penstock"],
    "Design Discharge (m³/s)": [Q_design_total, Q_design],
    "Max Discharge (m³/s)": [Q_max_total, Q_max],
    "Design Velocity (m/s)": [Q_design_total/A_pen, v_design],
    "Max Velocity (m/s)": [Q_max_total/A_pen, v_max]
})

st.dataframe(results.style.format({
    "Design Discharge (m³/s)": "{:.2f}",
    "Max Discharge (m³/s)": "{:.2f}",
    "Design Velocity (m/s)": "{:.2f}",
    "Max Velocity (m/s)": "{:.2f}"
}), use_container_width=True)

# =====================================
# Section 5: Velocity Validation
# =====================================
st.header("3. Velocity Validation")

col1, col2 = st.columns(2)
with col1:
    st.metric("Design Velocity", f"{v_design:.2f} m/s")
    st.metric("Max Velocity", f"{v_max:.2f} m/s")

with col2:
    st.markdown("### USBR Standards")
    st.markdown("- **Recommended range:** 4-6 m/s")
    st.markdown("- **Absolute maximum:** 7 m/s (short durations)")

if v_max > 7.0:
    st.error("""
    ⚠️ **Dangerous Velocity** (Exceeds all standards)
    - Immediate risk of cavitation and erosion
    - Urgent design modification required
    """)
elif v_max > 6.0:
    st.warning("""
    ⚠️ **Above Recommended Limit** (ASME/ICOLD)
    - Acceptable for short durations (<1 hour/day)
    - Consider for emergency operations only
    """)
elif v_max > 4.0:
    st.success("""
    ✓ **Optimal Range** (4–6 m/s)
    - Meets USBR standards for concrete
    - Good balance of efficiency and safety
    """)
else:
    st.info("""
    ℹ️ **Low Velocity** (<4 m/s)
    - Safe but potentially uneconomic
    - Consider smaller diameter for cost savings
    """)

# =====================================
# Section 6: Interactive System Curves
# =====================================
st.header("4. System Characteristics")

# First calculate Q_range based on system parameters
Q_range = np.linspace(0, Q_max_total*1.2, 100)  # Generate discharge range
h_net_range = h_net_design - (h_net_design - h_net_min) * (Q_range/Q_max_total)**2
P_range = N_penstocks * (rho * g * (Q_range/N_penstocks) * h_net_range * eta_t) / 1e6

# Now create the interactive plot
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Main power curve (left axis)
fig.add_trace(
    go.Scatter(
        x=Q_range,
        y=P_range,
        name="Power Output",
        line=dict(color="blue", width=3),
        hovertemplate="<b>%{x:.1f} m³/s</b><br>Power: %{y:.1f} MW",
    ),
    secondary_y=False,
)

# Add efficiency curve (right axis)
fig.add_trace(
    go.Scatter(
        x=Q_range,
        y=(P_range*1e6)/(rho*g*(Q_range/N_penstocks))*100,
        name="System Efficiency (%)",
        line=dict(color="purple", width=2, dash="dot"),
        hovertemplate="<b>%{x:.1f} m³/s</b><br>Eff: %{y:.1f}%",
        visible="legendonly"  # Hidden by default
    ),
    secondary_y=True,
)

# Add reference lines
fig.add_vline(
    x=Q_design_total,
    line=dict(color="green", dash="dash", width=2),
    annotation=dict(text="Design", xanchor="left"),
    name="Design Discharge"
)

fig.add_vline(
    x=Q_max_total,
    line=dict(color="red", dash="dash", width=2),
    annotation=dict(text="Max", xanchor="left"),
    name="Max Discharge"
)

# Add velocity markers
velocity_markers = [4, 6, 7]  # USBR standards
for v in velocity_markers:
    Q_v = v * (math.pi * (D_pen**2)/4) * N_penstocks
    fig.add_vline(
        x=Q_v,
        line=dict(color="orange", width=1, dash="dot"),
        annotation=dict(text=f"{v} m/s", yanchor="bottom"),
        name=f"{v} m/s velocity"
    )

# Layout configuration
fig.update_layout(
    title="System Operating Characteristics",
    xaxis_title="Total System Discharge (m³/s)",
    yaxis_title="Power Output (MW)",
    yaxis2_title="Efficiency (%)",
    hovermode="x unified",
    width=800,
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, b=50, t=80),
)

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1x", step="all"),
                dict(count=2, label="2x", step="all"),
                dict(step="all")
            ])
        )
    )
)

# Display the interactive plot
st.plotly_chart(fig, use_container_width=True)

# =====================================
# Additional Controls
# =====================================
with st.expander("⚙️ Chart Customization"):
    col1, col2 = st.columns(2)
    with col1:
        show_efficiency = st.checkbox("Show Efficiency Curve", value=False)
        show_velocity = st.checkbox("Show Velocity Markers", value=True)
    with col2:
        log_scale = st.checkbox("Logarithmic Scale (X-axis)")
    
    # Update visibility based on controls
    fig.update_traces(
        selector={"name": "System Efficiency (%)"},
        visible=show_efficiency
    )
    for v in velocity_markers:
        fig.update_traces(
            selector={"name": f"{v} m/s velocity"},
            visible=show_velocity
        )
    if log_scale:
        fig.update_xaxes(type="log")
    
    st.plotly_chart(fig, use_container_width=True)

# =====================================
# Operating Point Analysis
# =====================================
st.subheader("Operating Point Analysis")

selected_Q = st.slider(
    "Select Discharge (m³/s)",
    min_value=0.0,
    max_value=float(Q_max_total*1.2),
    value=float(Q_design_total),
    step=0.1
)

# Calculate values at selected point
idx = np.abs(Q_range - selected_Q).argmin()
P_selected = P_range[idx]
h_net_selected = h_net_range[idx]
v_selected = (selected_Q/N_penstocks)/(math.pi*(D_pen/2)**2)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Power Output", f"{P_selected:.1f} MW")
with col2:
    st.metric("Net Head", f"{h_net_selected:.1f} m")
with col3:
    st.metric("Flow Velocity", f"{v_selected:.1f} m/s", 
              delta="Above limit" if v_selected > 6 else None)

# =====================================
# Section 7: References
# =====================================
st.header("5. References")
st.markdown("""
- **USBR Design Standards No. 3:** Hydropower Penstock Design Guidelines
- **ICOLD Bulletins:** Penstock and Tunnel Design Recommendations
- **ASME Hydropower Standards:** Mechanical Design of Hydropower Systems
- **ASCE Guidelines:** Friction Factors for Penstocks
- **USACE Engineering Manuals:** Hydraulic Loss Coefficients
""")

