import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Constants
g = 9.81  # m/s²
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
# Head Loss Parameters
# =====================================
st.subheader("Head Loss Parameters")
col1, col2 = st.columns(2)
with col1:
    L_penstock = st.number_input("Penstock Length (m)", value=500.0, key="l_penstock")
    f = st.number_input("Friction Factor", value=0.015, min_value=0.01, max_value=0.03, key="friction")
with col2:
    K_sum = st.number_input("Total Local Loss Coefficients (ΣK)", value=4.5, key="k_sum")
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
st.header("2. Design Equations")

with st.expander("Show Design Equations"):
    st.markdown("""
    ### 2.1 Total Discharge Calculation
    $$
    Q_{\\text{total}} = \\frac{P \\times 10^6}{\\rho \\cdot g \\cdot h_{\\text{net}} \\cdot \\eta_t}
    $$
    **Where:**
    - $P$: Power (MW)
    - $h_{\\text{net}} = \\Delta H - h_f$: Net head (Gross head $\\Delta H$ minus head loss $h_f$) (m)
    - $\\rho$: Water density (1000 kg/m³)
    - $g$: Gravitational acceleration (9.81 m/s²)
    - $\\eta_t$: Turbine efficiency (e.g., 0.85 for 85%)
    
    ### 2.2 Per-Penstock Discharge
    $$
    Q_{\\text{penstock}} = \\frac{Q_{\\text{total}}}{N_{\\text{penstocks}}}
    $$
    **Where:**
    - $N_{\\text{penstocks}}$: Number of penstocks
    
    ### 2.3 Flow Velocity Validation
    $$
    v = \\frac{Q_{\\text{penstock}}}{A} = \\frac{4 Q_{\\text{penstock}}}{\\pi D^2}
    $$
    **Industry Standard (USBR):**
    - Recommended velocity range: **4–6 m/s** (balances efficiency and material erosion)
    
    ### 2.4 Head Loss (Darcy-Weisbach)
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
# Section 3: Calculations
# =====================================
st.header("3. Calculation Results")

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
st.header("4. Velocity Validation")

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
# Section 6: System Curves
# =====================================
st.header("5. System Characteristics")

# Generate operating curve
Q_range = np.linspace(0, Q_max_total*1.2, 100)
h_net_range = h_net_design - (h_net_design - h_net_min) * (Q_range/Q_max_total)**2
P_range = N_penstocks * (rho * g * (Q_range/N_penstocks) * h_net_range * eta_t) / 1e6

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(Q_range, P_range, 'b-', label='Power Output')
ax.axvline(Q_design_total, color='g', linestyle='--', label='Design Discharge')
ax.axvline(Q_max_total, color='r', linestyle='--', label='Max Discharge')
ax.set_xlabel("Total System Discharge (m³/s)")
ax.set_ylabel("Total Power Output (MW)")
ax.set_title("System Operating Curve")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# =====================================
# Section 7: References
# =====================================
st.header("6. References")
st.markdown("""
- USBR Design Standards No. 3
- ICOLD Bulletins
- ASME Hydropower Standards
""")
