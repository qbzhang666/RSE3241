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
    HWL_u = st.number_input("Upper Reservoir HWL (m)", value=1100.0)
    LWL_u = st.number_input("Upper Reservoir LWL (m)", value=1080.0)
    HWL_l = st.number_input("Lower Reservoir HWL (m)", value=450.0)
    TWL_l = st.number_input("Lower Reservoir TWL (m)", value=420.0)
    hf_design = st.number_input("Design Head Loss (m)", value=25.0)
    hf_max = st.number_input("Max Head Loss (m)", value=40.0)

with col2:
    st.subheader("Penstock Design")
    N_penstocks = st.number_input("Number of Penstocks", min_value=1, max_value=8, value=2)
    eta_t = st.number_input("Turbine Efficiency", value=0.90, min_value=0.7, max_value=1.0)
    D_pen = st.number_input("Penstock Diameter (m)", value=3.5)
    design_power = st.number_input("Design Power (MW)", value=500.0)
    max_power = st.number_input("Maximum Power (MW)", value=600.0)

# =====================================
# Section 2: Key Equations
# =====================================
st.header("2. Design Equations")
with st.expander("View Fundamental Equations"):
    st.markdown("""
    ### **1. Total Discharge Calculation**
    \[
    Q_{total} = \frac{P \times 10^6}{\rho \times g \times h_{net} \times \eta_t}
    \]
    Where:
    - \( P \) = Power (MW)
    - \( h_{net} = \Delta H - h_f \) (Gross head - head loss)
    
    ### **2. Per-Penstock Discharge**
    \[
    Q_{penstock} = \frac{Q_{total}}{N_{penstocks}}
    \]
    
    ### **3. Flow Velocity**
    \[
    v = \frac{Q_{penstock}}{A} = \frac{4Q_{penstock}}{\pi D^2}
    \]
    *Recommended limit: 4-6 m/s (USBR)*
    
    ### **4. Head Loss (Darcy-Weisbach)**
    \[
    h_f = \frac{fLv^2}{D2g}
    \]
    Where \( f \) = friction factor (~0.015 for concrete)
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
    "Velocity (m³/s)": ["-", v_max]
})

st.dataframe(results.style.format({
    "Design Discharge (m³/s)": "{:.2f}",
    "Max Discharge (m³/s)": "{:.2f}",
    "Velocity (m³/s)": "{:.2f}" if not isinstance(v, str) else v
    for v in results["Velocity (m³/s)"]
}), use_container_width=True)

# Velocity check
st.subheader("Velocity Validation")
st.latex(f"v_{{max}} = {v_max:.2f} \, \text{{m/s}}")

if v_max > 6.0:
    st.error(f"**Exceeds recommended limit (6 m/s, USBR)**")
    st.markdown("""
    **Mitigation Options:**
    - Increase diameter (current: {D_pen} m)
    - Add more penstocks (current: {N_penstocks})
    - Reduce max power (current: {max_power} MW)
    """)
elif v_max > 4.0:
    st.success("Within recommended range (4-6 m/s)")
else:
    st.warning("Low velocity (<4 m/s) - May lead to uneconomic design")

# =====================================
# Section 5: System Curves
# =====================================
st.header("4. System Characteristics")

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
# Section 6: References
# =====================================
st.header("5. Design Standards")
st.markdown("""
- **USBR (1987)**: *Design Standards No. 3 - Hydropower*
  - Concrete penstocks: 4-6 m/s
  - Steel penstocks: 5-7 m/s
  
- **ASME (2019)**: *Hydropower Technical Guidelines*
  - Short-term peaks: ≤8 m/s
  - Continuous operation: ≤6 m/s
  
- **Practical Design Handbook**:
  - Economic velocity range: 3.5-5.5 m/s
""")
