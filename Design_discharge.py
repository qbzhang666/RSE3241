import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Constants
g = 9.81  # m/s²
rho = 1000  # kg/m³

st.set_page_config(layout="wide")
st.title("Pumped Hydro Storage Discharge Analysis")

# ======================
# Section 1: Input Parameters
# ======================
st.header("1. System Parameters")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reservoir Levels (m)")
    HWL_u = st.number_input("Upper Reservoir HWL", value=1100.0)
    LWL_u = st.number_input("Upper Reservoir LWL", value=1080.0)
    HWL_l = st.number_input("Lower Reservoir HWL", value=450.0)
    TWL_l = st.number_input("Lower Reservoir TWL", value=420.0)

with col2:
    st.subheader("Design Parameters")
    eta_t = st.number_input("Turbine Efficiency", value=0.90, min_value=0.7, max_value=1.0)
    D_pen = st.number_input("Penstock Diameter (m)", value=3.5)
    design_power = st.number_input("Design Power (MW)", value=500.0)
    max_power = st.number_input("Maximum Power (MW)", value=600.0)

# ======================
# Section 2: Calculations
# ======================
st.header("2. Discharge Calculations")

# Head calculations
gross_head = HWL_u - TWL_l
h_net_design = HWL_u - TWL_l - 25  # Assuming 25m head loss at design
h_net_min = LWL_u - HWL_l - 40     # Assuming 40m head loss at max flow

# Discharge equations
with st.expander("Show Equations"):
    st.latex(r'''
    \text{Discharge Calculation: } 
    Q = \frac{P \times 10^6}{\rho \times g \times h_{net} \times \eta_t}
    ''')
    st.latex(r'''
    \text{Velocity Calculation: }
    v = \frac{Q}{A} = \frac{Q}{\pi \times (D_{pen}/2)^2}
    ''')
    st.latex(r'''
    \text{Power Calculation: }
    P = \frac{\rho \times g \times Q \times h_{net} \times \eta_t}{10^6}
    ''')

# Calculate discharges
Q_design = (design_power * 1e6) / (rho * g * h_net_design * eta_t)
Q_max = (max_power * 1e6) / (rho * g * h_net_min * eta_t)

# Calculate velocities
A_pen = math.pi * (D_pen/2)**2
v_design = Q_design / A_pen
v_max = Q_max / A_pen

# ======================
# Section 3: Results Display
# ======================
st.header("3. Results Comparison")

# Create results table
results = pd.DataFrame({
    "Parameter": ["Design", "Maximum"],
    "Power (MW)": [design_power, max_power],
    "Net Head (m)": [h_net_design, h_net_min],
    "Discharge (m³/s)": [Q_design, Q_max],
    "Velocity (m/s)": [v_design, v_max]
})

st.dataframe(results.style.format({
    "Power (MW)": "{:.1f}",
    "Net Head (m)": "{:.1f}",
    "Discharge (m³/s)": "{:.2f}",
    "Velocity (m/s)": "{:.2f}"
}), use_container_width=True)

# Safety margin
safety_margin = ((Q_max - Q_design) / Q_design) * 100
st.metric("Safety Margin", f"{safety_margin:.1f}%")

# ======================
# Section 4: Visualizations
# ======================
st.header("4. System Characteristics")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Discharge comparison
ax1.bar(["Design", "Maximum"], [Q_design, Q_max], color=["blue", "orange"])
ax1.set_ylabel("Discharge (m³/s)")
ax1.set_title("Design vs Maximum Discharge")
for i, val in enumerate([Q_design, Q_max]):
    ax1.text(i, val, f"{val:.1f}", ha='center', va='bottom')

# System curve
Q_range = np.linspace(0, Q_max*1.2, 100)
h_net_range = h_net_design - (h_net_design - h_net_min) * (Q_range/Q_max)**2
P_range = (rho * g * Q_range * h_net_range * eta_t) / 1e6

ax2.plot(Q_range, P_range, 'g-', label='Power Output')
ax2.axvline(Q_design, color='b', linestyle='--', label='Design Discharge')
ax2.axvline(Q_max, color='r', linestyle='--', label='Max Discharge')
ax2.set_xlabel("Discharge (m³/s)")
ax2.set_ylabel("Power (MW)")
ax2.set_title("System Operating Characteristics")
ax2.legend()
ax2.grid(True)

st.pyplot(fig)

# ======================
# Section 5: Design Checks
# ======================
st.header("5. Design Verification")

# Velocity check
st.subheader("Velocity Check")
st.write(f"Maximum velocity: {v_max:.2f} m/s")
if v_max > 6.0:
    st.error("Velocity exceeds typical limit of 6 m/s for concrete-lined tunnels")
else:
    st.success("Velocity within acceptable limits")

# Capacity check
st.subheader("Capacity Check")
if Q_design > Q_max:
    st.error("Design discharge exceeds maximum system capacity!")
else:
    st.success("Design discharge within system capacity")

# Design ratios
st.subheader("Design Ratios")
col1, col2 = st.columns(2)

with col1:
    load_ratio = Q_design / Q_max
    st.metric("Load Ratio (Q_design/Q_max)", f"{load_ratio:.2f}", 
             help="Optimal range: 0.7-0.9")

with col2:
    head_ratio = h_net_design / gross_head
    st.metric("Head Utilization", f"{head_ratio:.2f}",
             help="Ratio of design net head to gross head")

# ======================
# Relevant Equations Summary
# ======================
st.header("Key Equations Summary")

st.markdown("""
**1. Discharge Calculation:**
\[ Q = \frac{P \times 10^6}{\rho \times g \times h_{net} \times \eta_t} \]

**2. Flow Velocity:**
\[ v = \frac{Q}{A} = \frac{4Q}{\pi D^2} \]

**3. Power Output:**
\[ P = \frac{\rho \times g \times Q \times h_{net} \times \eta_t}{10^6} \]

**4. Head Loss Approximation:**
\[ h_f = k \times Q^n \] 
*(Where k and n are system-specific coefficients)*

**5. Safety Margin:**
\[ \text{Margin} = \left(\frac{Q_{max} - Q_{design}}{Q_{design}}\right) \times 100\% \]

**6. Load Ratio:**
\[ \text{Load Ratio} = \frac{Q_{design}}{Q_{max}} \]
""")
