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

st.subheader("Head Loss Parameters")
col1, col2 = st.columns(2)
with col1:
    L_penstock = st.number_input("Penstock Length (m)", value=500.0)
    f = st.number_input("Friction Factor", value=0.015, min_value=0.01, max_value=0.03)
with col2:
    K_sum = st.number_input("Total Local Loss Coefficients (ΣK)", value=4.5)
    auto_hf = st.checkbox("Calculate losses automatically")

if auto_hf:
    v_design = Q_design / (math.pi*(D_pen/2)**2)
    v_max = Q_max / (math.pi*(D_pen/2)**2)
    
    hf_design = (f * L_penstock/D_pen + K_sum) * (v_design**2)/(2*g)
    hf_max = (f * L_penstock/D_pen + K_sum) * (v_max**2)/(2*g)
else:
    hf_design = st.number_input("Design Head Loss (m)", value=25.0)
    hf_max = st.number_input("Max Head Loss (m)", value=40.0)

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
# =====================================
# Section 4: Results Display (Corrected)
# =====================================
results = pd.DataFrame({
    "Parameter": ["Total System", "Per Penstock"],
    "Design Discharge (m³/s)": [Q_design_total, Q_design],
    "Max Discharge (m³/s)": [Q_max_total, Q_max],
    "Velocity (m/s)": ["-", v_max]  # Changed unit to m/s for consistency
})

# Corrected formatting
format_dict = {
    "Design Discharge (m³/s)": "{:.2f}",
    "Max Discharge (m³/s)": "{:.2f}",
    "Velocity (m/s)": lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
}

st.dataframe(results.style.format(format_dict), use_container_width=True)

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
# =====================================
# Section 6: Design Standards & References
# =====================================
st.header("5. Industry Design Standards")
st.subheader("Penstock Velocity Limits")

standards_data = [
    {
        "Organization": "US Bureau of Reclamation (USBR)",
        "Recommendation": "4–6 m/s for concrete-lined penstocks",
        "Application": "General hydropower projects",
        "Source": "Engineering Monograph No. 20 (1987)"
    },
    {
        "Organization": "International Commission on Large Dams (ICOLD)",
        "Recommendation": "< 7 m/s for steel penstocks",
        "Application": "Cavitation prevention",
        "Source": "Bulletin on Hydropower Intakes (2015)"
    },
    {
        "Organization": "ASME (American Society of Mechanical Engineers)",
        "Recommendation": "5–8 m/s (peaks), < 6 m/s (continuous)",
        "Application": "Mechanical design standards",
        "Source": "Hydropower Technical Guidelines (2019)"
    },
    {
        "Organization": "Practical Hydropower Handbooks",
        "Recommendation": "3–6 m/s (concrete/steel)",
        "Application": "Economic design range",
        "Source": "Various industry publications"
    }
]

# Display as expandable table
with st.expander("📚 View Full Design Standards"):
    st.table(pd.DataFrame(standards_data))
    
    st.markdown("""
    ### **Velocity Design Considerations**
    - **Concrete Penstocks**: 4–6 m/s (USBR)
    - **Steel Penstocks**: 5–7 m/s (ICOLD)
    - **Short-term Peaks**: Up to 8 m/s (ASME)
    - **Economic Range**: 3.5–5.5 m/s (Handbook)
    """)

# Add to velocity validation section
st.subheader("Velocity Validation")
st.latex(f"v_{{max}} = {v_max:.2f} \, \text{{m/s}}")

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

# Add reference links
st.markdown("""
### **Official References**
- [USBR Design Standards No. 3](https://www.usbr.gov/tsc/techreferences/standards.html)
- [ICOLD Bulletins](https://www.icold-cigb.org/)
- [ASME Hydropower Standards](https://www.asme.org/)
""")
