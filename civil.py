import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import json

# Configure plotting style
plt.style.use('default')
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "axes.edgecolor": "0.15",
    "axes.linewidth": 1.25,
})

# Constants
g = 9.81  # m/s²
rho = 1000  # kg/m³

# Physics helper functions
def hoop_stress(pi, pe, ri, r):
    with np.errstate(divide='ignore', invalid='ignore'):
        stress = (pi*(r**2 + ri**2) - 2*pe*r**2)/(r**2 - ri**2)
    return stress

def required_pext_for_ft(pi_MPa, ri, re, ft_MPa):
    return (ft_MPa - pi_MPa) * (re**2 - ri**2) / (2.0 * re**2)

def snowy_vertical_cover(hs, gamma_w=9.81, gamma_R=26.0):
    return (hs * gamma_w) / gamma_R

def norwegian_FRV(CRV, hs, alpha_deg, gamma_w=9.81, gamma_R=26.0):
    return (CRV * gamma_R * math.cos(math.radians(alpha_deg))) / (hs * gamma_w)

def surge_tank_first_cut(Ah, Lh, ratio=4.0):
    As = ratio * Ah
    omega_n = math.sqrt(g * Ah / (Lh * As))
    Tn = 2*math.pi/omega_n
    return dict(As=As, omega_n=omega_n, Tn=Tn)

# App configuration
st.set_page_config(page_title="PHES Design Teaching App", layout="wide")
st.title("Pumped Hydro Energy Storage Design App")
st.markdown("""
**Teaching tool for hydropower engineering principles** · Combines penstock hydraulics, tunnel mechanics, and system design
""")

# =====================================
# SECTION 1: SYSTEM OVERVIEW
# =====================================
st.header("1. System Overview")

with st.expander("Project Description", expanded=True):
    st.markdown("""
    This app demonstrates key design principles for pumped hydro storage systems:
    
    1. **Reservoir Sizing & Head Determination**  
    2. **Penstock Hydraulics** (flow, velocity, head loss)  
    3. **Pressure Tunnel Design** (rock mechanics, lining stresses)  
    4. **Head Loss Analysis**  
    5. **Surge Tank Fundamentals**  
    
    Based on industry standards from Snowy 2.0, Kidston PHES, and international design practices.
    """)

# Presets system
with st.sidebar:
    st.header("Project Presets")
    preset = st.selectbox("Select Project", ["Custom", "Snowy 2.0 · Plateau", "Kidston PHES"])
    
    if preset == "Snowy 2.0 · Plateau":
        st.session_state.update(dict(
            HWL_u=1100.0, LWL_u=1080.0, HWL_l=450.0, TWL_l=420.0,
            N_penstocks=6, D_pen=4.8, design_power=1000, max_power=2000,
            f_material="Concrete (smooth)", f=0.015, L_penstock=15000
        ))
    elif preset == "Kidston PHES":
        st.session_state.update(dict(
            HWL_u=500.0, LWL_u=490.0, HWL_l=230.0, TWL_l=220.0,
            N_penstocks=2, D_pen=3.2, design_power=250, max_power=500,
            f_material="New steel (welded)", f=0.012, L_penstock=800
        ))

# =====================================
# SECTION 2: RESERVOIR PARAMETERS
# =====================================
st.header("2. Reservoir Parameters")
st.subheader("Water Level Elevations")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Upper Reservoir**")
    HWL_u = st.number_input("High Water Level (m)", 0.0, 3000.0, float(st.session_state.get("HWL_u", 1100.0)), 1.0, key="hwl_u")
    LWL_u = st.number_input("Low Water Level (m)", 0.0, 3000.0, float(st.session_state.get("LWL_u", 1080.0)), 1.0, key="lwl_u")
with col2:
    st.markdown("**Lower Reservoir**")
    HWL_l = st.number_input("High Water Level (m)", 0.0, 3000.0, float(st.session_state.get("HWL_l", 450.0)), 1.0, key="hwl_l")
    TWL_l = st.number_input("Tailwater Level (m)", 0.0, 3000.0, float(st.session_state.get("TWL_l", 420.0)), 1.0, key="twl_l")

# Calculate head parameters
gross_head = HWL_u - TWL_l
min_head = LWL_u - HWL_l
head_fluctuation = (LWL_u - TWL_l)/(HWL_u - TWL_l) if (HWL_u - TWL_l) > 0 else 0

# Visualization
fig_res, ax = plt.subplots(figsize=(10, 6))
ax.bar(['Upper'], [HWL_u - LWL_u], bottom=LWL_u, color='#3498DB', alpha=0.7)
ax.bar(['Lower'], [HWL_l - TWL_l], bottom=TWL_l, color='#2ECC71', alpha=0.7)
ax.annotate('', xy=(0, HWL_u), xytext=(0, TWL_l), arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=2))
ax.text(0.1, (HWL_u + TWL_l)/2, f'Max Head: {gross_head:.1f} m', ha='left', va='center', fontsize=12)
ax.annotate('', xy=(0.4, LWL_u), xytext=(0.4, HWL_l), arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=2))
ax.text(0.5, (LWL_u + HWL_l)/2, f'Min Head: {min_head:.1f} m', ha='left', va='center', fontsize=12)
ax.set_ylabel('Elevation (m)')
ax.set_title('Reservoir Operating Range')
ax.grid(True, linestyle='--', alpha=0.4)
st.pyplot(fig_res)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Gross Head", f"{gross_head:.1f} m")
col2.metric("Head Fluctuation", f"{head_fluctuation:.2f}")
col3.metric("Energy Storage", f"{(HWL_u - LWL_u) * (HWL_l - TWL_l) / 1e6:.1f} GL", "Based on reservoir areas")

# =====================================
# SECTION 3: PENSTOCK DESIGN
# =====================================
st.header("3. Penstock Design")
st.subheader("Hydraulic Parameters")

col1, col2 = st.columns(2)
with col1:
    N_penstocks = st.number_input("Number of Penstocks", min_value=1, max_value=8, 
                                 value=int(st.session_state.get("N_penstocks", 2)), key="n_penstocks")
    D_pen = st.number_input("Diameter (m)", value=float(st.session_state.get("D_pen", 3.5)), key="d_pen")
    design_power = st.number_input("Design Power (MW)", value=float(st.session_state.get("design_power", 500.0)), key="p_design")
    max_power = st.number_input("Maximum Power (MW)", value=float(st.session_state.get("max_power", 600.0)), key="p_max")
    
with col2:
    eta_t = st.number_input("Turbine Efficiency", value=0.90, min_value=0.7, max_value=1.0, key="eta_t")
    L_penstock = st.number_input("Length (m)", value=float(st.session_state.get("L_penstock", 500.0)), key="l_penstock")
    h_draft = st.number_input("Draft Head (m)", 5.0, 50.0, 15.0, 1.0, 
                             help="Distance from tailwater to turbine centerline")
    runner_CL = TWL_l - h_draft
    st.metric("Turbine Centerline Elevation", f"{runner_CL:.1f} m")

# Head Loss Parameters
st.subheader("Head Loss Parameters")
col1, col2 = st.columns(2)

with col1:
    # Friction factor selection
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
        index=2 if "Concrete" in st.session_state.get("f_material", "") else 0,
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
    
with col2:
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
    
    st.markdown("**Local Loss Coefficients (ΣK)**")
    K_sum = 0.0
    for comp, k_val in components.items():
        if st.checkbox(comp, value=(comp in ["Entrance (bellmouth)", "90° bend", "Exit"])):
            K_sum += k_val
    st.metric("Total ΣK", f"{K_sum:.2f}", "Sum of all selected components")

auto_hf = st.checkbox("Calculate head losses automatically", value=True, key="auto_hf")

# Initialize head losses
hf_design = 25.0  
hf_max = 40.0

if auto_hf:
    # Iterative head loss calculation
    h_net_design_temp = gross_head - hf_design
    h_net_min_temp = min_head - hf_max
    
    Q_design_total_temp = (design_power * 1e6) / (rho * g * h_net_design_temp * eta_t)
    Q_max_total_temp = (max_power * 1e6) / (rho * g * h_net_min_temp * eta_t)
    
    v_design = (4 * (Q_design_total_temp/N_penstocks)) / (math.pi * D_pen**2)
    v_max = (4 * (Q_max_total_temp/N_penstocks)) / (math.pi * D_pen**2)
    
    hf_design = (f * L_penstock/D_pen + K_sum) * (v_design**2)/(2*g)
    hf_max = (f * L_penstock/D_pen + K_sum) * (v_max**2)/(2*g)
    
    st.success(f"Calculated Head Losses: Design = {hf_design:.2f} m, Max = {hf_max:.2f} m")
else:
    hf_design = st.number_input("Design Head Loss (m)", value=25.0, key="hf_design")
    hf_max = st.number_input("Max Head Loss (m)", value=40.0, key="hf_max")

# =====================================
# SECTION 4: DESIGN PRINCIPLES & EQUATIONS
# =====================================
st.header("4. Fundamental Design Equations")

tab1, tab2, tab3 = st.tabs(["Hydraulics", "Structural Mechanics", "System Design"])

with tab1:
    st.subheader("Hydraulic Principles")
    st.markdown("""
    #### Continuity Equation
    $$ Q = A \cdot v $$
    
    #### Bernoulli's Energy Equation
    $$ \\frac{P_1}{\\rho g} + \\frac{v_1^2}{2g} + z_1 = \\frac{P_2}{\\rho g} + \\frac{v_2^2}{2g} + z_2 + h_f $$
    
    #### Power Calculation
    $$ P = \\rho \\cdot g \\cdot Q \\cdot H_{\\text{net}} \\cdot \\eta_t $$
    
    #### Darcy-Weisbach Head Loss
    $$ h_f = \\left( f \\frac{L}{D} + \\sum K \\right) \\frac{v^2}{2g} $$
    """)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Moody_diagram.jpg/800px-Moody_diagram.jpg", 
             caption="Darcy-Weisbach friction factor diagram (Moody Chart)", 
             width=500)

with tab2:
    st.subheader("Structural Mechanics")
    st.markdown("""
    #### Hoop Stress in Pressure Tunnels
    $$ \\sigma_\\theta = \\frac{p_i (r^2 + r_i^2) - 2p_e r^2}{r^2 - r_i^2} $$
    
    #### Minimum Cover Depth (Snowy Formula)
    $$ C_{RV} = \\frac{h_s \\cdot \\gamma_w}{\\gamma_R} $$
    
    #### Norwegian Stability Criterion
    $$ F_{RV} = \\frac{C_{RV} \\cdot \\gamma_R \\cdot \\cos \\alpha}{h_s \\cdot \\gamma_w} $$
    
    Where:
    - $F_{RV} > 1.5$ for stable rock conditions
    - $F_{RV} > 2.0$ for fractured rock
    """)

with tab3:
    st.subheader("System Design Principles")
    st.markdown("""
    #### Surge Tank Sizing
    $$ A_s = k \\cdot A_h $$
    $$ T_n = 2\\pi \\sqrt{\\frac{L_h A_s}{g A_h}} $$
    
    #### Head Loss Curve Fitting
    $$ h_f = k \\cdot Q^n $$
    
    Where:
    - $k$: System loss coefficient
    - $n$: Flow exponent (1.85-2.0)
    - $k = 4$ for conservative design
    """)
# =====================================
# SECTION 5: CALCULATION RESULTS
# =====================================
st.header("5. Calculation Results")

# Net heads
h_net_design = gross_head - hf_design
h_net_min = min_head - hf_max

# Discharges
Q_design_total = (design_power * 1e6) / (rho * g * h_net_design * eta_t)
Q_max_total = (max_power * 1e6) / (rho * g * h_net_min * eta_t)
Q_design = Q_design_total / N_penstocks
Q_max = Q_max_total / N_penstocks

# Velocities
A_pen = math.pi * (D_pen/2)**2
v_design = Q_design / A_pen
v_max = Q_max / A_pen

# Results table
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

# Velocity validation
st.subheader("Velocity Validation")
col1, col2 = st.columns(2)
with col1:
    st.metric("Design Velocity", f"{v_design:.2f} m/s")
    st.metric("Max Velocity", f"{v_max:.2f} m/s")

with col2:
    st.markdown("### USBR Standards")
    st.markdown("- **Recommended range:** 4-6 m/s")
    st.markdown("- **Absolute maximum:** 7 m/s (short durations)")

if v_max > 7.0:
    st.error("⚠️ **Dangerous Velocity** - Exceeds all standards")
elif v_max > 6.0:
    st.warning("⚠️ **Above Recommended Limit** - Acceptable for short durations only")
elif v_max > 4.0:
    st.success("✓ **Optimal Range** - Meets USBR standards")
else:
    st.info("ℹ️ **Low Velocity** - Consider smaller diameter for cost savings")

# =====================================
# SECTION 6: PRESSURE TUNNEL DESIGN
# =====================================
st.header("6. Pressure Tunnel Design")

col1, col2, col3, col4 = st.columns(4)
with col1:
    hs = st.number_input("Hydrostatic Head to Crown (m)", 100.0, 1000.0, 300.0, 10.0)
with col2:
    alpha = st.number_input("Tunnel Inclination (°)", 0.0, 90.0, 20.0, 1.0)
with col3:
    ri = st.number_input("Inner Radius (m)", 2.0, 10.0, 3.15, 0.05)
with col4:
    t = st.number_input("Lining Thickness (m)", 0.2, 2.0, 0.35, 0.01)

re = ri + t
pi_MPa = st.slider("Internal Pressure (MPa)", 0.1, 10.0, 2.0, 0.1)
ft_MPa = st.number_input("Concrete Tensile Strength (MPa)", 1.0, 100.0, 3.0, 0.1)

# Calculate stresses
sigma_theta_i = hoop_stress(pi_MPa, 0, ri, re)
pext_req = required_pext_for_ft(pi_MPa, ri, re, ft_MPa)

# Visualization
r_vals = np.linspace(ri*1.001, ri*2, 100)
sigma_vals = hoop_stress(pi_MPa, 0, ri, r_vals)

fig_stress, ax = plt.subplots(figsize=(10, 5))
ax.plot(r_vals, sigma_vals, 'b-', lw=2.5, label='Hoop Stress')
ax.axhline(ft_MPa, color='g', ls='--', label=f'Tensile Strength ({ft_MPa} MPa)')
ax.axvline(ri, color='k', ls=':', label=f'Inner Radius ({ri} m)')
ax.axvline(re, color='k', ls='--', label=f'Outer Radius ({re} m)')
ax.fill_between(r_vals, sigma_vals, ft_MPa, where=(sigma_vals>ft_MPa), color='red', alpha=0.2)
ax.set_xlabel('Radius (m)'); ax.set_ylabel('Stress (MPa)')
ax.set_title('Lining Stress Distribution'); ax.grid(True); ax.legend()
st.pyplot(fig_stress)

# Results
col1, col2, col3 = st.columns(3)
col1.metric("Max Stress", f"{sigma_theta_i:.1f} MPa")
col2.metric("Required p_ext", f"{pext_req:.2f} MPa")
col3.metric("Status", 
            "⚠️ Cracking Risk" if sigma_theta_i > ft_MPa else "✅ Safe", 
            "Stress exceeds strength" if sigma_theta_i > ft_MPa else "Within limits")

# =====================================
# SECTION 7: SYSTEM CHARACTERISTICS
# =====================================
st.header("7. System Characteristics")

# Generate operating curve
Q_range = np.linspace(0, Q_max_total*1.2, 100)
h_net_range = h_net_design - (h_net_design - h_net_min) * (Q_range/Q_max_total)**2
P_range = N_penstocks * (rho * g * (Q_range/N_penstocks) * h_net_range * eta_t) / 1e6

# Create interactive plot
fig_sys = make_subplots(specs=[[{"secondary_y": True}]])

# Main power curve
fig_sys.add_trace(
    go.Scatter(
        x=Q_range,
        y=P_range,
        name="Power Output",
        line=dict(color="blue", width=3),
        hovertemplate="Discharge: %{x:.1f} m³/s<br>Power: %{y:.1f} MW"
    ),
    secondary_y=False,
)

# Reference lines
fig_sys.add_vline(x=Q_design_total, line=dict(color="green", dash="dash", width=2), name="Design")
fig_sys.add_vline(x=Q_max_total, line=dict(color="red", dash="dash", width=2), name="Max")

# Layout
fig_sys.update_layout(
    title="System Operating Characteristics",
    xaxis_title="Total System Discharge (m³/s)",
    yaxis_title="Power Output (MW)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig_sys, use_container_width=True)

# =====================================
# SECTION 8: SURGE TANK DESIGN
# =====================================
st.header("8. Surge Tank Fundamentals")

Ah = math.pi * (D_pen**2)/4
Lh = st.number_input("Headrace Length to Surge Tank (m)", 1000.0, 20000.0, 5000.0, 100.0)
ratio = st.slider("Area Ratio (Aₛ/Aₕ)", 1.0, 10.0, 4.0, 0.1)

surge_params = surge_tank_first_cut(Ah, Lh, ratio=ratio)

col1, col2, col3 = st.columns(3)
col1.metric("Conduit Area (Aₕ)", f"{Ah:.1f} m²")
col2.metric("Surge Tank Area (Aₛ)", f"{surge_params['As']:.1f} m²")
col3.metric("Natural Period (Tₙ)", f"{surge_params['Tn']:.1f} s")

st.markdown("""
**Design Guidelines:**
- Natural period should be 60-90 seconds for stable operation
- Minimum area ratio: 3:1 for medium-head systems
- Consider differential surge tanks for large systems
""")

# =====================================
# SECTION 9: REFERENCES & DOWNLOADS
# =====================================
st.header("9. References & Export")

tab1, tab2, tab3 = st.tabs(["Standards", "References", "Export"])

with tab1:
    st.subheader("Design Standards")
    st.markdown("""
    #### Friction Factors (f)
    | Material              | Range       | Source          |
    |-----------------------|-------------|-----------------|
    | New steel (welded)    | 0.010-0.015 | ASCE (2017)     |
    | Concrete (smooth)     | 0.012-0.018 | ACI 351.3R      |
    
    #### Local Loss Coefficients (K)
    | Component         | Range    |
    |-------------------|----------|
    | Entrance (bell)   | 0.1-0.2  |
    | 90° bend          | 0.2-0.3  |
    | Exit              | 0.8-1.0  |
    """)

with tab2:
    st.subheader("Recommended References")
    st.markdown("""
    1. **USBR Design Standards No. 3** - Penstock design guidelines
    2. **ICOLD Bulletins** - Pressure tunnel recommendations
    3. **ASME Hydropower Standards** - Mechanical design
    4. Gordon, J.L. (2001) *Hydraulics of Hydroelectric Power*
    5. Chaudhry, M.H. (2014) *Applied Hydraulic Transients*
    """)

with tab3:
    st.subheader("Export Results")
    results_data = {
        "reservoirs": {
            "upper": {"HWL": HWL_u, "LWL": LWL_u},
            "lower": {"HWL": HWL_l, "TWL": TWL_l}
        },
        "penstock": {
            "diameter": D_pen,
            "length": L_penstock,
            "material": f_material,
            "friction_factor": f
        },
        "performance": {
            "design_power": design_power,
            "max_power": max_power,
            "design_flow": Q_design_total,
            "max_flow": Q_max_total
        },
        "tunnel": {
            "stress": sigma_theta_i,
            "required_external_pressure": pext_req
        }
    }
    
    json_data = json.dumps(results_data, indent=2)
    st.download_button("Download JSON Report", data=json_data, file_name="phes_design_report.json")
    
    st.download_button("Download Parameters CSV", 
                      data=pd.DataFrame(results_data).to_csv().encode('utf-8'),
                      file_name="phes_parameters.csv")

st.caption("Educational Tool · Hydropower Engineering · v2.0")
