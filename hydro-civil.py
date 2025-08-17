# PHES Design Teaching App — with Moody helper (Swamee–Jain) for f(Re, ε/D)
# Reservoir head, penstock hydraulics, lining stress (modular), losses, surge tanks
# Enhanced for smooth application flow and teaching experience

import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ------------------------------- Constants -------------------------------
G = 9.81        # m/s²
RHO = 1000.0    # kg/m³
# -------------------------------------------------------------------------

# ------------------------------- Water properties ------------------------
def water_mu_dynamic_PaS(T_C: float) -> float:
    """Dynamic viscosity (Pa·s) of water vs temperature (°C)."""
    T_K = T_C + 273.15
    return 2.414e-5 * 10**(247.8/(T_K - 140))

def water_nu_kinematic_m2s(T_C: float, rho=RHO) -> float:
    """Kinematic viscosity ν = μ/ρ in m²/s."""
    return water_mu_dynamic_PaS(T_C) / rho

# ------------------------------- Geometry & algebra ----------------------
def safe_div(a, b):
    return a / b if (b is not None and b != 0) else float("nan")

def area_circle(D):
    return (math.pi * D**2) / 4.0

def Q_from_power(P_MW, h_net, eta):
    """Total discharge (m³/s) from power (MW), net head (m), efficiency (-)."""
    if h_net <= 0 or eta <= 0:
        return float("nan")
    return P_MW * 1e6 / (RHO * G * h_net * eta)

def headloss_darcy(f, L, D, v, Ksum=0.0):
    """hf = (f*L/D + ΣK) * v²/(2g)"""
    if D <= 0:
        return float("nan")
    return (f * L / D + Ksum) * (v**2) / (2 * G)

# ------------------------------- Moody helper ---------------------------
def f_moody_swamee_jain(Re, rel_rough):
    """
    Swamee–Jain explicit approximation to Colebrook-White (turbulent).
    Also smooth handling for laminar (<2000) and transitional (2000–4000).
    """
    Re = float(Re)
    if np.isnan(Re) or Re <= 0:
        return float("nan")
    if Re < 2000:
        return 64.0 / Re
    f_turb = 0.25 / (math.log10(rel_rough/3.7 + 5.74/(Re**0.9)))**2
    if Re < 4000:
        f_lam = 64.0 / Re
        w = (Re - 2000.0) / 2000.0
        return (1 - w)*f_lam + w*f_turb
    return f_turb

def roughness_library():
    """Absolute roughness ε [m] — indicative teaching values."""
    return {
        "New steel (welded)": 0.000045,
        "New steel (riveted)": 0.00015,
        "Ductile iron": 0.00026,
        "Concrete (smooth)": 0.00030,
        "Concrete (finished)": 0.00060,
        "Concrete (rough)": 0.00150,
        "PVC/HDPE": 0.0000015,
        "Rock tunnel (good lining)": 0.00100,
        "Rock tunnel (rough)": 0.00300,
        "Custom...": None,
    }

# ------------------------------- App Shell -------------------------------
st.set_page_config(
    page_title="PHES Design Teaching App",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded"
)

st.title("Pumped Hydro Energy Storage — Design Teaching App")
st.caption("Interactive tool for teaching hydroelectric design principles with Moody friction factor calculations")

# Initialize session state for presets
if 'preset_applied' not in st.session_state:
    st.session_state.preset_applied = False
    st.session_state.preset_name = "None"
    st.session_state.eta_t = 0.90
    st.session_state.eta_p = 0.88
    st.session_state.N = 6
    st.session_state.N_pen = 6
    st.session_state.P_design = 2000.0

# ------------------------------- Presets -------------------------------
with st.sidebar:
    st.header("Presets & Settings")
    preset = st.selectbox(
        "Preset", 
        ["None", "Snowy 2.0 · Plateau (NEW)", "Snowy 2.0 · Plateau (DET)", "Kidston (example)"],
        index=0
    )
    
    if st.button("Apply preset", key="apply_preset"):
        if preset == "Snowy 2.0 · Plateau (NEW)":
            st.session_state.update({
                "HWL_u": 1100.0, "LWL_u": 1080.0, "HWL_l": 450.0, "TWL_l": 420.0,
                "hf1": 28.0, "hf2": 70.0, "P1": 1000.0, "P2": 2000.0,
                "P_design": 2000.0, "N_pen": 6,
                "preset_applied": True, "preset_name": "Snowy 2.0 · Plateau (NEW)"
            })
            st.success("Snowy 2.0 (NEW) preset applied!")
        elif preset == "Snowy 2.0 · Plateau (DET)":
            st.session_state.update({
                "HWL_u": 1100.0, "LWL_u": 1080.0, "HWL_l": 450.0, "TWL_l": 420.0,
                "hf1": 30.0, "hf2": 106.0, "P1": 1000.0, "P2": 2000.0,
                "P_design": 2000.0, "N_pen": 6,
                "preset_applied": True, "preset_name": "Snowy 2.0 · Plateau (DET)"
            })
            st.success("Snowy 2.0 (DET) preset applied!")
        elif preset == "Kidston (example)":
            st.session_state.update({
                "HWL_u": 500.0, "LWL_u": 490.0, "HWL_l": 230.0, "TWL_l": 220.0,
                "hf1": 6.0, "hf2": 18.0, "P1": 250.0, "P2": 500.0,
                "P_design": 500.0, "N_pen": 2,
                "preset_applied": True, "preset_name": "Kidston (example)"
            })
            st.success("Kidston example preset applied!")
    
    if st.session_state.preset_applied:
        st.info(f"Active preset: {st.session_state.preset_name}")

    # Power settings
    P_design = st.number_input(
        "Design power (MW)", 1.0, 5000.0, 
        st.session_state.P_design, 10.0,
        key="design_power"
    )
    
    # Machine numbers
    N = st.number_input(
        "Units (N)", 1, 20, st.session_state.N, 1,
        key="units"
    )
    
    N_pen = st.number_input(
        "Number of penstocks", 1, 20, st.session_state.N_pen, 1,
        key="penstocks"
    )
    
    # Efficiencies
    eta_t = st.number_input(
        "Turbine efficiency ηₜ", 0.70, 1.00, 
        st.session_state.eta_t, 0.01,
        key="turbine_eff"
    )
    
    eta_p = st.number_input(
        "Pump efficiency ηₚ (ref.)", 0.60, 1.00, st.session_state.eta_p, 0.01,
        key="pump_eff"
    )

    st.caption("All units SI; water ρ=1000 kg/m³, g=9.81 m/s².")

# ------------------------------- Section 1: Reservoirs -------------------------------
st.header("1) Reservoir Levels, NWL & Rating Head")

# Initialize session state for reservoir levels
for var in ["HWL_u", "LWL_u", "HWL_l", "TWL_l"]:
    if var not in st.session_state:
        st.session_state[var] = {
            "Snowy 2.0 · Plateau (NEW)": 1100.0,
            "Snowy 2.0 · Plateau (DET)": 1100.0,
            "Kidston (example)": 500.0
        }.get(st.session_state.preset_name, 0.0)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Upper reservoir**")
    HWL_u = st.number_input("Upper HWL (m)", 0.0, 3000.0, st.session_state.HWL_u, 1.0, key="HWL_u")
    LWL_u = st.number_input("Upper LWL (m)", 0.0, 3000.0, st.session_state.LWL_u, 1.0, key="LWL_u")
    
with c2:
    st.markdown("**Lower reservoir**")
    HWL_l = st.number_input("Lower HWL (m)", 0.0, 3000.0, st.session_state.HWL_l, 1.0, key="HWL_l")
    TWL_l = st.number_input("Lower TWL (m)", 0.0, 3000.0, st.session_state.TWL_l, 1.0, key="TWL_l")

# Calculate reservoir parameters
try:
    Ha_u = HWL_u - LWL_u
    NWL_u = HWL_u - Ha_u / 3.0
    gross_head = NWL_u - TWL_l
    min_head = LWL_u - HWL_l
    head_fluct_ratio = safe_div((LWL_u - TWL_l), (HWL_u - TWL_l))
    
    # Visualization
    fig_res, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Upper"], [HWL_u - LWL_u], bottom=LWL_u, color="#3498DB", alpha=0.75, width=0.4)
    ax.bar(["Lower"], [HWL_l - TWL_l], bottom=TWL_l, color="#2ECC71", alpha=0.75, width=0.4)
    ax.hlines(NWL_u, -0.4, 1.4, colors="#34495E", linestyles="--", linewidth=2, label="NWL (upper)")
    ax.annotate("", xy=(1.0, NWL_u), xytext=(1.0, TWL_l),
                arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=2))
    ax.text(1.05, (NWL_u + TWL_l)/2, f"Hg ≈ {gross_head:.1f} m", color="#E74C3C", va="center")
    ax.annotate("", xy=(0.2, LWL_u), xytext=(0.2, HWL_l),
                arrowprops=dict(arrowstyle="<->", color="#27AE60", lw=2))
    ax.text(0.08, (LWL_u + HWL_l)/2, f"Min ≈ {min_head:.1f} m", color="#27AE60", va="center")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Reservoir Operating Range")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9)
    st.pyplot(fig_res)
    
except Exception as e:
    st.error(f"Error in reservoir calculation: {str(e)}")
    st.stop()

# Display metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Available drawdown H_a (m)", f"{Ha_u:.1f}")
m2.metric("NWL (m)", f"{NWL_u:.1f}")
m3.metric("Gross head H_g (m)", f"{gross_head:.1f}")
m4.metric("Head fluctuation ratio", f"{head_fluct_ratio:.3f}")

# Head fluctuation criterion
st.markdown("**Head fluctuation rate**")
st.latex(r"\text{HFR} = \frac{LWL - TWL}{HWL - TWL}")
crit_col1, crit_col2 = st.columns([2, 1])
with crit_col1:
    turbine_choice = st.selectbox(
        "Criterion (lower limit):",
        ["None (no check)", "Francis (≥ 0.70)", "Kaplan (≥ 0.55)"],
        index=1
    )
with crit_col2:
    custom_limit = st.number_input("Custom limit", 0.0, 1.0, 0.70, 0.01)

if turbine_choice.startswith("Francis"): 
    limit = 0.70
elif turbine_choice.startswith("Kaplan"):  
    limit = 0.55
else:    
    limit = None

if limit is not None and not np.isnan(head_fluct_ratio):
    st.markdown(f"**HFR:** {head_fluct_ratio:.3f}  •  **Lower limit:** {limit:.2f}")
    if head_fluct_ratio >= limit:
        st.success("Meets recommended minimum (HFR ≥ limit)")
    else:
        st.error("Below recommended minimum. Consider raising LWL or increasing HWL − TWL")

# ---------------------------- Section 2: Waterway profile & L estimator ----------------------------
st.header("2) Waterway Profile & Penstock Geometry")

# UI for profile data
left, right = st.columns([2, 1])
with left:
    csv_file = st.file_uploader("Upload CSV with columns: Chainage_m, Elevation_m",
                                type=["csv"], key="profile_csv")
    if csv_file:
        try:
            df_profile = pd.read_csv(csv_file)
            st.success("CSV loaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            df_profile = pd.DataFrame({
                "Chainage_m": [0, 500, 1000, 1500, 2000, 2300],
                "Elevation_m": [NWL_u, NWL_u-1, NWL_u-3, NWL_u-8, 450, 360],
            })
    else:
        st.caption("Using default profile. Upload CSV to customize.")
        df_profile = pd.DataFrame({
            "Chainage_m": [0, 500, 1000, 1500, 2000, 2300],
            "Elevation_m": [NWL_u, NWL_u-1, NWL_u-3, NWL_u-8, 450, 360],
        })
        df_profile = st.data_editor(df_profile, num_rows="dynamic", use_container_width=True)

with right:
    uploaded_img = st.file_uploader("Profile image (optional)", type=["png","jpg","jpeg"])

# Validate and process profile data
if "Chainage_m" in df_profile.columns and "Elevation_m" in df_profile.columns:
    try:
        df_profile = df_profile.sort_values("Chainage_m").reset_index(drop=True)
        fig_prof, axp = plt.subplots(figsize=(6, 2.8), dpi=120)
        axp.plot(df_profile["Chainage_m"], df_profile["Elevation_m"], lw=2, color="#1f77b4")
        axp.set_xlabel("Chainage (m)")
        axp.set_ylabel("Elevation (m)")
        axp.set_title("Waterway Profile")
        axp.grid(True, linestyle="--", alpha=0.35)
        st.pyplot(fig_prof, use_container_width=False)
        
        if uploaded_img:
            st.image(uploaded_img, caption="Profile Reference", use_column_width=True)
        
        # Get penstock parameters
        ch_min, ch_max = float(df_profile["Chainage_m"].min()), float(df_profile["Chainage_m"].max())
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            ch_start = st.number_input("Head-tank outlet chainage (m)", min_value=ch_min, max_value=ch_max,
                                       value=ch_min, step=10.0, format="%.1f")
        with c2:
            ch_end = st.number_input("Turbine inlet chainage (m)", min_value=ch_min, max_value=ch_max,
                                     value=ch_max, step=10.0, format="%.1f")
        with c3:
            h_draft = st.number_input("Runner draft below TWL_l (m)", 0.0, 200.0, 5.0, 0.5)
        
        # Calculate penstock length
        mask = (df_profile["Chainage_m"] >= ch_start) & (df_profile["Chainage_m"] <= ch_end)
        sub = df_profile.loc[mask].copy()
        dx = np.diff(sub["Chainage_m"].values)
        dz = np.diff(sub["Elevation_m"].values)
        seg_len = np.sqrt(dx**2 + dz**2)
        L_pen_est = float(seg_len.sum())
        
        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("Horizontal run Δx (m)", f"{ch_end - ch_start:.1f}")
        colm2.metric("Elevation change Δz (m)", f"{sub['Elevation_m'].iloc[-1] - sub['Elevation_m'].iloc[0]:.1f}")
        colm3.metric("Penstock center-line L (m)", f"{L_pen_est:.1f}")
        
        if st.button("Apply to Penstock Length"):
            st.session_state.L_penstock = L_pen_est
            st.success(f"Applied L = {L_pen_est:.1f} m to penstock length")
            
    except Exception as e:
        st.error(f"Error processing profile: {str(e)}")
else:
    st.warning("Profile data requires 'Chainage_m' and 'Elevation_m' columns")

# ---------------------------- Quick diameter-by-velocity helper ----------------------------
st.header("3) Penstock Sizing")

# Get design per-penstock discharge
Qp_design = st.session_state.get("Q_per", float("nan"))

st.subheader("Diameter from Target Velocity")
colv1, colv2, colv3 = st.columns(3)
with colv1:
    st.metric("Design per-penstock flow Qₚ (m³/s)",
              f"{Qp_design:.3f}" if not np.isnan(Qp_design) else "—")
with colv2:
    v_target = st.slider("Target velocity v (m/s)", 2.0, 8.0, 4.5, 0.1)
with colv3:
    D_suggest = math.sqrt(4.0 * Qp_design / (math.pi * v_target)) if Qp_design > 0 else float("nan")
    st.metric("Suggested diameter D (m)", f"{D_suggest:.3f}" if not np.isnan(D_suggest) else "—")

if st.button("Apply to Penstock Diameter") and not np.isnan(D_suggest):
    st.session_state.D_pen = float(D_suggest)
    st.success(f"Applied D = {D_suggest:.3f} m to penstock diameter")

# ------------------------------- Section 4: Penstock & Moody -------------------------
st.header("4) Hydraulic Analysis")

# Initialize penstock parameters
if 'L_penstock' not in st.session_state:
    st.session_state.L_penstock = 500.0
if 'D_pen' not in st.session_state:
    st.session_state.D_pen = 3.5

c1, c2 = st.columns(2)
with c1:
    D_pen = st.number_input("Penstock diameter D (m)", 0.5, 12.0, st.session_state.D_pen, 0.1)
    L_pen = st.number_input("Penstock length L (m)", 10.0, 50000.0, st.session_state.L_penstock, 10.0)
with c2:
    P_max = st.number_input("Maximum power (MW)", 10.0, 6000.0, st.session_state.get("max_power", 600.0), 10.0)

st.subheader("Friction Factor Calculation")
mode_f = st.radio("Friction factor method:",
                  ["Manual (slider)", "Compute from Moody (Swamee–Jain)"], index=1)
if mode_f == "Manual (slider)":
    f = st.slider("Friction factor f (Darcy)", 0.005, 0.03, 0.015, 0.001)
    rough_choice = "—"; eps = None; T_C = None
else:
    colA, colB, colC = st.columns(3)
    with colA:
        T_C = st.number_input("Water temperature (°C)", 0.0, 60.0, 20.0, 0.5)
    with colB:
        rl = roughness_library()
        rough_choice = st.selectbox("Material / roughness ε (m)", list(rl.keys()))
    with colC:
        eps = st.number_input("Custom ε (m)", 0.0, 0.01,
                              rl[rough_choice] if rl[rough_choice] else 0.00030,
                              0.00001, format="%.5f") if rough_choice == "Custom..." else rl[rough_choice]

# Display roughness reference
with st.expander("Roughness Reference"):
    rlib = roughness_library()
    rows = []
    for mat, eps_val in rlib.items():
        if eps_val is None: continue
        rows.append({
            "Material": mat,
            "ε (mm)": eps_val * 1e3,
            "ε (m)": eps_val,
            "ε/D": (eps_val / D_pen) if D_pen else float("nan"),
        })
    st.dataframe(pd.DataFrame(rows))

# Hydraulic calculations
def compute_block(P_MW, h_span, Ksum, hf_guess=30.0):
    """Hydraulic calculations for power block"""
    try:
        A = area_circle(D_pen)
        # First pass
        h_net = h_span - hf_guess
        Q_total = Q_from_power(P_MW, h_net, eta_t)
        Q_per = safe_div(Q_total, N_pen)
        v = safe_div(Q_per, A)

        if mode_f == "Manual (slider)":
            f_used = f
        else:
            nu = water_nu_kinematic_m2s(T_C)
            Re = safe_div(v * D_pen, nu)
            rel_rough = safe_div(eps, D_pen)
            f_used = f_moody_swamee_jain(Re, rel_rough)

        hf = headloss_darcy(f_used, L_pen, D_pen, v, Ksum=Ksum)

        # Second pass for refinement
        h_net2 = h_span - hf
        Q_total2 = Q_from_power(P_MW, h_net2, eta_t)
        Q_per2 = safe_div(Q_total2, N_pen)
        v2 = safe_div(Q_per2, A)

        if mode_f == "Manual (slider)":
            f_used2 = f
            Re2 = safe_div(v2 * D_pen, water_nu_kinematic_m2s(20.0))
        else:
            nu2 = water_nu_kinematic_m2s(T_C)
            Re2 = safe_div(v2 * D_pen, nu2)
            rel_rough2 = safe_div(eps, D_pen)
            f_used2 = f_moody_swamee_jain(Re2, rel_rough2)

        hf2 = headloss_darcy(f_used2, L_pen, D_pen, v2, Ksum=Ksum)

        return {
            "f": f_used2, "Re": Re2, "v": v2, "Q_total": Q_total2, "Q_per": Q_per2,
            "h_net": h_net2, "hf": hf2, "rel_rough": rel_rough2 if mode_f != "Manual (slider)" else None
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return {}

# Run calculations
out_design_flow = compute_block(P_design, gross_head, 0.0)
out_max_flow = compute_block(P_max, min_head, 0.0)

# Display results
if out_design_flow and out_max_flow:
    st.subheader("Flow Results")
    results_flow = pd.DataFrame({
        "Case": ["Design", "Maximum"],
        "Net head h_net (m)": [out_design_flow["h_net"], out_max_flow["h_net"]],
        "Total Q (m³/s)": [out_design_flow["Q_total"], out_max_flow["Q_total"]],
        "Per-penstock Q (m³/s)": [out_design_flow["Q_per"], out_max_flow["Q_per"]],
        "Velocity v (m/s)": [out_design_flow["v"], out_max_flow["v"]],
        "Reynolds Re (-)": [out_design_flow["Re"], out_max_flow["Re"]],
    })
    st.dataframe(results_flow)
    
    # Velocity check
    st.subheader("Velocity Validation")
    v_design = out_design_flow["v"]
    v_max = out_max_flow["v"]
    
    c1, c2 = st.columns(2)
    c1.metric("Design velocity (m/s)", f"{v_design:.2f}")
    c2.metric("Max velocity (m/s)", f"{v_max:.2f}")
    
    if v_max > 7.0:
        st.error("⚠️ Excessive velocity (>7 m/s) - Risk of erosion and vibration")
    elif v_max > 6.0:
        st.warning("⚠️ High velocity (>6 m/s) - Acceptable for short durations only")
    elif 4.0 <= v_max <= 6.0:
        st.success("✓ Velocity within recommended range (4-6 m/s)")
    else:
        st.info("ℹ️ Low velocity - Consider smaller diameter for cost efficiency")

# ------------------------------- Section 5: Results Summary -------------------------
st.header("5) Design Summary")

# Display key design parameters
st.subheader("Project Parameters")
col1, col2 = st.columns(2)
col1.metric("Design Power (MW)", f"{P_design:.1f}")
col1.metric("Number of Units", N)
col1.metric("Number of Penstocks", N_pen)
col2.metric("Gross Head (m)", f"{gross_head:.1f}")
col2.metric("Min Head (m)", f"{min_head:.1f}")

if out_design_flow:
    st.subheader("Hydraulic Summary")
    col3, col4 = st.columns(2)
    col3.metric("Design Flow (m³/s)", f"{out_design_flow['Q_total']:.1f}")
    col3.metric("Per Penstock Flow (m³/s)", f"{out_design_flow['Q_per']:.1f}")
    col4.metric("Penstock Diameter (m)", f"{D_pen:.2f}")
    col4.metric("Penstock Length (m)", f"{L_pen:.1f}")

# Final recommendations
st.subheader("Design Recommendations")
if out_design_flow and 'v' in out_design_flow:
    if out_design_flow['v'] < 4.0:
        st.info("**Velocity Recommendation:** Consider reducing penstock diameter to increase velocity (target 4-6 m/s)")
    elif out_design_flow['v'] > 6.0:
        st.info("**Velocity Recommendation:** Consider increasing penstock diameter to reduce velocity (target 4-6 m/s)")
    
if 'head_fluct_ratio' in locals() and head_fluct_ratio < 0.7:
    st.info("**Head Fluctuation Recommendation:** Increase operating range or adjust reservoir levels to improve HFR")

# ------------------------------- Section 6: Educational Resources -------------------------
st.header("Educational Resources")
with st.expander("Hydraulic Principles"):
    st.markdown("""
    **Key Equations:**
    - Darcy-Weisbach head loss: $h_f = f \\frac{L}{D} \\frac{v^2}{2g}$
    - Flow velocity: $v = \\frac{Q}{A} = \\frac{4Q}{\\pi D^2}$
    - Reynolds number: $Re = \\frac{vD}{\\nu}$
    - Swamee-Jain friction factor: $f = \\frac{0.25}{\\left[ \\log_{10} \\left( \\frac{\\varepsilon}{3.7D} + \\frac{5.74}{Re^{0.9}} \\right) \\right]^2}$
    """)
    
with st.expander("Recommended Literature"):
    st.markdown("""
    1. **Design of Small Dams** - USBR (1987)  
    2. **Hydropower Engineering Handbook** - McGraw Hill  
    3. **Fluid Mechanics** - Frank M. White  
    4. **Hydraulic Structures** - P. Novak et al.
    """)

# Footer
st.divider()
st.caption("PHES Design Teaching Tool | Developed for Engineering Education | v2.0")
# --------------- Section 4: Head Losses & Diameter Sizing ----------------
st.header("4) Head Losses & Diameter Sizing")

# Local loss builder (ΣK)
st.subheader("Local loss components (ΣK)")
components = {
    "Entrance (bellmouth)": 0.15, "Entrance (square)": 0.50,
    "90° bend": 0.25, "45° bend": 0.15,
    "Gate valve (open)": 0.20, "Butterfly valve (open)": 0.30,
    "T-junction": 0.40, "Exit": 1.00
}
K_sum_global = 0.0
cols = st.columns(4)
for i, (comp, kval) in enumerate(components.items()):
    default_on = comp in ["Entrance (bellmouth)", "90° bend", "Exit"]
    with cols[i % 4]:
        if st.checkbox(comp, value=default_on):
            K_sum_global += kval
st.metric("ΣK (selected)", f"{K_sum_global:.2f}")

# Compute with ΣK to show h_f and f (two-pass block)
out_design = compute_block(P_design, gross_head, K_sum_global, hf_guess=25.0)
out_max    = compute_block(P_max,    min_head,  K_sum_global, hf_guess=40.0)

results_losses = pd.DataFrame({
    "Case": ["Design", "Maximum"],
    "Darcy f (-)": [out_design["f"], out_max["f"]],
    "Reynolds Re (-)": [out_design["Re"], out_max["Re"]],
    "Head loss h_f (m)": [out_design["hf"], out_max["hf"]],
    "Net head h_net (m)": [out_design["h_net"], out_max["h_net"]],
})
st.dataframe(
    results_losses, use_container_width=True,
    column_config={
        "Darcy f (-)": st.column_config.NumberColumn(format="%.004f"),
        "Reynolds Re (-)": st.column_config.NumberColumn(format="%.0f"),
        "Head loss h_f (m)": st.column_config.NumberColumn(format="%.2f"),
        "Net head h_net (m)": st.column_config.NumberColumn(format="%.2f"),
    }
)

# Diameter Estimator (three methods)
st.subheader("Diameter estimator (pick a method)")
Q_for_sizing = out_design_flow.get("Q_total", float("nan")) or 0.0

tabA, tabB, tabC = st.tabs(["Chart extrapolation", "Velocity target", "Head-loss target"])

with tabA:
    a_fit, b_fit, D_of_Q = fit_extrapolate_Q_to_D(Q_chart, D_chart)
    D_ext = float(D_of_Q(Q_for_sizing)) if Q_for_sizing > 0 else float("nan")
    st.write(f"Fitted curve: **D ≈ {a_fit:.3f} · Q^{b_fit:.3f}**  (Q in m³/s, D in m)")
    st.metric("Suggested D (m)", f"{D_ext:.2f}" if not np.isnan(D_ext) else "—")
    if st.button("Apply suggested D (chart fit)"):
        if not np.isnan(D_ext):
            st.session_state["D_pen"] = float(D_ext)
            st.success(f"Applied D = {D_ext:.2f} m to the Penstock Geometry panel (re-run to see effect).")

with tabB:
    V_target = st.slider("Target velocity V (m/s)", 2.0, 8.0, 4.5, 0.1,
                         help="Pick a reasonable operating velocity; see Section 3 velocity guidance.")
    D_vel = D_from_velocity(Q_for_sizing, V_target) if Q_for_sizing > 0 else float("nan")
    st.metric("Suggested D (m)", f"{D_vel:.2f}" if not np.isnan(D_vel) else "—")
    if st.button("Apply suggested D (velocity)"):
        if not np.isnan(D_vel):
            st.session_state["D_pen"] = float(D_vel)
            st.success(f"Applied D = {D_vel:.2f} m to the Penstock Geometry panel (re-run to see effect).")

with tabC:
    hf_allow = st.number_input("Allowable head loss h_f (m)", 1.0, 100.0, 15.0, 0.5,
                               help="Total Darcy–Weisbach + local losses allowance along the penstock.")
    eps_used = eps if (mode_f != "Manual (slider)" and eps is not None) else 3e-4
    T_used   = T_C if (mode_f != "Manual (slider)" and T_C is not None) else 15.0
    D_iter, f_it, Re_it, v_it, hf_it = D_from_headloss(Q_for_sizing, L_pen, hf_allow,
                                                       eps=eps_used, Ksum=K_sum_global,
                                                       T_C=T_used)
    st.metric("Suggested D (m)", f"{D_iter:.2f}" if not np.isnan(D_iter) else "—")
    st.caption(f"At that D: f≈{f_it:.4f}, Re≈{Re_it:.2e}, v≈{v_it:.2f} m/s, h_f≈{hf_it:.2f} m")
    if st.button("Apply suggested D (head-loss)"):
        if not np.isnan(D_iter):
            st.session_state["D_pen"] = float(D_iter)
            st.success(f"Applied D = {D_iter:.2f} m to the Penstock Geometry panel (re-run to see effect).")

# ---- Figures / Equations reference (Section 4) ----
with st.expander("Show figures / equations used (Section 4)"):
    st.markdown("**Core relations**")
    st.latex(r"h_f = \left(f \frac{L}{D} + \sum K \right)\frac{v^2}{2g}")
    st.latex(r"f \approx \frac{0.25}{\left[\log_{10}\!\left(\frac{\varepsilon}{3.7D} + \frac{5.74}{\mathrm{Re}^{0.9}}\right)\right]^2}\quad\text{(Swamee–Jain)}")
    st.latex(r"\mathrm{Re}=\frac{vD}{\nu},\quad A=\frac{\pi D^2}{4},\quad v=\frac{Q_p}{A},\quad Q_p=\frac{Q_{\text{total}}}{N_{\text{pen}}}")
    st.latex(r"\frac{\varepsilon}{D}\;\text{(relative roughness)}")
    st.markdown("**Diameter sizing ideas**")
    st.latex(r"D\;=\;\sqrt{\frac{4Q}{\pi V_{\text{target}}}}\quad\text{(velocity target)}")
    st.latex(r"\text{Find }D\text{ s.t. }h_f(D)\approx h_{f,\text{allow}}\quad\text{(iterate with }f(\mathrm{Re},\varepsilon/D)\text{)}")
    st.markdown("**Figures**")
    st.markdown(
        "- *Moody diagram:* friction factor **f** vs **Re** and **ε/D** (your ‘Mini Moody’ plot in Section 2/3).\n"
        "- *Head-loss breakdown:* Darcy–Weisbach main loss plus selected local loss coefficients (ΣK) above."
    )
    st.markdown("**Teaching references**")
    st.markdown(
        "- USBR **Design of Small Dams** (3rd ed., 1987), Penstocks & hydraulics chapters.\n"
        "- USACE EM 1110-2-1602 / Idelchik: typical **ΣK** ranges and local-loss data.\n"
        "- AWWA / ASCE tables: indicative absolute roughness **ε** for common materials.\n"
        "- Swamee, P.K. & Jain, A.K. (1976): explicit friction-factor formula used here."
    )


# ------------------------------- Section 5: System Curve ------------------
st.header("5) System Power Curve (didactic)")
Q_max_total = out_max["Q_total"]
if np.isnan(Q_max_total) or Q_max_total <= 0:
    Q_max_total = max(1.0, (P_max * 1e6) / (RHO * G * max(min_head, 1.0) * max(eta_t, 0.6)))

Q_grid = np.linspace(0, 1.2 * Q_max_total, 140)
h_net_design = out_design["h_net"]; h_net_min = out_max["h_net"]
if any(np.isnan([h_net_design, h_net_min])):
    h_net_design, h_net_min = max(gross_head - 10, 1.0), max(min_head - 10, 0.5)

h_net_curve = h_net_design - (h_net_design - h_net_min) * (Q_grid / max(Q_max_total, 1e-6))**2
P_curve = RHO * G * Q_grid * h_net_curve * eta_t / 1e6

fig = make_subplots(specs=[[{"secondary_y": False}]])
fig.add_trace(go.Scatter(x=Q_grid, y=P_curve, name="Power (MW)", line=dict(width=3)))
fig.add_vline(x=out_design["Q_total"], line=dict(color="green", dash="dash"), annotation_text="Design Q")
fig.add_vline(x=out_max["Q_total"], line=dict(color="red", dash="dash"), annotation_text="Max Q")
fig.update_layout(
    title="Operating Characteristics (didactic)",
    xaxis_title="Total discharge Q (m³/s)",
    yaxis_title="Power (MW)",
    hovermode="x unified",
    height=480
)
st.plotly_chart(fig, use_container_width=True)

# --------------------- Section 6 (modular): Rock Cover & Lining ----------
def rock_cover_and_lining_ui():
    """Self-contained UI section for Rock Cover & Lining — returns a summary dict."""
    st.header("5) Pressure Tunnel: Rock Cover & Lining Stress")

    # Rock cover inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hs = st.number_input("Hydrostatic head to crown h_s (m)", 10.0, 2000.0, 300.0, 1.0)
    with c2:
        alpha = st.number_input("Tunnel inclination α (deg)", 0.0, 90.0, 20.0, 1.0)
    with c3:
        ri = st.number_input("Lining inner radius r_i (m)", 0.2, 10.0, 3.15, 0.05)
    with c4:
        t = st.number_input("Lining thickness t (m)", 0.1, 2.0, 0.35, 0.01)

    re = ri + t
    gamma_R = st.slider("Rock unit weight γ_R (kN/m³)", 15.0, 30.0, 26.0, 0.5)
    CRV = snowy_vertical_cover(hs, gamma_w=9.81, gamma_R=gamma_R)
    FRV = norwegian_FRV(CRV, hs, alpha, gamma_w=9.81, gamma_R=gamma_R)

    cc1, cc2 = st.columns(2)
    cc1.metric("Snowy vertical cover C_RV (m)", f"{CRV:.1f}")
    cc2.metric("Norwegian factor F_RV (-)", f"{FRV:.2f}")
    st.markdown("**Target**: Typically F_RV ≥ 1.2–1.5 (site-dependent).")

    # Lining stress
    st.subheader("Lining Hoop Stress (Lame solution)")
    c1, c2, c3 = st.columns(3)
    with c1:
        pi_MPa = st.number_input("Internal water pressure p_i (MPa)", 0.1, 20.0, 2.0, 0.1, key="pi_MPa")
    with c2:
        pext = st.number_input("External confinement p_ext (MPa)", 0.0, 20.0, 0.0, 0.1, key="pext")
    with c3:
        ft_MPa = st.number_input("Concrete tensile strength f_t (MPa)", 1.0, 10.0, 3.0, 0.1, key="ft_MPa")

    sigma_outer = hoop_stress(pi_MPa, pext, ri, re)   # at outer face
    pext_req = required_pext_for_ft(pi_MPa, ri, re, ft_MPa)

    r_plot = np.linspace(ri * 1.001, re, 200)
    sigma_profile = hoop_stress(pi_MPa, pext, ri, r_plot)

    fig_s, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(r_plot, sigma_profile, lw=2.2, label="σθ(r)")
    ax.axhline(ft_MPa, color="g", ls="--", label=f"f_t = {ft_MPa:.1f} MPa")
    ax.axvline(ri, color="k", ls=":", label=f"ri={ri:.2f} m")
    ax.axvline(re, color="k", ls="--", label=f"re={re:.2f} m")
    ax.fill_between(r_plot, sigma_profile, ft_MPa, where=(sigma_profile > ft_MPa),
                    color="red", alpha=0.2, label="Cracking risk")
    ax.set_xlabel("Radius r (m)")
    ax.set_ylabel("Hoop stress σθ (MPa)")
    ax.set_title("Lining hoop stress distribution")
    ax.set_ylim(0, 100)  # for readability
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    st.pyplot(fig_s)

    c1, c2, c3 = st.columns(3)
    c1.metric("σθ @ outer face (MPa)", f"{sigma_outer:.1f}")
    c2.metric("Required p_ext (MPa)", f"{pext_req:.2f}")
    c3.metric(
        "Status",
        "⚠️ Cracking likely" if sigma_outer > ft_MPa else "✅ OK",
        help=("Stress exceeds tensile strength; increase thickness or confinement."
              if sigma_outer > ft_MPa else "Within tensile capacity at outer face.")
    )

    return {
        "hs": hs, "alpha_deg": alpha, "ri_m": ri, "t_m": t, "re_m": re,
        "gamma_R_kNpm3": gamma_R, "CRV_m": CRV, "FRV": FRV,
        "pi_MPa": pi_MPa, "pext_MPa": pext, "ft_MPa": ft_MPa,
        "sigma_outer_MPa": sigma_outer, "pext_required_MPa": pext_req
    }

rock_summary = rock_cover_and_lining_ui()  # returns dict (for export)

# ------------------------------- Section 7: Surge Tank -------------------
st.header("7) Surge Tank — First Cut")
Ah = area_circle(D_pen)  # per conduit; for multi-branch, use local area at the tank location
Lh = st.number_input("Headrace length to surge tank L_h (m)", 100.0, 100000.0, 15000.0, 100.0)
ratio = st.number_input("Area ratio A_s/A_h (-)", 1.0, 10.0, 4.0, 0.1)
surge = surge_tank_first_cut(Ah, Lh, ratio=ratio)

c1, c2, c3 = st.columns(3)
c1.metric("A_h (m²)", f"{Ah:.2f}")
c2.metric("A_s (m²)", f"{surge['As']:.2f}")
c3.metric("Natural period T_n (s)", f"{surge['Tn']:.1f}")
st.caption("Rule-of-thumb only. Real designs require full water-hammer/transient analysis.")

# ------------------------------- Section 8: Equations --------------------
st.header("8) Core Equations (for teaching)")
tabH, tabM, tabS = st.tabs(["Hydraulics", "Mechanics (Lining)", "Surge/Waterhammer"])
with tabH:
    st.markdown("#### Continuity"); st.latex(r"Q = A \, v")
    st.markdown("#### Bernoulli (with losses)")
    st.latex(r"\frac{P_1}{\rho g} + \frac{v_1^2}{2g} + z_1 = \frac{P_2}{\rho g} + \frac{v_2^2}{2g} + z_2 + h_f")
    st.markdown("#### Turbine Power"); st.latex(r"P = \rho g Q H_{\text{net}} \eta_t")
    st.markdown("#### Darcy–Weisbach Head Loss (with local losses)")
    st.latex(r"h_f = \left(f \frac{L}{D} + \sum K \right) \frac{v^2}{2g}")
with tabM:
    st.markdown("#### Lame (thick-walled cylinder) — hoop stress")
    st.latex(r"\sigma_\theta(r) = \frac{p_i (r^2 + r_i^2) - 2 p_e r^2}{r^2 - r_i^2}")
    st.markdown("#### Required external confinement (didactic inner-fibre check)")
    st.latex(r"p_{e,\text{req}} \approx \frac{(p_i - f_t) (r_o^2 - r_i^2)}{2 r_o^2}")
    st.markdown("#### Snowy vertical cover"); st.latex(r"C_{RV} = \frac{h_s \, \gamma_w}{\gamma_R}")
    st.markdown("#### Norwegian valley-side stability factor")
    st.latex(r"F_{RV} = \frac{C_{RV} \, \gamma_R \cos\alpha}{h_s \, \gamma_w}")
with tabS:
    st.markdown("#### First-cut surge tank sizing (simple oscillator)")
    st.latex(r"A_s = k \, A_h, \quad \omega_n = \sqrt{\frac{g A_h}{L_h A_s}}, \quad T_n = \frac{2\pi}{\omega_n}")
    st.caption("Use only as a teaching baseline; proper design requires transient simulation (e.g., method of characteristics).")

# ------------------------------- Section 9: Reference Tables -------------
st.header("9) Reference Tables (typical classroom values)")
with st.expander("📚 Friction Factors (Darcy) — typical ranges & sources", expanded=False):
    df_f = pd.DataFrame({
        "Material": ["New steel (welded)", "New steel (riveted)", "Concrete (smooth)", "Concrete (rough)", "PVC/Plastic"],
        "Typical f": [0.012, 0.017, 0.015, 0.022, 0.009],
        "Range": ["0.010–0.015", "0.015–0.020", "0.012–0.018", "0.018–0.025", "0.007–0.012"],
        "Source (teaching)": ["ASCE (2017)","USBR (1987)","ACI 351.3R (2018)","USACE EM (2008)","AWWA (2012)"]
    })
    st.table(df_f)
with st.expander("📚 Local Loss Coefficients ΣK — indicative ranges & notes", expanded=False):
    df_k = pd.DataFrame({
        "Component": ["Entrance (bellmouth)", "Entrance (square)", "90° bend", "45° bend", "Gate valve (open)",
                      "Butterfly valve (open)", "T-junction", "Exit"],
        "K (typical)": [0.15, 0.50, 0.25, 0.15, 0.20, 0.30, 0.40, 1.00],
        "Range": ["0.1–0.2","0.4–0.5","0.2–0.3","0.1–0.2","0.1–0.3","0.2–0.4","0.3–0.5","0.8–1.0"],
        "Notes": ["Best-case entrance","Worst-case entrance","Radius/diameter dependent",
                  "Gentler than 90°","Design dependent","Position dependent","Flow split losses","Kinetic recovery lost"]
    })
    st.table(df_k)
    st.caption("Typical ΣK for well-designed penstock trunks: ~2–5 (teaching values).")

# ------------------------------- Section 10: Downloads -------------------
st.header("10) Downloads & Bibliography")
bundle = {
    "reservoirs": {"upper": {"HWL": HWL_u, "LWL": LWL_u}, "lower": {"HWL": HWL_l, "TWL": TWL_l}},
    "penstock": {"N": N_pen, "D": D_pen, "L": L_pen,
                 "mode_f": mode_f,
                 "f_manual": (f if mode_f == "Manual (slider)" else None),
                 "T_C": (T_C if mode_f != "Manual (slider)" else None),
                 "eps_m": (eps if mode_f != "Manual (slider)" else None),
                 "rel_rough": (out_design["rel_rough"] if mode_f != "Manual (slider)" else None),
                 "SigmaK": K_sum_global},
    "efficiency": {"eta_t": eta_t},
    "operating": {"design": out_design, "max": out_max},
    "surge": {"Ah": area_circle(D_pen), **surge},
    "rock_cover_lining": rock_summary
}
st.download_button("Download JSON", data=json.dumps(bundle, indent=2), file_name="phes_results.json")

flat = {
    "HWL_u": HWL_u, "LWL_u": LWL_u, "HWL_l": HWL_l, "TWL_l": TWL_l,
    "N_pen": N_pen, "D_pen": D_pen, "L_pen": L_pen,
    "mode_f": mode_f,
    "f": out_design["f"], "Re_design": out_design["Re"],
    "f_max": out_max["f"], "Re_max": out_max["Re"],
    "SigmaK": K_sum_global, "eta_t": eta_t,
    "P_design_MW": P_design, "P_max_MW": P_max,
    "hnet_design_m": out_design["h_net"], "Q_total_design_m3s": out_design["Q_total"],
    "v_design_ms": out_design_flow["v"], "hf_design_m": out_design["hf"],
    "hnet_max_m": out_max["h_net"], "Q_total_max_m3s": out_max["Q_total"],
    "v_max_ms": out_max_flow["v"], "hf_max_m": out_max["hf"],
    "T_C": (T_C if mode_f != "Manual (slider)" else None),
    "eps_m": (eps if mode_f != "Manual (slider)" else None),
    "rel_rough": (out_design["rel_rough"] if mode_f != "Manual (slider)" else None),
}
st.download_button(
    "Download CSV (parameters)",
    data=pd.DataFrame([flat]).to_csv(index=False).encode("utf-8"),
    file_name="phes_parameters.csv"
)

st.markdown("""
**Bibliography (teaching references)**  
- USBR Design Standards (Penstocks; Hydraulics)  
- ICOLD Bulletins on Pressure Tunnels and Surge Tanks  
- ASCE Manuals & ACI 351.3R (Concrete & friction ranges)  
- USACE Engineering Manuals (Hydraulic Loss Coefficients)  
- Chaudhry, M.H. (2014). *Applied Hydraulic Transients*.  
- Gordon, J.L. (2001). *Hydraulics of Hydroelectric Power*.  
""")
st.caption("Educational tool • Use for teaching & scoping only • © Your Course / Lab")
