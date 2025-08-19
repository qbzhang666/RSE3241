# PHES Design Teaching App — with Moody helper (Swamee–Jain) for f(Re, ε/D)
# Reservoir head, penstock hydraulics, lining stress (modular), losses, surge tanks
# Classroom-friendly; robust on Streamlit Cloud; no Styler usage

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
RHO = 1000.0    # kg/m³  (held constant for teaching; you can make it T-dependent)
# -------------------------------------------------------------------------

# ------------------------------- Water properties ------------------------
def water_mu_dynamic_PaS(T_C: float) -> float:
    """Dynamic viscosity (Pa·s) of water vs temperature (°C)."""
    T_K = T_C + 273.15
    return 2.414e-5 * 10**(247.8/(T_K - 140))  # Pa·s

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

# ------------------------------- Mechanics helpers ----------------------
def hoop_stress(pi, pe, ri, r):
    """
    Lame solution (thick-walled cylinder, elastic):
    σθ(r) = [pi*(r^2 + ri^2) - 2*pe*r^2] / (r^2 - ri^2)
    Accepts scalars or numpy arrays for r.
    """
    r_arr = np.array([r]) if np.isscalar(r) else np.array(r)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (pi * (r_arr**2 + ri**2) - 2 * pe * r_arr**2) / (r_arr**2 - ri**2)
    s[r_arr <= ri] = np.nan
    return s.item() if np.isscalar(r) else s

def required_pext_for_ft(pi_MPa, ri, re, ft_MPa):
    """Didactic check for external confinement (inner-fibre form)."""
    return max(0.0, (pi_MPa - ft_MPa) * (re**2 - ri**2) / (2.0 * re**2))

def snowy_vertical_cover(hs, gamma_w=9.81, gamma_R=26.0):
    """Snowy-style vertical cover C_RV (m)."""
    return (hs * gamma_w) / gamma_R

def norwegian_FRV(CRV, hs, alpha_deg, gamma_w=9.81, gamma_R=26.0):
    """Norwegian stability factor FRV (valley side). Target often ≥1.2–1.5."""
    if hs <= 0:
        return float("nan")
    return (CRV * gamma_R * math.cos(math.radians(alpha_deg))) / (hs * gamma_w)

def surge_tank_first_cut(Ah, Lh, ratio=4.0):
    """Very first-cut surge tank sizing (didactic)."""
    if Ah <= 0 or Lh <= 0 or ratio <= 0:
        return {"As": float("nan"), "omega_n": float("nan"), "Tn": float("nan")}
    As = ratio * Ah
    omega_n = math.sqrt(G * Ah / (Lh * As))
    Tn = 2 * math.pi / omega_n
    return dict(As=As, omega_n=omega_n, Tn=Tn)

# ------------------------------- Diameter helpers -----------------------
# (A) Chart extrapolation D = a Q^b  — seeded with typical points up to 150 m³/s.
Q_chart = np.array([5, 10, 20, 30, 50, 75, 100, 125, 150], dtype=float)       # m³/s
D_chart = np.array([1.6, 2.1, 3.0, 3.8, 4.6, 5.3, 5.9, 6.5, 7.0], dtype=float)  # m

def fit_extrapolate_Q_to_D(Q_data, D_data):
    Q_data = np.asarray(Q_data, float)
    D_data = np.asarray(D_data, float)
    x = np.log(Q_data); y = np.log(D_data)
    b, ln_a = np.polyfit(x, y, 1)  # y = b x + ln a
    a = np.exp(ln_a)
    def D_of_Q(Q): return a * (np.asarray(Q, float) ** b)
    return a, b, D_of_Q

def D_from_velocity(Q, V):
    """Diameter from target velocity."""
    return math.sqrt(4.0 * Q / (math.pi * V))

def D_from_headloss(Q, L, hf_allow, eps=3e-4, Ksum=2.0, T_C=15.0, rho=1000.0, g=9.81,
                    D_init=None, tol=1e-6, itmax=80):
    """Solve D for a target head loss using Swamee–Jain + ΣK."""
    if hf_allow <= 0 or L <= 0 or Q <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    nu = water_nu_kinematic_m2s(T_C, rho)
    D = D_from_velocity(Q, 4.0) if D_init is None else float(D_init)
    for _ in range(itmax):
        A = math.pi * D**2 / 4.0
        v = Q / A
        Re = v * D / nu
        f = f_moody_swamee_jain(Re, eps / D)
        hf = (f * L / D + Ksum) * v**2 / (2.0 * g)
        # numeric derivative
        dD = 1e-6 * max(1.0, D)
        A2 = math.pi * (D + dD)**2 / 4.0
        v2 = Q / A2
        Re2 = v2 * (D + dD) / nu
        f2 = f_moody_swamee_jain(Re2, eps / (D + dD))
        hf2 = (f2 * L / (D + dD) + Ksum) * v2**2 / (2.0 * g)
        dhf_dD = (hf2 - hf) / dD if dD != 0 else 0.0
        if abs(dhf_dD) < 1e-10:
            break
        D_new = D - (hf - hf_allow) / dhf_dD
        if D_new <= 0: D_new = 0.5 * D
        if abs(D_new - D) < tol * D:
            D = D_new; break
        D = D_new
    A = math.pi * D**2 / 4.0
    v = Q / A
    Re = v * D / nu
    f = f_moody_swamee_jain(Re, eps / D)
    hf = (f * L / D + Ksum) * v**2 / (2.0 * g)
    return D, f, Re, v, hf

# ------------------------------- App Shell -------------------------------
st.set_page_config(page_title="PHES Civil Design", layout="wide")
st.title("Pumped Hydro Energy Storage — Civil Design")


# ------------------------------- Presets -------------------------------
with st.sidebar:
    st.header("Presets & Settings")
    preset = st.selectbox(
        "Preset", 
        ["Select Project", "Snowy 2.0 · Ravine", "Snowy 2.0 · Plateau", "Kidston"]
    )

    # Show warning if a preset is chosen but not yet applied
    if preset != "Select Project":
        st.warning("Click **Apply preset** to confirm this preset before continuing.")

    if st.button("Apply preset"):
        if preset == "Snowy 2.0 · Ravine":
            st.session_state.update(dict(
                HWL_u=1100.0, LWL_u=1000.0, HWL_l=450.0, TWL_l=410.0,
                eta_t=0.90, N=6, hf1=28.0, hf2=70.0, 
                P1=1000.0, P2=2000.0,
                P_design=2000.0, N_pen=6
            ))
        elif preset == "Snowy 2.0 · Plateau":
            st.session_state.update(dict(
                HWL_u=1100.0, LWL_u=1080.0, HWL_l=450.0, TWL_l=420.0,
                eta_t=0.90, N=6, hf1=30.0, hf2=106.0, 
                P1=1000.0, P2=2000.0,
                P_design=2000.0, N_pen=6
            ))
        elif preset == "Kidston":
            st.session_state.update(dict(
                HWL_u=500.0, LWL_u=490.0, HWL_l=230.0, TWL_l=220.0,
                eta_t=0.90, N=2, hf1=6.0, hf2=18.0, 
                P1=250.0, P2=500.0,
                P_design=500.0, N_pen=2
            ))

    # Power
    P_design = st.number_input(
        "Design power: P_design (MW)", 1.0, 5000.0, 
        float(st.session_state.get("P_design", 2000.0)), 10.0
    )
    # Machine numbers
    N = st.number_input(
        "Units (N)", 1, 20, int(st.session_state.get("N", 6)), 1
    )
    N_pen = st.number_input(
        "Number of penstocks: N_pen", 1, 20, int(st.session_state.get("N_pen", 6)), 1
    )   
    # Efficiencies
    eta_t = st.number_input(
        "Turbine efficiency ηₜ", 0.70, 1.00, 
        float(st.session_state.get("eta_t", 0.90)), 0.01
    )
    eta_p = st.number_input(
        "Pump efficiency ηₚ (ref.)", 0.60, 1.00, 0.88, 0.01
    )



    st.caption("All units SI; water ρ=1000 kg/m³, g=9.81 m/s².")


import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (6, 4),   # default width, height in inches
    "figure.dpi": 100,          # resolution
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# ------------------------------- Section 1: Reservoirs -------------------------------
st.header("1) Reservoir Levels, NWL & Rating Head")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Upper reservoir**")
    HWL_u = st.number_input("Upper HWL (m)", 0.0, 3000.0,
                            float(st.session_state.get("HWL_u", 1100.0)), 1.0)
    LWL_u = st.number_input("Upper LWL (m)", 0.0, 3000.0,
                            float(st.session_state.get("LWL_u", 1080.0)), 1.0)
with c2:
    st.markdown("**Lower reservoir**")
    HWL_l = st.number_input("Lower HWL (m)", 0.0, 3000.0,
                            float(st.session_state.get("HWL_l", 450.0)), 1.0)
    TWL_l = st.number_input("Lower TWL (m)", 0.0, 3000.0,
                            float(st.session_state.get("TWL_l", 420.0)), 1.0)

# Drawdown & NWL (upper pond)
Ha_u  = HWL_u - LWL_u                 # available drawdown
NWL_u = HWL_u - Ha_u / 3.0            # NWL = HWL − Ha/3

# Heads (NWL-based gross head)
gross_head = NWL_u - TWL_l            # H_g = NWL − TWL
min_head   = LWL_u - HWL_l            # worst case
head_fluct_ratio = safe_div((LWL_u - TWL_l), (HWL_u - TWL_l))  # (LWL − TWL)/(HWL − TWL)

# --- Save outputs so other sections can access ---
st.session_state["gross_head"] = gross_head
st.session_state["NWL_u"] = NWL_u
st.session_state["Ha_u"] = Ha_u
st.session_state["HFR"] = head_fluct_ratio

# --- Visualisation ---
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
ax.set_title("Reservoir Operating Range (with NWL)")
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="upper left", fontsize=9)
st.pyplot(fig_res)

# Equations (for teaching)
with st.expander("Show equations used"):
    st.latex(r"H_a = HWL - LWL")
    st.latex(r"NWL = HWL - \frac{H_a}{3}")
    st.latex(r"H_g = NWL - TWL")
    st.latex(r"\text{Head fluctuation rate (HFR)} = \frac{LWL - TWL}{HWL - TWL}")

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Available drawdown H_a (m)", f"{Ha_u:.1f}")
m2.metric("NWL (m)", f"{NWL_u:.1f}")
m3.metric("Gross head H_g = NWL−TWL (m)", f"{gross_head:.1f}")
m4.metric("Head fluctuation ratio (HFR)", f"{head_fluct_ratio:.3f}")


# ---------------------------- Section 2: Waterway profile & L estimator ----------------------------
st.header("2) Waterway Profile & Penstock Geometry")

# UI: CSV or quick editor
left, right = st.columns([2, 1])
with left:
    csv_file = st.file_uploader("Upload CSV with columns: Chainage_m, Elevation_m",
                                type=["csv"], key="profile_csv")
    if csv_file:
        df_profile = pd.read_csv(csv_file)
    else:
        st.caption("No CSV? Edit a small table below (chainage increasing):")
        df_profile = pd.DataFrame({
            "Chainage_m": [0, 500, 1000, 1500, 2300],           # demo values
            "Elevation_m": [NWL_u, NWL_u-1, NWL_u-3, NWL_u-8, 450],
        })
        df_profile = st.data_editor(df_profile, num_rows="dynamic", use_container_width=True)

with right:
    uploaded_img = st.file_uploader("(Optional) Profile image (png/jpg)", type=["png","jpg","jpeg"],
                                    key="profile_img")

# Validate columns
valid_cols = {"Chainage_m", "Elevation_m"}
if set(df_profile.columns) >= valid_cols and len(df_profile) >= 2:
    # sort by chainage just in case
    df_profile = df_profile.sort_values("Chainage_m").reset_index(drop=True)

    # Small, compact profile figure
    fig_prof, axp = plt.subplots(figsize=(6, 2.8), dpi=120)
    axp.plot(df_profile["Chainage_m"], df_profile["Elevation_m"], lw=2, color="#1f77b4")
    axp.set_xlabel("Chainage (m)")
    axp.set_ylabel("Elevation (m)")
    axp.set_title("Waterway long-section (compact)")
    axp.grid(True, linestyle="--", alpha=0.35)
    st.pyplot(fig_prof, use_container_width=False)

    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded profile image", use_column_width=True)

    # Pick the start/end chainages that bound the PRESSURIZED line (head tank → turbine)
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

    # Slice profile between ch_start and ch_end
    ch_lo, ch_hi = (ch_start, ch_end) if ch_start <= ch_end else (ch_end, ch_start)
    mask = (df_profile["Chainage_m"] >= ch_lo) & (df_profile["Chainage_m"] <= ch_hi)
    sub = df_profile.loc[mask].copy()
    # Ensure endpoints exist exactly at ch_start/ch_end by linear interpolation if needed
    for target in [ch_lo, ch_hi]:
        if not np.any(np.isclose(sub["Chainage_m"], target)):
            # interpolate elevation at 'target'
            above = df_profile[df_profile["Chainage_m"] >= target].iloc[0]
            below = df_profile[df_profile["Chainage_m"] <= target].iloc[-1]
            if above["Chainage_m"] == below["Chainage_m"]:
                zt = float(above["Elevation_m"])
            else:
                w = (target - below["Chainage_m"]) / (above["Chainage_m"] - below["Chainage_m"])
                zt = float((1-w)*below["Elevation_m"] + w*above["Elevation_m"])
            sub = pd.concat([sub, pd.DataFrame({"Chainage_m":[target], "Elevation_m":[zt]})], ignore_index=True)
            sub = sub.sort_values("Chainage_m").reset_index(drop=True)

    # Compute center-line length along the polyline
    dx = np.diff(sub["Chainage_m"].values)
    dz = np.diff(sub["Elevation_m"].values)
    seg_len = np.sqrt(dx**2 + dz**2)
    L_pen_est = float(seg_len.sum())

    # Simple diagnostics
    Lh = float(ch_hi - ch_lo)
    dz_tot = float(sub["Elevation_m"].iloc[-1] - sub["Elevation_m"].iloc[0])
    runner_CL = TWL_l - h_draft  # informational only

    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("Horizontal run Δx (m)", f"{Lh:.1f}")
    colm2.metric("Elevation change Δz (m)", f"{dz_tot:.1f}")
    colm3.metric("Penstock Length L (m)", f"{L_pen_est:.1f}")

    # Auto-apply L to Step 2 (no button needed)
if not np.isnan(L_pen_est):
    st.session_state["L_penstock"] = float(L_pen_est)

# Reference & equations (collapsed)
with st.expander("How is L computed? (figures / equations)"):
    st.markdown(
        "**Pressurized length definition:** "
        "L is the distance along the pipe centreline from the head-tank outlet to the turbine inlet "
        "(including the short powerhouse run). The open-channel headrace is **not** included."
    )

    st.markdown("**Polyline summation**")
    st.latex(r"L = \sum_{i=0}^{n-1} \sqrt{(x_{i+1}-x_i)^2 + (z_{i+1}-z_i)^2}")

    st.markdown("**Single-slope approximation (optional)**")
    st.latex(r"L \approx \sqrt{(\Delta x)^2 + (\Delta z)^2}")

    st.markdown("**Application in head-loss (Darcy–Weisbach)**")
    st.latex(r"h_f = \left( f \frac{L}{D} + \sum K \right)\frac{v^2}{2g}")

    st.caption(
        r"with friction factor \( f = f(\mathrm{Re}, \varepsilon/D) \) obtained from the Moody relation "
        r"(explicit Swamee–Jain form in this app)."
    )

# ---------------------------- Quick diameter-by-velocity helper ----------------------------
st.subheader("Quick diameter from target velocity")

# Compute total design flow and per-penstock flow from sidebar inputs + Section 1 head
P_design = st.session_state.get("P_design", float("nan"))     # MW
eta_t    = st.session_state.get("eta_t",  float("nan"))       # -
N_pen    = st.session_state.get("N_pen",  0)                  # count
H_g      = gross_head                                         # from Section 1

Q_total_design = float("nan")
Qp_design      = float("nan")

if (not np.isnan(P_design)) and (not np.isnan(eta_t)) and (H_g is not None) and (H_g > 0) and (eta_t > 0) and (N_pen > 0):
    Q_total_design = (P_design * 1e6) / (RHO * G * H_g * eta_t)  # m³/s
    Qp_design      = Q_total_design / N_pen

# Two-column layout
col_left, col_right = st.columns([1.2, 1.8])

# LEFT: Total & per-penstock flows
with col_left:
    st.metric(
        "Total design flow Q_total (m³/s)",
        f"{Q_total_design:.3f}" if not np.isnan(Q_total_design) else "—"
    )
    st.metric(
        "Design per-penstock flow Qₚ (m³/s)",
        f"{Qp_design:.3f}" if not np.isnan(Qp_design) else "—"
    )
    if np.isnan(Q_total_design):
        st.caption(
            ":red[Set **Design power (MW)**, **ηₜ**, **# penstocks**, and complete **Section 1** to get H_g.]"
        )

# RIGHT: Target velocity slider + suggested diameter
with col_right:
    v_target = st.slider("Target velocity v (m/s)", 1.0, 10.0, 4.0, 0.1, format="%.1f")
    if not np.isnan(Qp_design) and v_target > 0:
        D_suggested = math.sqrt(4.0 * Qp_design / (math.pi * v_target))
        st.metric("Suggested diameter D (m)", f"{D_suggested:.3f}")
        # auto-apply to model
        st.session_state["D_pen"] = float(D_suggested)
        st.caption("✔ Diameter has been applied automatically to the model.")
    else:
        st.metric("Suggested diameter D (m)", "—")
        st.caption(":red[Provide valid inputs to compute D.]")

# Reference / equations
with st.expander("Figures & equations used (diameter by velocity)"):
    st.markdown("**Total design flow rate**")
    st.latex(r"""
    Q_{\text{total,design}}
    = \frac{ P_{\text{design}}\,10^{6} }{ \rho\, g\, H_g\, \eta_{\text{live}} }
    \;\;[\mathrm{m}^3\,\mathrm{s}^{-1}]
    """)
    
    st.markdown("**Per-penstock flow from continuity**")
    st.latex(r"Q_p = \frac{Q_{\text{total}}}{N_{\text{pen}}}")
    st.latex(r"A = \frac{\pi D^2}{4}")
    st.latex(r"v = \frac{Q_p}{A}")
    st.markdown("**Solve for diameter from target velocity**")
    st.latex(r"D = \sqrt{\frac{4\,Q_p}{\pi\,v}}")
    st.markdown("**Head-loss check (Darcy–Weisbach)**")
    st.latex(r"h_f = \left(f\frac{L}{D}+\sum K\right)\frac{v^2}{2g}")
    st.latex(r"f \approx \frac{0.25}{\left[\log_{10}\!\left(\frac{\varepsilon}{3.7D}+\frac{5.74}{\mathrm{Re}^{0.9}}\right)\right]^2}")

st.subheader("Penstock Geometry")
c1, c2 = st.columns(2)
with c1:
    L_pen = st.number_input("Penstock length L (m)", 10.0, 50000.0, float(st.session_state.get("L_penstock", 500.0)), 10.0)
with c2:
    D_pen = st.number_input("Penstock diameter D (m)", 0.5, 12.0, float(st.session_state.get("D_pen", 3.5)), 0.1)

st.subheader("Turbine Center Line (CL) and Draft Head")

st.markdown("**Relation between Maximum Pumping Head and Draft Head**")

# Digitized reference points from the purple curve (x = max pumping head [m], y = draft head [m, negative])
xk = np.array([100, 200, 300, 400, 500, 600, 800], dtype=float)
yk = np.array([-23, -33, -42, -50, -58, -66, -84], dtype=float)

# Smooth cubic fit to the reference points
coef = np.polyfit(xk, yk, 3)
x = np.linspace(0.0, 600.0, 500)
y = np.polyval(coef, x)

# Use gross head from Section 1 as the "maximum pumping head" (horizontal axis input)
H_g_val = float(gross_head) if not np.isnan(gross_head) else float("nan")
y_at_Hg = float(np.polyval(coef, H_g_val)) if not np.isnan(H_g_val) else float("nan")

# Show numbers
colA, colB = st.columns(2)
with colA:
    st.metric("Maximum pumping head (≈ Gross head H_g)", f"{H_g_val:.1f} m" if not np.isnan(H_g_val) else "—")
with colB:
    st.metric("Estimated draft head from fit", f"{y_at_Hg:.1f} m" if not np.isnan(y_at_Hg) else "—")

# LaTeX reference (click to expand)
with st.expander("Draft head estimate from fitted curve (click to expand)"):
    st.latex(r"\text{Given gross head } H_g \text{ (used as maximum pumping head):}")
    st.latex(r"h_{\text{draft}}(H_g) \;\approx\; a\,H_g^{3} \;+\; b\,H_g^{2} \;+\; c\,H_g \;+\; d")
    st.caption("Coefficients (a, b, c, d) are obtained by least-squares fit to the digitized purple curve.")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Fitted curve and reference points
ax.plot(x, y, color="#8A2BE2", lw=3, label="Fitted curve (purple)")
ax.plot(xk, yk, "o", color="#8A2BE2", mfc="white", ms=7, label="Reference points")

# Mark current operating point at H_g
if not np.isnan(H_g_val):
    ax.axvline(H_g_val, color="#555555", ls="--", lw=1.5, label=f"H_g = {H_g_val:.1f} m")
    ax.plot([H_g_val], [y_at_Hg], "o", color="red", ms=8, label=f"Draft ≈ {y_at_Hg:.1f} m")
    ax.annotate(f"({H_g_val:.0f}, {y_at_Hg:.1f})",
                xy=(H_g_val, y_at_Hg), xytext=(10, -12),
                textcoords="offset points", fontsize=10, color="red")

# Axis: 0 at bottom, negatives upward (as per your screenshot)
ax.set_xlim(-10, 610)
ax.set_ylim(0, -70)
ax.set_xlabel("Maximum pumping head (m)", fontsize=12)
ax.set_ylabel("Draft head (m)", fontsize=12)
ax.set_title("Relationship: Pumping head vs Draft head", fontsize=14)

# Guide lines
for yref in range(-70, 1, 10):
    ax.axhline(yref, color="gray", linestyle=":", alpha=0.4)
for xref in [0, 100, 200, 300, 400, 500, 600]:
    ax.axvline(xref, color="gray", linestyle=":", alpha=0.25)

ax.legend(loc="upper right", fontsize=9)
ax.grid(False)
st.pyplot(fig)



with st.expander("Turbine Center Setting (click to expand)"):

    st.markdown("**Turbine Center Setting**")

    st.latex(r"Turbine\ CL\ elevation \;=\; TWL_{\ell} \;-\; h_{\text{draft}}")

    st.latex(r"TWL_{\ell} : \; \text{Lower reservoir tailwater level (m), taken from Section 1}")

    st.latex(r"h_{\text{draft}} : \; \text{Draft head (m), vertical distance from water surface to turbine centreline}")


# ----------------- Inputs that use Step 1 values -----------------
# y_at_Hg is from your fitted purple curve (likely negative). Use its magnitude for draft head.
hdraft_from_fit = float(abs(y_at_Hg)) if not np.isnan(y_at_Hg) else float("nan")

st.subheader("Set Turbine Center Elevation")

col_a, col_b = st.columns(2)

with col_a:
    # Pull Lower TWL directly from Section 1
    lwl = float(TWL_l) if not np.isnan(TWL_l) else st.number_input(
        "Lower TWL (m) — fallback input",
        min_value=0.0, value=420.0, step=0.5, help="Auto-filled from Section 1 when available."
    )

with col_b:
    # Manual override control
    allow_override = st.checkbox("Manually override draft head?", value=False, help="Uncheck to use fitted value.")
    
    # Resolve the value to show in the input (positive number)
    default_hdraft = hdraft_from_fit if not np.isnan(hdraft_from_fit) else 5.0
    
    # Keep a single source of truth in session_state
    if "h_draft" not in st.session_state:
        st.session_state["h_draft"] = default_hdraft
    
    # If we have a new fit value and override is off, sync to fit
    if (not allow_override) and (not np.isnan(hdraft_from_fit)):
        st.session_state["h_draft"] = hdraft_from_fit
    
    # Show the number input; disable it unless override is on
    st.number_input(
        "Draft head below lower TWL (m)",
        min_value=0.0, max_value=100.0,
        value=float(st.session_state["h_draft"]),
        step=0.5,
        key="h_draft",
        disabled=not allow_override
    )

# Lower reservoir tailwater level (m), from Section 1
lower_TWL = float(TWL_l) if not np.isnan(TWL_l) else st.number_input(
    "Lower TWL (m) — fallback input",
    min_value=0.0, value=420.0, step=0.5,
    help="Auto-filled from Section 1 when available."
)

# Draft head
h_draft = float(st.session_state["h_draft"])

# Turbine center elevation
turbine_abs = lower_TWL - h_draft

c1, c2 = st.columns(2)
with c1:
    st.metric("Lower Reservoir TWL", f"{lower_TWL:.2f} m")
with c2:
    st.metric("Draft head used (from fit unless overridden)", f"{h_draft:.2f} m")

# Duplicate metric if you want to emphasise
st.metric("Calculated Turbine CL elevation", f"{turbine_abs:.2f} m")


# ----------------- Simple vertical sketch -----------------
st.subheader("Water Level Diagram")
fig2, ax2 = plt.subplots(figsize=(10, 4))

# Water line at TWL
ax2.axhline(lower_TWL, color='blue', linewidth=3, label="Lower TWL")
# Turbine CL line
ax2.axhline(turbine_abs, color='red', linewidth=2, linestyle='-', label="Turbine CL")

# Draft head arrow
ax2.annotate(
    '', xy=(0.5, lower_TWL), xytext=(0.5, turbine_abs),
    arrowprops=dict(arrowstyle='<->', color='green', linewidth=2)
)
ax2.text(
    0.55, (lower_TWL + turbine_abs) / 2,
    f"Draft head = {h_draft:.1f} m",
    fontsize=12, va='center', color='green'
)

# Formatting
pad = max(2.0, 0.5 * max(1.0, h_draft))
ax2.set_ylim(turbine_abs - pad, lower_TWL + pad)
ax2.set_xlim(0, 1)
ax2.set_yticks([turbine_abs, lower_TWL])
ax2.set_yticklabels([f"Turbine CL: {turbine_abs:.2f} m", f"Lower TWL: {lower_TWL:.2f} m"])
ax2.set_xticks([])
ax2.set_title("Turbine Position Relative to Lower TWL", fontsize=12)
ax2.grid(True, axis='y', alpha=0.3)
ax2.legend(loc='upper right')

st.pyplot(fig2)
    
# ------------------------------- Section 3: Penstock & Moody -------------------------
st.header("3) Head Loss of Hydraulic System")
st.subheader("Major Water Loss (with Moody)")
c1, c2 = st.columns(2)
with c1:
    P_design = st.number_input("Design power (MW)", 10.0, 5000.0, float(st.session_state.get("design_power", P_design)))
with c2:
    P_max = st.number_input("Maximum power (MW)", 10.0, 6000.0, float(st.session_state.get("max_power", 600.0)), 10.0)
    st.caption("💡 Hint: Maximum power should generally equal the Design power. Please check consistency.")

st.subheader("Friction factor mode")
mode_f = st.radio("Choose how to set Darcy friction factor f:",
                  ["Manual (slider)", "Compute from Moody (Swamee–Jain)"], index=1)
if mode_f == "Manual (slider)":
    f = st.slider("Friction factor f (Darcy)", 0.005, 0.03, 0.015, 0.001)
    rough_choice = "—"; eps = None; T_C = None
else:
    colA, colB, colC = st.columns(3)
    with colA:
        T_C = st.number_input("Water temperature (°C)", 0.0, 60.0, float(st.session_state.get("T_C", 20.0)), 0.5)
    with colB:
        rl = roughness_library()
        rough_choice = st.selectbox("Material / absolute roughness ε (m)", list(rl.keys()),
                                    index=list(rl.keys()).index(st.session_state.get("rough_choice","Concrete (smooth)")))
    with colC:
        eps = st.number_input("Custom ε (m)", 0.0, 0.01,
                              float(st.session_state.get("eps_custom", rl[rough_choice] if rl[rough_choice] else 0.00030)),
                              0.00001, format="%.5f") if rough_choice == "Custom..." else rl[rough_choice]

# --- Absolute roughness reference + live ε/D for current D_pen ---
with st.expander("Refer to ε (click to expand)"):

    st.markdown("### Roughness reference & your current ε/D")

    # Build reference table (from the same values your app uses)
    rlib = roughness_library()
    rows = []
    for mat, eps_val in rlib.items():
        if eps_val is None:  # skip "Custom..."
            continue
        rows.append({
            "Material": mat,
            "ε (mm)": eps_val * 1e3,
            "ε (m)": eps_val,
            "ε/D (for your D)": (eps_val / D_pen) if D_pen else float("nan"),
        })
    df_eps = pd.DataFrame(rows)

    # Display the table
    st.dataframe(
        df_eps,
        use_container_width=True,
        column_config={
            "ε (mm)": st.column_config.NumberColumn(format="%.3f"),
            "ε (m)": st.column_config.NumberColumn(format="%.6f"),
            "ε/D (for your D)": st.column_config.NumberColumn(format="%.6f"),
        }
    )

    # Show the currently selected roughness & ε/D (works for both preset & custom)
    eps_current = None
    if mode_f == "Manual (slider)":
        st.caption("Friction factor set manually; roughness not used for f.")
    else:
        eps_current = eps if eps is not None else rlib.get(rough_choice, None)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Selected ε (mm)", f"{(eps_current*1e3):.3f}" if eps_current else "—")
    with col_r2:
        rr = (eps_current / D_pen) if (eps_current and D_pen) else float("nan")
        st.metric("Relative roughness ε/D (–)", f"{rr:.6f}" if not np.isnan(rr) else "—")


with st.expander("What is ε/D? (click to expand)"):
    st.markdown("""
**Relative roughness** \\( \\varepsilon / D \\) compares the wall roughness height \\( \\varepsilon \\)
to the pipe diameter \\( D \\). It is dimensionless and, together with the Reynolds number
\\( \\mathrm{Re} \\), is used to determine the Darcy friction factor \\( f \\) from a Moody diagram.
""")
    st.latex(r"\text{Relative roughness} \;=\; \frac{\varepsilon}{D}")
    st.latex(r"f \;=\; f\!\left(\mathrm{Re},\, \frac{\varepsilon}{D}\right)\quad\text{(turbulent flow)}")

    st.markdown("**Typical absolute roughness \\(\\varepsilon\\) (order of magnitude):**")
    st.markdown(r"""
- PVC/HDPE: \(1.5 \times 10^{-6}\,\text{m}\)  
- New steel (welded): \(4.5 \times 10^{-5}\,\text{m}\)  
- Concrete (smooth): \(3.0 \times 10^{-4}\,\text{m}\)  
- Rock tunnel (good lining): \(1.0 \times 10^{-3}\,\text{m}\)
""")

    st.caption("Teaching sources: USBR (1987), AWWA, ASCE/USACE typical roughness tables.")

def compute_block(P_MW, h_span, Ksum, hf_guess=30.0):
    """Two-pass iteration to refine f, Re, hf_major, hf_minor, and hf_total for given Ksum."""
    A = area_circle(D_pen)

    # ---- Pass 1 (initial guess) ----
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

    # split losses
    hf_major = headloss_darcy(f_used, L_pen, D_pen, v, Ksum=0.0)
    hf_minor = headloss_darcy(0.0, L_pen, D_pen, v, Ksum=Ksum)
    hf_total = hf_major + hf_minor

    # ---- Pass 2 (refined with hf_total) ----
    h_draft = float(st.session_state.get("h_draft", 0.0))
    h_net2 = h_span - hf_total - h_draft
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

    # recompute split losses
    hf_major2 = headloss_darcy(f_used2, L_pen, D_pen, v2, Ksum=0.0)
    hf_minor2 = headloss_darcy(0.0, L_pen, D_pen, v2, Ksum=Ksum)
    hf_total2 = hf_major2 + hf_minor2

    return dict(
        f=f_used2, Re=Re2, v=v2,
        Q_total=Q_total2, Q_per=Q_per2,
        h_net=h_net2,
        hf=hf_total2,          # total
        hf_major=hf_major2,    # new
        hf_minor=hf_minor2,    # new
        rel_rough=(safe_div(eps, D_pen) if mode_f != "Manual (slider)" else None)
    )


# Compute ignoring local losses first (Ksum=0) — clean view of Q & v
out_design_flow = compute_block(P_design, gross_head, 0.0, hf_guess=20.0)
out_max_flow    = compute_block(P_max,    min_head,  0.0, hf_guess=30.0)

# ---------- Reynolds number — quick check (AFTER flows exist) ----------
st.subheader("Reynolds number — quick check")

def _coerce_flow(res):
    """Return a dict with keys Q_total, Qp, v, Re from either a dict or a tuple/list."""
    if isinstance(res, dict):
        return {
            "Q_total": res.get("Q_total", float("nan")),
            "Qp":      res.get("Q_per",  res.get("Qp", float("nan"))),
            "v":       res.get("v", float("nan")),
            "Re":      res.get("Re", float("nan")),
        }
    elif isinstance(res, (list, tuple)):
        vals = list(res) + [float("nan")] * 4
        return {"Q_total": vals[0], "Qp": vals[1], "v": vals[2], "Re": vals[3]}
    else:
        return {"Q_total": float("nan"), "Qp": float("nan"), "v": float("nan"), "Re": float("nan")}

design_flow = _coerce_flow(out_design_flow)
max_flow    = _coerce_flow(out_max_flow)

T_for_nu = T_C if (mode_f != "Manual (slider)" and T_C is not None) else 20.0
nu_used  = water_nu_kinematic_m2s(T_for_nu)

def Re_quick(Q_total, N, D, nu):
    """Re = ( (Q_total/N) / (πD²/4) ) * D / ν  =  4 Q_total / (π N D ν)"""
    A = math.pi * D**2 / 4.0
    v = (Q_total / max(N, 1)) / A
    return v * D / nu

Re_quick_design = Re_quick(design_flow["Q_total"], N_pen, D_pen, nu_used)
Re_quick_max    = Re_quick(max_flow["Q_total"],    N_pen, D_pen, nu_used)

def pct_diff(a, b):
    return float("nan") if (np.isnan(a) or np.isnan(b) or b == 0) else 100.0 * (a - b) / b

re_table = pd.DataFrame({
    "Case": ["Design", "Maximum"],
    "Re (two-pass)": [design_flow["Re"], max_flow["Re"]],
    "Re (quick 4Q/(πNDν))": [Re_quick_design, Re_quick_max],
    "% diff (quick vs two-pass)": [
        pct_diff(Re_quick_design, design_flow["Re"]),
        pct_diff(Re_quick_max,    max_flow["Re"])
    ],
})
st.dataframe(
    re_table, use_container_width=True,
    column_config={
        "Re (two-pass)": st.column_config.NumberColumn(format="%.0f"),
        "Re (quick 4Q/(πNDν))": st.column_config.NumberColumn(format="%.0f"),
        "% diff (quick vs two-pass)": st.column_config.NumberColumn(format="%.2f %%"),
    }
)
st.caption(f"Quick-Re uses ν at T = {T_for_nu:.1f} °C.")

with st.expander("Show equations used (Reynolds quick check)"):

    # Cross-sectional area
    st.latex(r"A = \frac{\pi D^{2}}{4}")
    st.caption("The cross-sectional area of the penstock (A) is calculated from its diameter (D).")

    # Flow per penstock
    st.latex(r"Q_p = \frac{Q_\text{total}}{N_\text{pen}}")
    st.caption("The discharge per penstock (Qp) is obtained by dividing the total discharge (Qtotal) by the number of penstocks (Npen).")

    # Velocity from flow and area
    st.latex(r"v = \frac{Q_p}{A}")
    st.caption("The mean velocity (v) inside each penstock is calculated using Qp and A. "
               "⚠️ Note: This is the *calculated velocity*, not the target velocity set by the design slider.")

    # Reynolds number
    st.latex(r"\mathrm{Re} = \frac{v D}{\nu}")
    st.caption("The Reynolds number is calculated from the mean velocity (v), penstock diameter (D), and kinematic viscosity (ν).")

    # Combined formula
    st.latex(r"\boxed{\;\mathrm{Re}=\dfrac{4\,Q_\text{total}}{\pi\,N_\text{pen}\,D\,\nu}\;}")
    st.caption("Combining the above expressions gives a compact formula for Re in terms of total discharge, "
               "number of penstocks, diameter, and viscosity.")

    # Optional: compare slider velocity vs calculated velocity
    if 'v_target' in locals() and 'v_calc' in locals():
        st.write(f"🎯 Target velocity (slider): {v_target:.2f} m/s")
        st.write(f"📐 Calculated mean velocity: {v_calc:.2f} m/s")
        if abs(v_target - v_calc) > 0.5:
            st.warning("The calculated velocity differs significantly from the target velocity set by the slider.")
        else:
            st.success("Calculated velocity is close to the target velocity.")

# ---------- Mini Moody chart (AFTER flows exist) ----------
if (mode_f != "Manual (slider)"):
    st.subheader("Moody Diagram for Friction Factor")
    Re_vals = np.logspace(3, 8, 300)
    epsD_list = [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]

    # Smaller figure size
    fig_m, axm = plt.subplots(figsize=(4.5, 3.0))   # 👈 reduced size
    for rr in epsD_list:
        f_line = [f_moody_swamee_jain(Re, rr) for Re in Re_vals]
        axm.plot(Re_vals, f_line, lw=1.2, label=f"ε/D={rr:g}")

    for label, out_pt in (("Design point", out_design_flow), ("Max point", out_max_flow)):
        Re_pt = out_pt.get("Re", np.nan)
        f_pt  = out_pt.get("f",  np.nan)
        if (not np.isnan(Re_pt)) and (not np.isnan(f_pt)):
            axm.scatter([Re_pt], [f_pt], s=40, zorder=5, label=label)

    axm.set_xscale("log"); axm.set_yscale("log")
    axm.set_xlabel("Reynolds number Re", fontsize=9)
    axm.set_ylabel("Darcy friction factor f", fontsize=9)
    axm.set_title("Moody chart (Swamee–Jain approx.)", fontsize=10)
    axm.tick_params(axis="both", labelsize=8)
    axm.grid(True, which="both", ls="--", alpha=0.3)
    axm.legend(loc="best", fontsize=7)
    st.pyplot(fig_m, clear_figure=True)

# ---------- Friction factor comparison (Swamee–Jain vs Colebrook) ----------
st.subheader("Friction factor comparison")

def colebrook_white_iter(Re, eps_over_D, itmax=50, tol=1e-10):
    """
    Solve 1/sqrt(f) = -2 log10( (ε/D)/3.7 + 2.51/(Re*sqrt(f)) )
    Fixed-point iteration on 1/sqrt(f). Returns NaN if inputs invalid.
    """
    Re = float(Re)
    rr = float(eps_over_D)
    if not np.isfinite(Re) or Re <= 0:
        return float("nan")
    # laminar shortcut
    if Re < 2000:
        return 64.0/Re
    # initial guess from Swamee–Jain for stability
    try:
        f0 = 0.25/(math.log10(rr/3.7 + 5.74/(Re**0.9)))**2
    except ValueError:
        f0 = 0.02
    x = 1.0/math.sqrt(max(f0, 1e-6))
    for _ in range(itmax):
        rhs = -2.0*math.log10(rr/3.7 + 2.51/(Re*x))
        x_new = rhs
        if abs(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    f = 1.0/(x*x)
    return float(f)

def f_haaland(Re, eps_over_D):
    """
    Haaland explicit correlation (turbulent, decent accuracy).
    """
    Re = float(Re)
    rr = float(eps_over_D)
    if not np.isfinite(Re) or Re <= 0:
        return float("nan")
    if Re < 2000:
        return 64.0/Re
    return ( -1.8*math.log10( (rr/3.7)**1.11 + 6.9/Re ) )**-2

# get relative roughness (if available)
if mode_f == "Manual (slider)":
    st.info("Friction factor is currently set **manually**. For the comparison below, "
            "the app uses your current D and the selected material ε (if any).")
    rr_used = float("nan") if eps is None or D_pen <= 0 else (eps / D_pen)
else:
    rr_used = float("nan") if D_pen <= 0 or eps is None else (eps / D_pen)

def pick_Re(two_pass_val, quick_val):
    """Prefer two-pass Re; fall back to quick formula if needed."""
    return two_pass_val if np.isfinite(two_pass_val) and two_pass_val > 0 else quick_val

Re_design_used = pick_Re(design_flow["Re"], Re_quick_design)
Re_max_used    = pick_Re(max_flow["Re"],    Re_quick_max)

rows = []
for case, Re_used in [("Design", Re_design_used), ("Maximum", Re_max_used)]:
    # f via Swamee–Jain (your app function already blends laminar/transitional)
    f_sj = f_moody_swamee_jain(Re_used, rr_used)
    # f via Colebrook–White (iterative)
    f_cb = colebrook_white_iter(Re_used, rr_used)
    # f via Haaland (optional extra)
    f_hl = f_haaland(Re_used, rr_used)

    def pdiff(a, b):
        return float("nan") if (not np.isfinite(a) or not np.isfinite(b) or b == 0) else 100.0*(a-b)/b

    rows.append({
        "Case": case,
        "Re (used)": Re_used,
        "ε/D (used)": rr_used,
        "f — Swamee–Jain": f_sj,
        "f — Colebrook (iter.)": f_cb,
        "f — Haaland": f_hl,
        "Δ% (Colebrook vs SJ)": pdiff(f_cb, f_sj),
        "Δ% (Haaland vs SJ)": pdiff(f_hl, f_sj),
    })

df_f = pd.DataFrame(rows)

st.dataframe(
    df_f,
    use_container_width=True,
    column_config={
        "Re (used)": st.column_config.NumberColumn(format="%.0f"),
        "ε/D (used)": st.column_config.NumberColumn(format="%.6f"),
        "f — Swamee–Jain": st.column_config.NumberColumn(format="%.5f"),
        "f — Colebrook (iter.)": st.column_config.NumberColumn(format="%.5f"),
        "f — Haaland": st.column_config.NumberColumn(format="%.5f"),
        "Δ% (Colebrook vs SJ)": st.column_config.NumberColumn(format="%.2f %%"),
        "Δ% (Haaland vs SJ)": st.column_config.NumberColumn(format="%.2f %%"),
    }
)

st.caption("Notes: Swamee–Jain is the explicit formula used by the app; "
           "Colebrook–White is the iterative reference; Haaland is another explicit correlation. "
           "All use the same Re and ε/D shown in the table.")

# ---------- References: friction factor & related methods ----------
with st.expander("What is the Darcy–Weisbach equation? (click to expand)"):
    st.markdown(
        r"""
The Darcy–Weisbach equation gives head loss due to wall friction:

$$
\Delta h_f \;=\; \lambda \;\frac{L}{d_h}\; \frac{v^{2}}{2g}
$$

where  
- $ \lambda $ = Darcy friction factor  
- $ L $ = pipe length (m)  
- $ d_h $ = hydraulic diameter (m)  
- $ v $ = mean velocity (m/s)  
- $ g $ = gravitational acceleration (9.81 m/s²)  
"""
    )

with st.expander("What is the Colebrook–White equation? (click to expand)"):
    st.markdown(
        r"""
The Colebrook–White (implicit) equation for turbulent flow relates $ \lambda $, $ Re $, and relative roughness $ \varepsilon/d_h $:

$$
\frac{1}{\sqrt{\lambda}}
= -2 \log_{10}\!\left(
\frac{\varepsilon}{3.7\,d_h}
+ \frac{2.51}{Re\,\sqrt{\lambda}}
\right)
$$

where  
- $ \varepsilon $ = absolute roughness (m)  
- $ d_h $ = hydraulic diameter (m)  
- $ Re = \dfrac{\rho v d_h}{\mu} $ = Reynolds number  
"""
    )

with st.expander("What is the Moody chart? (click to expand)"):
    st.markdown(
        r"""
The **Moody chart** is a graphical relation of the Darcy friction factor $ \lambda $ vs. Reynolds number $ Re $ and relative roughness $ \varepsilon/d_h $.

- Covers **laminar**, **transitional**, and **turbulent** regimes.  
- Lets you estimate $ \lambda $ without solving Colebrook–White numerically.  
"""
    )

with st.expander("What is the Haaland approximation? (click to expand)"):
    st.markdown(
        r"""
The **Haaland** explicit approximation (no iteration) for turbulent flow:

$$
\frac{1}{\sqrt{\lambda}}
\;\approx\;
-1.8 \,\log_{10}\!\left[
\left(\frac{\varepsilon}{3.7\,d_h}\right)^{1.11}
+ \frac{6.9}{Re}
\right]
$$

- Typical accuracy: **within ~1–2%** of Colebrook–White.  
- Widely used for quick engineering calculations.  
"""
    )

with st.expander("What is the Swamee–Jain explicit formula? (click to expand)"):
    st.markdown(
        r"""
The **Swamee–Jain** explicit correlation (turbulent) used in this app’s Moody helper:

$$
\lambda
\;=\;
\frac{0.25}{
\left[
\log_{10}\!\left(\dfrac{\varepsilon}{3.7\,d_h}
+\dfrac{5.74}{Re^{0.9}}\right)
\right]^2}
$$

- Convenient, accurate for fully turbulent ranges.  
"""
    )

with st.expander("How does relative roughness enter? (click to expand)"):
    st.markdown(
        r"""
**Relative roughness**:

$$
\frac{\varepsilon}{d_h}
$$

- $ \varepsilon $ = absolute roughness (m), $ d_h $ = hydraulic diameter (m).  
- Together with $ Re $, it determines $ \lambda $ via Colebrook–White / Haaland / Swamee–Jain or a Moody chart.  
"""
    )

# ---- Darcy–Weisbach major head loss (no local losses) ----
def major_head_loss(f_darcy: float, L: float, D_h: float, v: float, g: float = 9.81) -> float:
    """Δh_major = f * (L/D) * v^2 / (2g)  (m of water)"""
    if any(map(lambda x: x is None or x <= 0, [f_darcy, L, D_h, g])) or v is None:
        return float("nan")
    return f_darcy * (L / D_h) * (v**2) / (2.0 * g)

# --- Example: compute for your Design and Maximum cases (using your existing outputs)
h_major_design = major_head_loss(
    f_darcy = out_design_flow.get("f", float("nan")),
    L       = L_pen,
    D_h     = D_pen,
    v       = out_design_flow.get("v", float("nan"))
)

h_major_max = major_head_loss(
    f_darcy = out_max_flow.get("f", float("nan")),
    L       = L_pen,
    D_h     = D_pen,
    v       = out_max_flow.get("v", float("nan"))
)

# --- Show equation first --- 
st.markdown(
    r"""
    ### Darcy–Weisbach Equation for Major Head Loss

    $$
    \Delta h_f \;=\; \lambda \;\frac{L}{d_h}\; \frac{v^{2}}{2g}
    $$

    where  
    - $\lambda$ = Darcy friction factor  
    - $L$ = pipe length (m)  
    - $d_h$ = hydraulic diameter (m)  
    - $v$ = mean velocity (m/s)  
    - $g$ = gravitational acceleration (9.81 m/s²)  
    """
)
# Pull losses returned by the two-pass block
hf_model_design = out_design_flow.get("hf", float("nan"))
hf_model_max    = out_max_flow.get("hf",    float("nan"))

# Recompute Darcy–Weisbach loss from the displayed values
hf_dw_design = major_head_loss(
    f_darcy = out_design_flow.get("f", float("nan")),
    L       = L_pen,
    D_h     = D_pen,
    v       = out_design_flow.get("v", float("nan"))
)
hf_dw_max = major_head_loss(
    f_darcy = out_max_flow.get("f", float("nan")),
    L       = L_pen,
    D_h     = D_pen,
    v       = out_max_flow.get("v", float("nan"))
)

# Net head recomputed from gross head and DW loss
hnet_dw_design = gross_head - hf_dw_design
hnet_dw_max    = min_head   - hf_dw_max

df_hydraulics = pd.DataFrame({
    "Case": ["Design", "Maximum"],
    "Gross head H_g (m)":   [gross_head,                 min_head],
    # Model (two-pass) results
    "Net head h_net (model) (m)": [out_design_flow["h_net"], out_max_flow["h_net"]],
    "h_f (model two-pass) (m)":   [hf_model_design,          hf_model_max],
    # Recomputed single-shot DW using displayed f, v, D
    "Δh_major DW (recomp) (m)":   [hf_dw_design,             hf_dw_max],
    "Net head H_g−Δh (recomp) (m)":[hnet_dw_design,          hnet_dw_max],
    # Diagnostics
    "Δh diff (model − DW) (m)":   [hf_model_design - hf_dw_design,
                                   hf_model_max    - hf_dw_max],
    "f (Darcy)":                  [out_design_flow.get("f", float("nan")),
                                   out_max_flow.get("f",    float("nan"))],
    "Velocity v (m/s)":           [out_design_flow.get("v", float("nan")),
                                   out_max_flow.get("v",    float("nan"))],
    "L (m)":                      [L_pen, L_pen],
    "d_h (m)":                    [D_pen, D_pen],
    "Reynolds Re (–)":            [out_design_flow["Re"], out_max_flow["Re"]],
    "Per-penstock Q (m³/s)":      [out_design_flow["Q_per"], out_max_flow["Q_per"]],
    "Total Q (m³/s)":             [out_design_flow["Q_total"], out_max_flow["Q_total"]],
})

st.dataframe(
    df_hydraulics,
    use_container_width=True,
    column_config={
        "Gross head H_g (m)":              st.column_config.NumberColumn(format="%.2f"),
        "Net head h_net (model) (m)":      st.column_config.NumberColumn(format="%.2f"),
        "h_f (model two-pass) (m)":        st.column_config.NumberColumn(format="%.2f"),
        "Δh_major DW (recomp) (m)":        st.column_config.NumberColumn(format="%.2f"),
        "Net head H_g−Δh (recomp) (m)":    st.column_config.NumberColumn(format="%.2f"),
        "Δh diff (model − DW) (m)":        st.column_config.NumberColumn(format="%.2f"),
        "f (Darcy)":                       st.column_config.NumberColumn(format="%.4f"),
        "Velocity v (m/s)":                st.column_config.NumberColumn(format="%.2f"),
        "L (m)":                           st.column_config.NumberColumn(format="%.1f"),
        "d_h (m)":                         st.column_config.NumberColumn(format="%.3f"),
        "Reynolds Re (–)":                 st.column_config.NumberColumn(format="%.0f"),
        "Per-penstock Q (m³/s)":           st.column_config.NumberColumn(format="%.2f"),
        "Total Q (m³/s)":                  st.column_config.NumberColumn(format="%.2f"),
    }
)

st.caption(
    "‘Model’ uses the two-pass Moody/Swamee–Jain iteration inside the flow block. "
    "‘Recomp’ evaluates Darcy–Weisbach once using the displayed f, v and D. "
    "Small differences arise from iteration and rounding; the diagnostics column shows the gap."
)

# --- Step-by-step learning table (Design case) ---
# Use current penstock diameter, flows, and viscosity already computed above
A_design   = area_circle(D_pen)                              # A = πD²/4
Q_total_d  = design_flow.get("Q_total", float("nan"))        # from two-pass block
Qp_d       = design_flow.get("Qp",      float("nan"))        # per-penstock flow
v_calc     = design_flow.get("v",       float("nan"))        # mean velocity (computed)
Re_calc    = design_flow.get("Re",      float("nan"))        # two-pass Re
Re_compact = Re_quick_design                                 # 4Q/(π N D ν)
Re_diffpct = pct_diff(Re_compact, Re_calc)                   # consistency check

# Small teaching table
learn_tbl = pd.DataFrame({
    "Quantity": [
        "Cross-sectional area A (m²)",
        "Per-penstock flow Qₚ (m³/s)",
        "Mean velocity v (m/s)",
        "Re (two-pass) = vD/ν (–)",
        "Re (compact) = 4Q/(πNDν) (–)",
        "Δ% (compact vs two-pass)"
    ],
    "Value": [
        f"{A_design:.3f}" if not np.isnan(A_design) else "—",
        f"{Qp_d:.3f}"     if not np.isnan(Qp_d)     else "—",
        f"{v_calc:.3f}"   if not np.isnan(v_calc)   else "—",
        f"{Re_calc:.0f}"  if not np.isnan(Re_calc)  else "—",
        f"{Re_compact:.0f}" if not np.isnan(Re_compact) else "—",
        (f"{Re_diffpct:.2f} %" if not np.isnan(Re_diffpct) else "—")
    ]
})

with st.expander("Step-by-step calculation table (Design case)"):
    st.dataframe(
        learn_tbl,
        use_container_width=True
    )
    # Show the actual equations neatly (for learning)
    st.markdown("**Equations used**")
    st.latex(r"A = \frac{\pi D^{2}}{4}")
    st.latex(r"Q_p = \frac{Q_\text{total}}{N_\text{pen}}")
    st.latex(r"v = \frac{Q_p}{A}")
    st.latex(r"\mathrm{Re}_{\text{two-pass}} = \frac{v D}{\nu}")
    st.latex(r"\mathrm{Re}_{\text{compact}} = \frac{4\,Q_\text{total}}{\pi\,N_\text{pen}\,D\,\nu}")

# (Optional) gentle hint comparing calculated velocity to any target-velocity choice elsewhere
if "v_target" in st.session_state and not np.isnan(v_calc):
    v_t = float(st.session_state["v_target"])
    if v_t > 0:
        st.caption(
            f"🎯 Target velocity (if set elsewhere): **{v_t:.2f} m/s**  •  "
            f"📐 Calculated mean velocity: **{v_calc:.2f} m/s**"
        )


# --------------- Section 3: Minor Head Loss ----------------
st.subheader("Minor Head Loss by Local loss components (ΣK)")

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

# Compute with ΣK to show h_f_minor and f (two-pass block)
out_design = compute_block(P_design, gross_head, K_sum_global, hf_guess=25.0)
out_max    = compute_block(P_max,    min_head,  K_sum_global, hf_guess=40.0)

results_losses = pd.DataFrame({
    "Case": ["Design", "Maximum"],
    "Darcy f (-)": [out_design["f"], out_max["f"]],
    "Reynolds Re (-)": [out_design["Re"], out_max["Re"]],
    "Head loss h_f minor (m)": [out_design["hf_minor"], out_max["hf_minor"]],
    "Net head h_net (m)": [out_design["h_net"], out_max["h_net"]],
})

st.dataframe(
    results_losses, use_container_width=True,
    column_config={
        "Darcy f (-)": st.column_config.NumberColumn(format="%.004f"),
        "Reynolds Re (-)": st.column_config.NumberColumn(format="%.0f"),
        "Head loss h_f minor (m)": st.column_config.NumberColumn(format="%.2f"),
        "Net head h_net (m)": st.column_config.NumberColumn(format="%.2f"),
    }
)

st.header("4) Effective Net Head")

with st.expander("Head Loss & Net Head Equations (click to expand)"):

    st.markdown("**Gross head (difference in reservoir levels):**")
    st.latex(r"H_\text{gross} = \text{NWL} - \text{TWL}")

    st.latex(r"h_f = h_{f,\text{major}} + h_{f,\text{minor}}")

    st.markdown("**Major head loss (Darcy–Weisbach):**")
    st.latex(r"h_{f,\text{major}} = f \cdot \frac{L}{D} \cdot \frac{v^2}{2g}")

    st.markdown("**Minor head loss (Local loss components ΣK):**")
    st.latex(r"h_{f,\text{minor}} = \Sigma K \cdot \frac{v^2}{2g}")

    st.markdown("**Net head (effective):**")
    st.latex(r"H_\text{net} = H_\text{gross} - h_f - h_\text{draft}")



# ---------------- Show numerical results ----------------
st.subheader("Calculated Results")

hf_major = out_design.get("hf_major", float("nan"))   # Darcy–Weisbach
hf_minor = out_design.get("hf_minor", float("nan"))   # ΣK losses
hf_total = out_design.get("hf", float("nan"))         # total = major + minor
H_gross  = float(st.session_state.get("gross_head", float("nan")))  # <-- from Section 1
H_net    = out_design.get("h_net", float("nan"))
h_draft  = float(st.session_state.get("h_draft", float("nan")))

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("h_f major (m)", f"{hf_major:.2f}")
c2.metric("h_f minor (m)", f"{hf_minor:.2f}")
c3.metric("h_f total (m)", f"{hf_total:.2f}")
c4.metric("Draft head (m)", f"{h_draft:.2f}")
c5.metric("Gross head (m)", f"{H_gross:.2f}")
c6.metric("Net head (m)", f"{H_net:.2f}")


# ---------------- Diameter Estimator and Verification ----------------
st.header("5) Penstock Diameter and velocity Verification")

# Velocity checks
st.subheader("Velocity checks (USBR guidance)")
v_design = out_design_flow["v"]; v_max = out_max_flow["v"]
c1, c2 = st.columns(2)
with c1:
    st.metric("v_design (m/s)", f"{v_design:.2f}")
    st.metric("v_max (m/s)", f"{v_max:.2f}")
with c2:
    st.markdown("- **Recommended range:** 4–6 m/s (concrete penstocks)")
    st.markdown("- **Absolute max:** ~7 m/s (short duration)")

if v_max > 7.0:
    st.error("⚠️ Dangerous velocity (exceeds ~7 m/s). Revisit D or layout.")
elif v_max > 6.0:
    st.warning("⚠️ Above recommended 6 m/s. Acceptable only for short periods.")
elif v_max >= 4.0:
    st.success("✓ Within recommended 4–6 m/s range.")
else:
    st.info("ℹ️ Low velocity (<4 m/s): safe but potentially uneconomic (oversized).")

# Collapsible equations (for Section 3)
with st.expander("Velocity guidance (USBR)"):
    st.markdown(
        """
        - Low-pressure steel penstocks: *3 - 5 m/s*  
        - Medium-pressure steel penstocks: *5 - 7 m/s*  
        - High-pressure steel penstocks: *7 - 10 m/s*  

        *Reference: USBR, Design of Small Dams, 3rd Ed. (1987), Ch. 10 - Penstocks.*
        """
    )
# --- Get per-penstock design flow robustly ---
Qp_for_sizing = out_design_flow.get("Q_per", float("nan"))
if (np.isnan(Qp_for_sizing) or Qp_for_sizing <= 0) and not np.isnan(out_design_flow.get("Q_total", float("nan"))):
    # fallback from total flow if available
    try:
        Qp_for_sizing = out_design_flow["Q_total"] / max(1, int(N_pen))
    except Exception:
        Qp_for_sizing = float("nan")

# Show what we're sizing with
st.metric("Design per-penstock flow \(Q_p\) (m³/s)",
          f"{Qp_for_sizing:.3f}" if not np.isnan(Qp_for_sizing) else "—")

tabA, tabB, tabC = st.tabs(["Chart extrapolation", "Velocity target", "Head-loss target"])

# ---------- A) Chart extrapolation ----------
with tabA:
    a_fit, b_fit, D_of_Q = fit_extrapolate_Q_to_D(Q_chart, D_chart)
    D_ext = float(D_of_Q(Qp_for_sizing)) if (Qp_for_sizing > 0) else float("nan")

    st.write(f"Fitted curve: **D ≈ {a_fit:.3f} · Q^{b_fit:.3f}**  (Q in m³/s, D in m)")
    st.metric("Suggested D (m)", f"{D_ext:.2f}" if not np.isnan(D_ext) else "—")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Q_chart, D_chart, 'bo', ms=8, label="Reference data")
    Q_range = np.linspace(0, 500, 500)
    ax.plot(Q_range, a_fit * Q_range**b_fit, 'r-', lw=2,
            label=f"Fit: D = {a_fit:.3f}·Q^{b_fit:.3f}")
    if not np.isnan(D_ext):
        ax.plot(Qp_for_sizing, D_ext, 'go', ms=10, label=f"Design Qp={Qp_for_sizing:.1f}")
        ax.annotate(f"D={D_ext:.2f} m", (Qp_for_sizing, D_ext),
                    xytext=(10, -15), textcoords="offset points",
                    ha='left', arrowprops=dict(arrowstyle="->", color="green"))
    ax.set_xlim(0, 500); ax.set_ylim(0, 12)
    ax.set_xlabel("Per-penstock discharge Qp (m³/s)")
    ax.set_ylabel("Penstock diameter D (m)")
    ax.set_title("Diameter vs discharge")
    ax.grid(True, ls="--", alpha=0.7); ax.legend(loc="upper left")
    st.pyplot(fig)

    if st.button("Apply suggested D (chart fit)", key="btn_apply_chart"):
        if not np.isnan(D_ext):
            st.session_state["D_pen"] = float(D_ext)
            st.success(f"Applied D = {D_ext:.2f} m to the Penstock Geometry panel.")
        else:
            st.warning("Need a valid Qp to compute D from the chart fit.")

# ---------- B) Velocity target ----------
with tabB:
    V_target = st.slider("Target velocity V (m/s)", 2.0, 8.0, 4.5, 0.1,
                         help="Pick an operating velocity (see velocity guidance above).")
    D_vel = D_from_velocity(Qp_for_sizing, V_target) if (Qp_for_sizing > 0) else float("nan")
    st.metric("Suggested D (m)", f"{D_vel:.2f}" if not np.isnan(D_vel) else "—")

    if st.button("Apply suggested D (velocity)", key="btn_apply_vel"):
        if not np.isnan(D_vel):
            st.session_state["D_pen"] = float(D_vel)
            st.success(f"Applied D = {D_vel:.2f} m to the Penstock Geometry panel.")
        else:
            st.warning("Set design power, gross head, ηₜ and number of penstocks first so Qp is available.")

# ---------- C) Head-loss target ----------
with tabC:
    hf_allow = st.number_input("Allowable head loss h_f (m)", 1.0, 100.0, 15.0, 0.5,
                               help="Total Darcy–Weisbach + local losses allowance.")
    eps_used = eps if (mode_f != "Manual (slider)" and eps is not None) else 3e-4
    T_used   = T_C if (mode_f != "Manual (slider)" and T_C is not None) else 15.0

    if Qp_for_sizing > 0 and L_pen > 0:
        D_iter, f_it, Re_it, v_it, hf_it = D_from_headloss(Qp_for_sizing, L_pen, hf_allow,
                                                           eps=eps_used, Ksum=K_sum_global, T_C=T_used)
    else:
        D_iter, f_it, Re_it, v_it, hf_it = (float("nan"),)*5

    st.metric("Suggested D (m)", f"{D_iter:.2f}" if not np.isnan(D_iter) else "—")
    st.caption(f"At that D: f≈{f_it:.4f}, Re≈{Re_it:.2e}, v≈{v_it:.2f} m/s, h_f≈{hf_it:.2f} m")

    if st.button("Apply suggested D (head-loss)", key="btn_apply_hloss"):
        if not np.isnan(D_iter):
            st.session_state["D_pen"] = float(D_iter)
            st.success(f"Applied D = {D_iter:.2f} m to the Penstock Geometry panel.")
        else:
            st.warning("Need valid Qp and L to run the head-loss-target method.")


# ---- Figures / Equations reference  ----
with st.expander("Show figures / equations used"):
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
st.subheader("5) System Head & Power Curves (didactic)")

# Gross head from Section 1
H_gross = gross_head

# Discharge range (0 to 120% of max design Q)
Q_max_total = out_max.get("Q_total", np.nan)
if np.isnan(Q_max_total) or Q_max_total <= 0:
    Q_max_total = max(1.0, (P_max * 1e6) / (RHO * G * max(H_gross, 1.0) * max(eta_t, 0.6)))
Q_grid = np.linspace(0, 1.2 * Q_max_total, 140)

# Compute net head curve (gross - head losses)
hf_list = []
H_net_list = []
for Q in Q_grid:
    # discharge per penstock
    Q_per = safe_div(Q, N_pen)
    v = safe_div(Q_per, area_circle(D_pen))

    # friction factor
    if mode_f == "Manual (slider)":
        f_used = f
    else:
        nu = water_nu_kinematic_m2s(T_C)
        Re = safe_div(v * D_pen, nu)
        rel_rough = safe_div(eps, D_pen)
        f_used = f_moody_swamee_jain(Re, rel_rough)

    # head loss
    hf = headloss_darcy(f_used, L_pen, D_pen, v, Ksum=out_design.get("Ksum", 0.0))
    hf_list.append(hf)
    H_net_list.append(H_gross - hf)

hf_array = np.array(hf_list)
H_net_array = np.array(H_net_list)

# Power curves
P_gross = RHO * G * Q_grid * H_gross * eta_t / 1e6  # MW
P_net   = RHO * G * Q_grid * H_net_array * eta_t / 1e6

# --- Plot Heads ---
fig_head = make_subplots(specs=[[{"secondary_y": False}]])
fig_head.add_trace(go.Scatter(x=Q_grid, y=[H_gross]*len(Q_grid),
                              name="Gross Head", line=dict(color="blue", dash="dot")))
fig_head.add_trace(go.Scatter(x=Q_grid, y=H_net_array,
                              name="Net Head", line=dict(color="red", width=3)))
fig_head.update_layout(
    title="Head vs Discharge",
    xaxis_title="Total Discharge Q (m³/s)",
    yaxis_title="Head (m)",
    hovermode="x unified",
    height=420
)
st.plotly_chart(fig_head, use_container_width=True)

# --- Plot Power ---
fig_power = make_subplots(specs=[[{"secondary_y": False}]])
fig_power.add_trace(go.Scatter(x=Q_grid, y=P_gross,
                               name="Gross Power", line=dict(color="blue", dash="dot")))
fig_power.add_trace(go.Scatter(x=Q_grid, y=P_net,
                               name="Net Power", line=dict(color="red", width=3)))
fig_power.add_vline(x=out_design["Q_total"], line=dict(color="green", dash="dash"), annotation_text="Design Q")
fig_power.add_vline(x=out_max["Q_total"], line=dict(color="black", dash="dash"), annotation_text="Max Q")
fig_power.update_layout(
    title="Power vs Discharge",
    xaxis_title="Total Discharge Q (m³/s)",
    yaxis_title="Power (MW)",
    hovermode="x unified",
    height=420
)
st.plotly_chart(fig_power, use_container_width=True)

# ===============================================
# 6) Confinement Check (Norwegian Confinement Criteria)
# ===============================================
st.header("6) Confinement Check (Norwegian Criteria)")

st.markdown(
    "This section checks tunnel confinement stability using the Norwegian criteria. "
    "It evaluates both the **factor of safety** for given rock covers and the **required rock cover** "
    "for a target safety factor."
)

# --- User Inputs ---
st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    hs = st.number_input("Hydrostatic head h_s (m)", value=300.0, min_value=0.0)
    alpha = st.number_input("Tunnel inclination α (degrees)", value=20.0, min_value=0.0, max_value=90.0)
    beta = st.number_input("Slope angle β (degrees)", value=40.0, min_value=0.0, max_value=90.0)

with col2:
    gamma_w = st.number_input("Unit weight of water γ_w (kN/m³)", value=9.81, min_value=0.0)
    gamma_r = st.number_input("Unit weight of rock γ_r (kN/m³)", value=26.0, min_value=0.0)

with col3:
    C_RV = st.number_input("Given vertical cover C_RV (m)", value=214.0, min_value=0.0)
    C_RM = st.number_input("Given minimum cover C_RM (m)", value=182.0, min_value=0.0)
    F_req = st.number_input("Required Factor of Safety F_req", value=1.2, min_value=0.1)

# --- Calculations ---
def confinement_factor(C_RV, C_RM, hs, gamma_r, gamma_w, alpha, beta):
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    F_RV = (C_RV * gamma_r * np.cos(alpha_rad)) / (hs * gamma_w)
    F_RM = (C_RM * gamma_r * np.cos(beta_rad)) / (hs * gamma_w)
    return F_RV, F_RM

def confinement_cover(F_req, hs, gamma_r, gamma_w, alpha, beta):
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    C_RV_req = F_req * hs * gamma_w / (gamma_r * np.cos(alpha_rad))
    C_RM_req = F_req * hs * gamma_w / (gamma_r * np.cos(beta_rad))
    return C_RV_req, C_RM_req

F_RV, F_RM = confinement_factor(C_RV, C_RM, hs, gamma_r, gamma_w, alpha, beta)
C_RV_req, C_RM_req = confinement_cover(F_req, hs, gamma_r, gamma_w, alpha, beta)

# --- Equations ---
with st.expander("Confinement Criteria Equations (click to expand)"):

    st.markdown("**Safety factors:**")
    st.latex(r"F_{RV} = \frac{C_{RV} \cdot \gamma_r \cdot \cos(\alpha)}{h_s \cdot \gamma_w}")
    st.latex(r"F_{RM} = \frac{C_{RM} \cdot \gamma_r \cdot \cos(\beta)}{h_s \cdot \gamma_w}")

    st.markdown("**Required covers for a target factor of safety:**")
    st.latex(r"C_{RV,req} = \frac{F_{req} \cdot h_s \cdot \gamma_w}{\gamma_r \cdot \cos(\alpha)}")
    st.latex(r"C_{RM,req} = \frac{F_{req} \cdot h_s \cdot \gamma_w}{\gamma_r \cdot \cos(\beta)}")

# --- Results ---
st.subheader("Calculated Results")

results = {
    "Given Cover (m)": [C_RV, C_RM],
    "Factor of Safety": [F_RV, F_RM],
    "Required Cover (m)": [C_RV_req, C_RM_req],
}

df_conf = pd.DataFrame(results, index=["Vertical (RV)", "Minimum (RM)"])
st.dataframe(df_conf.style.format("{:.2f}"))

# --- Check Compliance ---
if F_RV >= F_req and F_RM >= F_req:
    st.success("✅ Both confinement criteria are satisfied.")
else:
    st.error("⚠️ One or both confinement criteria are **not satisfied**.")


# ===============================
# --- Section 7: Pressure Tunnel Lining Stress ---
st.header("7) Pressure Tunnel: Lining Stress")

gamma_w = 9800.0  # N/m³ (unit weight of water)

# ------------------ Grouped inputs ------------------
st.subheader("Input Parameters")

with st.expander("Hydraulic Heads", expanded=True):
    c1, c2, c3 = st.columns(3)
    h_s = c1.number_input("Hydrostatic head to crown h_s (m)", value=204.0)
    h_w = c2.number_input("Groundwater level head h_w (m)", value=150.0)
    eta = c3.number_input("Effective pore pressure factor η", value=1.0)

with st.expander("Geometry", expanded=True):
    c1, c2 = st.columns(2)
    d_p = c1.number_input("Penstock diameter (m)", value=3.0)
    t_l = c2.number_input("Lining thickness (m)", value=0.5)

with st.expander("Concrete Properties", expanded=False):
    c1, c2, c3 = st.columns(3)
    E_c  = c1.number_input("Concrete modulus E_c (Pa)", value=3.5e10, format="%.2e")
    v_c  = c2.number_input("Concrete Poisson’s ratio ν_c", value=0.17)
    ft_MPa = c3.number_input("Concrete tensile strength f_t (MPa)", value=2.0)

with st.expander("Rock Properties", expanded=False):
    c1, c2 = st.columns(2)
    E_r = c1.number_input("Rock modulus E_r (Pa)", value=2.7e10, format="%.2e")
    v_r = c2.number_input("Rock Poisson’s ratio ν_r", value=0.20)

# ------------------ Calculations ------------------
try:
    r_i = d_p / 2.0                  # inner radius (m)
    r_o = r_i + t_l                  # outer radius (m)

    if r_o <= r_i:
        st.error("Invalid geometry: outer radius must be larger than inner radius.")
    else:
        # Pressures
        p_i = gamma_w * h_s              # internal water pressure (Pa)
        p_e = gamma_w * h_w              # external water pressure (Pa)
        p_f = eta * (p_i - p_e)          # effective pore pressure (Pa)

        # --- Stress functions (Lamé solution) ---
        def radial_stress(pi, pe, ri, ro, r):
            return (
                (pi * ri**2 - pe * ro**2) / (ro**2 - ri**2)
                - ((pi - pe) * ri**2 * ro**2) / ((ro**2 - ri**2) * r**2)
            )

        def hoop_stress(pi, pe, ri, ro, r):
            return (
                (pi * ri**2 - pe * ro**2) / (ro**2 - ri**2)
                + ((pi - pe) * ri**2 * ro**2) / ((ro**2 - ri**2) * r**2)
            )

        # Inner & outer faces
        sigma_theta_i = hoop_stress(p_i, p_e, r_i, r_o, r_i)
        sigma_theta_o = hoop_stress(p_i, p_e, r_i, r_o, r_o)
        sigma_r_i = radial_stress(p_i, p_e, r_i, r_o, r_i)
        sigma_r_o = radial_stress(p_i, p_e, r_i, r_o, r_o)

        # Convert to MPa
        sigma_theta_i_MPa = sigma_theta_i / 1e6
        sigma_theta_o_MPa = sigma_theta_o / 1e6
        sigma_r_i_MPa = sigma_r_i / 1e6
        sigma_r_o_MPa = sigma_r_o / 1e6

        # ------------------ Results ------------------
        st.subheader("Calculated Lining Stress Results")
        st.write(f"Internal pressure pᵢ = {p_i/1e6:.2f} MPa")
        st.write(f"External pressure pₑ = {p_e/1e6:.2f} MPa")
        st.write(f"Effective pore pressure p_f = {p_f/1e6:.2f} MPa")


        st.markdown("**Hoop (circumferential) stress:**")
        st.write(f"σθ,i (inner surface) = {sigma_theta_i_MPa:.2f} MPa")
        st.write(f"σθ,o (outer surface) = {sigma_theta_o_MPa:.2f} MPa")

        st.markdown("**Radial stress:**")
        st.write(f"σr,i (inner surface) = {sigma_r_i_MPa:.2f} MPa")
        st.write(f"σr,o (outer surface) = {sigma_r_o_MPa:.2f} MPa")

        if sigma_theta_i_MPa <= ft_MPa and sigma_theta_o_MPa <= ft_MPa:
            st.success("Hoop stresses are within allowable tensile limits ✅")
        else:
            st.error("Hoop stresses exceed allowable tensile strength ❌")

        # ------------------ Stress distribution plot ------------------
        r_plot = np.linspace(r_i * 1.001, r_o, 200)
        sigma_theta_profile = hoop_stress(p_i, p_e, r_i, r_o, r_plot) / 1e6  # MPa
        sigma_r_profile = radial_stress(p_i, p_e, r_i, r_o, r_plot) / 1e6    # MPa

        fig_s, ax = plt.subplots(figsize=(8, 4.5), dpi=120)   # add dpi for crispness
        ax.plot(r_plot, sigma_theta_profile, lw=2.2, label="Hoop stress σθ(r)")
        ax.plot(r_plot, sigma_r_profile, lw=2.2, label="Radial stress σr(r)")
        ax.axhline(ft_MPa, color="g", ls="--", label=f"f_t = {ft_MPa:.1f} MPa")
        ax.axvline(r_i, color="k", ls=":", label=f"ri={r_i:.2f} m")
        ax.axvline(r_o, color="k", ls="--", label=f"ro={r_o:.2f} m")
        ax.fill_between(
            r_plot, sigma_theta_profile, ft_MPa,
            where=(sigma_theta_profile > ft_MPa),
            color="red", alpha=0.2, label="Cracking risk"
        )
        ax.set_xlabel("Radius r (m)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title("Radial & Hoop Stress Distribution in Lining")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")

# prevent auto-expansion in Streamlit
        st.pyplot(fig_s, use_container_width=False, clear_figure=True)

        
        # ------------------ Metrics summary ------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("σθ @ inner (MPa)", f"{sigma_theta_i_MPa:.1f}")
        c2.metric("σθ @ outer (MPa)", f"{sigma_theta_o_MPa:.1f}")
        c3.metric("σr @ inner (MPa)", f"{sigma_r_i_MPa:.1f}")
        c4.metric("σr @ outer (MPa)", f"{sigma_r_o_MPa:.1f}")


 # ------------------ Equations Reference ------------------
        with st.expander("Equations Used (Section 6)", expanded=False):
            st.latex(r"p_f = \eta \cdot (p_i - p_e) \quad \text{(effective pore pressure, Pa)}")
            st.latex(r"p_f = \frac{p_i - p_e}{1 + \dfrac{E_r (1 - \nu_c)}{E_c (1 - \nu_r)} \cdot \dfrac{r_i^2 (r_o^2 + r_i^2)}{r_o^2 (r_o^2 - r_i^2)}}")
            st.latex(r"\sigma_r(r) = \frac{p_f r_i^2 - p_e r_o^2}{r_o^2 - r_i^2} - \frac{(p_f - p_e) r_i^2 r_o^2}{(r_o^2 - r_i^2) r^2}")
            st.latex(r"\sigma_\theta(r) = \frac{p_f r_i^2 - p_e r_o^2}{r_o^2 - r_i^2} + \frac{(p_f - p_e) r_i^2 r_o^2}{(r_o^2 - r_i^2) r^2}")

except Exception as e:
    st.error(f"Error in stress calculation: {e}")

# ------------------ Section 8: Surge Tank ------------------ 
st.header("8) Surge Tank Design")

with st.expander("Input Parameters for Surge Tank", expanded=True):
    # Penstock setup
    n_pipes = st.number_input("Number of Penstocks connected to Surge Tank", value=1, step=1, min_value=1)
    D_p = st.number_input("Penstock Diameter Dₚ (m)", value=4.0, step=0.1, format="%.2f")
    
    # Compute total penstock cross-sectional area
    A_p_single = np.pi * (D_p**2) / 4
    A_p_total = n_pipes * A_p_single
    st.write(f"Penstock Area per pipe = {A_p_single:.2f} m²")
    st.write(f"Total Penstock Area Aₚ = {A_p_total:.2f} m²")

    # Discharge and head inputs
    Q0 = st.number_input("Rated Discharge Q₀ (m³/s)", value=50.0, step=1.0)
    H = st.number_input("Net Head H (m)", value=100.0, step=1.0)
    L = st.number_input("Headrace Tunnel Length L (m)", value=1500.0, step=50.0)

    # --- Surge Tank Area Options ---
    option = st.radio("How to determine Surge Tank Area Aₛ:",
                      ["Enter manually", 
                       "Estimate using Area Ratio (Aₛ/Aₚ)", 
                       "Estimate using Stability Formula",
                       "Check Rule-of-Thumb Stability"])

    if option == "Enter manually":
        A_s = st.number_input("Surge Tank Cross-sectional Area Aₛ (m²)", value=200.0, step=5.0)

    elif option == "Estimate using Area Ratio (Aₛ/Aₚ)":
        R = st.number_input("Choose Area Ratio R = Aₛ/Aₚ (default safe ~8)", value=8.0, step=1.0)
        A_s = R * A_p_total
        st.write(f"Estimated Surge Tank Area: Aₛ = {A_s:.2f} m² (using R = {R})")

    elif option == "Estimate using Stability Formula":
        # Assume water wave speed ~ 1000 m/s
        a = 1000.0  
        omega = np.pi * a / L  # angular frequency
        A_s = Q0 / (omega * H)
        st.write(f"Estimated Surge Tank Area (stability): Aₛ = {A_s:.2f} m²")

    elif option == "Check Rule-of-Thumb Stability":
        # Minimum R requirement
        R_min = L / H
        R_safe = 1.5 * R_min   # lower bound safety margin
        R_high = 2.0 * R_min   # upper bound safety margin
        
        st.write(f"Minimum required ratio R_min = L/H = {R_min:.2f}")
        st.write(f"Recommended practical range: {R_safe:.2f} ≤ R ≤ {R_high:.2f}")
        
        # Choose a value within the range
        R = st.number_input("Choose R (within safe range)", value=R_safe, step=0.5)
        A_s = R * A_p_total
        st.write(f"Estimated Surge Tank Area: Aₛ = {A_s:.2f} m² (using R = {R:.2f})")

    # Equivalent diameter (cylindrical tank assumption)
    D_s = np.sqrt(4 * A_s / np.pi)
    st.write(f"Equivalent Surge Tank Diameter ≈ {D_s:.2f} m")

# ---- Equations ----
with st.expander("Equations Used (Section 7)", expanded=False):
    st.markdown("**(1) Penstock Area per Pipe**")
    st.latex(r"A_{p,\;single} = \frac{\pi D_p^2}{4}")

    st.markdown("**(2) Total Penstock Area**")
    st.latex(r"A_p = n \cdot A_{p,\;single} = n \cdot \frac{\pi D_p^2}{4}")

    st.markdown("**(3) Surge Tank Area Ratio**")
    st.latex(r"R = \frac{A_s}{A_p}")

    st.markdown("**(4) Stability-based Surge Tank Area**")
    st.latex(r"A_s = \frac{Q_0}{\omega H}, \quad \omega \approx \frac{\pi a}{L}")

    st.markdown("**(5) Rule-of-Thumb Stability Check**")
    st.latex(r"\frac{A_s}{A_p} \geq \frac{L}{H}")

    st.markdown("**(6) Equivalent Tank Diameter (cylindrical assumption)**")
    st.latex(r"D_s = \sqrt{\frac{4 A_s}{\pi}}")

# ---- Results ----
st.subheader("Surge Tank Results")
st.write(f"Number of Penstocks: {n_pipes}")
st.write(f"Diameter per Penstock: {D_p:.2f} m")
st.write(f"Total Penstock Area Aₚ: {A_p_total:.2f} m²")
st.write(f"Surge Tank Area Aₛ: {A_s:.2f} m²")
st.write(f"Area Ratio (Aₛ / Aₚ): {A_s / A_p_total:.2f}")
st.write(f"Equivalent Surge Tank Diameter: {D_s:.2f} m")



st.header("9) Core Equations")

tabH, tabM, tabS = st.tabs(["Hydraulics", "Mechanics (Lining & Rock)", "Surge/Waterhammer"])

with tabH:
    st.markdown("### Hydraulics")
    st.markdown("**Continuity (flow balance):**")
    st.latex(r"Q = A \, v")
    st.markdown("**Bernoulli (with head losses):**")
    st.latex(r"\frac{P_1}{\rho g} + \frac{v_1^2}{2g} + z_1 \;=\; \frac{P_2}{\rho g} + \frac{v_2^2}{2g} + z_2 + h_f")
    st.markdown("**Turbine Power:**")
    st.latex(r"P = \rho g Q H_{\text{net}} \eta_t")
    st.markdown("**Darcy–Weisbach (with local losses):**")
    st.latex(r"h_f = \left(f \frac{L}{D} + \Sigma K \right) \frac{v^2}{2g}")
    st.markdown("**Reynolds Number:**")
    st.latex(r"\mathrm{Re} = \frac{vD}{\nu}, \quad f=f(\mathrm{Re}, \varepsilon/D)")
    st.caption("Links hydraulic losses to velocity, pipe geometry, roughness, and flow regime.")

with tabM:
    st.markdown("### Mechanics (Lining & Rock)")
    st.markdown("**Lamé hoop stress (thick-walled cylinder):**")
    st.latex(r"\sigma_\theta(r) = \frac{p_i(r^2 + r_i^2) - 2p_e r^2}{r^2 - r_i^2}")
    st.markdown("**External confinement requirement (inner-fibre check):**")
    st.latex(r"p_{e,\text{req}} \approx \frac{(p_i - f_t)(r_o^2 - r_i^2)}{2r_o^2}")
    st.markdown("**Snowy vertical rock cover criterion:**")
    st.latex(r"C_{RV} = \frac{h_s \gamma_w}{\gamma_R}")
    st.markdown("**Norwegian stability factor (valley-side):**")
    st.latex(r"F_{RV} = \frac{C_{RV}\,\gamma_R \cos \alpha}{h_s \gamma_w}")
    st.caption("Used for evaluating tunnel stability under headrace pressure.")

with tabS:
    st.markdown("### Surge / Waterhammer")
    st.markdown("**First-cut surge tank sizing (oscillator analogy):**")
    st.latex(r"A_s \approx k \, A_h")
    st.latex(r"\omega_n = \sqrt{\frac{g A_h}{L_h A_s}}, \quad T_n = \frac{2\pi}{\omega_n}")
    st.markdown("**Rule-of-thumb stability:**")
    st.latex(r"\frac{A_s}{A_p} \;\geq\; \frac{L}{H}")
    st.caption("⚠️ Teaching approximations only — detailed design needs transient surge analysis (e.g., Method of Characteristics).")


st.subheader("9) Reference Tables")

with st.expander("📘 Typical Darcy Friction Factors (f)", expanded=False):
    df_f = pd.DataFrame({
        "Material": ["PVC/HDPE", "New steel (welded)", "Concrete (smooth)", "Concrete (rough)", "Rock tunnel (lined)"],
        "f (typical)": [0.009, 0.012, 0.015, 0.022, 0.025],
        "Range": ["0.007–0.012", "0.010–0.015", "0.012–0.018", "0.018–0.025", "0.020–0.030"],
        "Source": ["AWWA (2012)", "ASCE (2017)", "ACI 351.3R", "USBR (1987)", "USACE EM (2008)"]
    })
    st.table(df_f)

with st.expander("📘 Local Loss Coefficients ΣK", expanded=False):
    df_k = pd.DataFrame({
        "Component": ["Entrance (bellmouth)", "90° bend (r/D ≈ 1)", "Gate valve (open)", "T-junction", "Exit"],
        "K (typical)": [0.15, 0.25, 0.20, 0.40, 1.00],
        "Range": ["0.1–0.2", "0.2–0.3", "0.1–0.3", "0.3–0.5", "0.8–1.0"],
        "Notes": ["Smooth inlet", "Moderate bend radius", "Depends on valve type", "Flow division losses", "Kinetic recovery lost"]
    })
    st.table(df_k)
    st.caption("Rule of thumb: ΣK ≈ 2–5 for a well-designed hydropower penstock system.")

with st.expander("📘 Rock & Concrete Properties (for lining checks)", expanded=False):
    df_mat = pd.DataFrame({
        "Property": ["Concrete tensile strength f_t", "Concrete modulus E_c", "Rock modulus E_r", "Unit weight of rock γ_r"],
        "Typical Value": ["2–4 MPa", "30–40 GPa", "20–60 GPa", "25–27 kN/m³"],
        "Reference": ["ACI 318", "ACI 363R", "ISRM (2007)", "Hoek & Brown"]
    })
    st.table(df_mat)


st.header("10) Bibliography")
st.markdown("### 📚 Bibliography (Teaching References)")
st.markdown("""
- USBR (1987). *Design of Small Dams*. 3rd ed. — Penstocks & hydraulics guidance  
- USACE EM 1110-2-1602. *Hydraulic Design of Reservoir Outlet Works*  
- ICOLD Bulletins: *Pressure tunnels, surge tanks*  
- ASCE Manuals of Practice; ACI 351.3R — Concrete & friction references  
- AWWA (2012). *Hydraulic Roughness Tables*  
- Chaudhry, M.H. (2014). *Applied Hydraulic Transients*, Springer  
- Gordon, J.L. (2001). *Hydraulics of Hydroelectric Power*  
- Hoek, E. & Brown, E. (1997). *Practical estimates of rock mass properties*  
""")
st.caption("This app is for classroom learning & scoping studies only — not detailed engineering design.")
