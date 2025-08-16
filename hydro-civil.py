# PHES Design Teaching App — with Moody helper (Swamee–Jain) for f(Re, ε/D)
# Reservoir head, penstock hydraulics, lining stress, losses, surge tanks
# Classroom-friendly; robust on Streamlit Cloud; no Styler usage

import io
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
    """
    Dynamic viscosity (Pa·s) of water vs temperature (°C).
    Simple curve fit good for 0–50°C classroom range.
    """
    # Viscosity in mPa·s via empirical (Korson-like) then convert to Pa·s
    # mu_mPa_s ≈ 2.414e-5 * 10^(247.8/(T_K-140))  [note: that formula returns Pa·s directly]
    T_K = T_C + 273.15
    mu = 2.414e-5 * 10**(247.8/(T_K - 140))  # Pa·s
    return mu

def water_nu_kinematic_m2s(T_C: float, rho=RHO) -> float:
    """Kinematic viscosity ν = μ/ρ in m²/s."""
    mu = water_mu_dynamic_PaS(T_C)
    return mu / rho

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
    Re: Reynolds number, rel_rough: ε/D (dimensionless).
    Also handles laminar and transitional ranges smoothly for teaching.
    """
    Re = float(Re)
    if np.isnan(Re) or Re <= 0:
        return float("nan")

    # Laminar: f = 64/Re
    if Re < 2000:
        return 64.0 / Re

    # Fully turbulent explicit (Swamee–Jain)
    f_turb = 0.25 / (math.log10(rel_rough/3.7 + 5.74/(Re**0.9)))**2

    # Transitional (2000–4000): linear blend laminar↔turbulent (didactic)
    if Re < 4000:
        f_lam = 64.0 / Re
        w = (Re - 2000.0) / 2000.0  # 0→1
        return (1 - w)*f_lam + w*f_turb

    return f_turb

def roughness_library():
    """
    Absolute roughness ε [m] — indicative teaching values.
    Sources: USBR/USACE/ASCE typical tables.
    """
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
    """
    External confinement to keep σθ at the inner/outer fibre ≤ f_t (didactic inner-fibre form).
    Approx classroom expression:
        p_ext,req ≈ ((p_i - f_t) * (r_o^2 - r_i^2)) / (2 r_o^2)
    """
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

# ------------------------------- App Shell -------------------------------
st.set_page_config(page_title="PHES Design Teaching App (with Moody)", layout="wide")
st.title("Pumped Hydro Energy Storage — Design Teaching App")
st.caption("Now with a Moody helper: compute friction factor from Re and relative roughness. "
           "Teaching tool — not a substitute for detailed design or transient analysis.")

# ------------------------------- Presets -------------------------------
with st.sidebar:
    st.header("Presets")
    preset = st.selectbox("Project", ["Custom", "Snowy 2.0 · Plateau", "Kidston"])
    if st.button("Apply preset"):
        if preset == "Snowy 2.0 · Plateau":
            st.session_state.update(dict(
                HWL_u=1100.0, LWL_u=1080.0, HWL_l=450.0, TWL_l=420.0,
                N_penstocks=6, D_pen=4.8, design_power=1000.0, max_power=2000.0,
                L_penstock=15000.0, eta_t=0.90,
                rough_choice="Concrete (smooth)", T_C=15.0, eps_custom=0.00030
            ))
        elif preset == "Kidston":
            st.session_state.update(dict(
                HWL_u=500.0, LWL_u=490.0, HWL_l=230.0, TWL_l=220.0,
                N_penstocks=2, D_pen=3.2, design_power=250.0, max_power=500.0,
                L_penstock=800.0, eta_t=0.90,
                rough_choice="New steel (welded)", T_C=25.0, eps_custom=0.000045
            ))

# ------------------------------- Section 1: Reservoirs -------------------------------
st.header("1) Reservoir Levels & Rating Head")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Upper reservoir**")
    HWL_u = st.number_input("Upper HWL (m)", 0.0, 3000.0, float(st.session_state.get("HWL_u", 1100.0)), 1.0)
    LWL_u = st.number_input("Upper LWL (m)", 0.0, 3000.0, float(st.session_state.get("LWL_u", 1080.0)), 1.0)
with c2:
    st.markdown("**Lower reservoir**")
    HWL_l = st.number_input("Lower HWL (m)", 0.0, 3000.0, float(st.session_state.get("HWL_l", 450.0)), 1.0)
    TWL_l = st.number_input("Lower TWL (m)", 0.0, 3000.0, float(st.session_state.get("TWL_l", 420.0)), 1.0)

gross_head = HWL_u - TWL_l             # maximum operating head (rating head often near this)
min_head = LWL_u - HWL_l               # minimum operating head (worst for power)
head_fluct_ratio = safe_div((LWL_u - TWL_l), (HWL_u - TWL_l))

# Visualization (simple)
fig_res, ax = plt.subplots(figsize=(8, 5))
ax.bar(["Upper"], [HWL_u - LWL_u], bottom=LWL_u, color="#3498DB", alpha=0.75, width=0.4)
ax.bar(["Lower"], [HWL_l - TWL_l], bottom=TWL_l, color="#2ECC71", alpha=0.75, width=0.4)
ax.annotate("", xy=(0, HWL_u), xytext=(0, TWL_l), arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=2))
ax.text(-0.1, (HWL_u + TWL_l)/2, f"Gross ≈ {gross_head:.1f} m", color="#E74C3C", va="center")
ax.annotate("", xy=(0.2, LWL_u), xytext=(0.2, HWL_l), arrowprops=dict(arrowstyle="<->", color="#27AE60", lw=2))
ax.text(0.1, (LWL_u + HWL_l)/2, f"Min ≈ {min_head:.1f} m", color="#27AE60", va="center")
ax.set_ylabel("Elevation (m)")
ax.set_title("Reservoir Operating Range")
ax.grid(True, linestyle="--", alpha=0.35)
st.pyplot(fig_res)

m1, m2, m3 = st.columns(3)
m1.metric("Gross head (m)", f"{gross_head:.1f}")
m2.metric("Min head (m)", f"{min_head:.1f}")
m3.metric("Head fluctuation ratio (LWL→TWL)", f"{head_fluct_ratio:.3f}")

# ------------------------------- Section 2: Penstock & Moody -------------------------
st.header("2) Penstock Geometry & Efficiencies (with Moody)")
c1, c2 = st.columns(2)
with c1:
    N_pen = st.number_input("Number of penstocks", 1, 16, int(st.session_state.get("N_penstocks", 2)))
    D_pen = st.number_input("Penstock diameter D (m)", 0.5, 12.0, float(st.session_state.get("D_pen", 3.5)), 0.1)
    L_pen = st.number_input("Penstock length L (m)", 10.0, 50000.0, float(st.session_state.get("L_penstock", 500.0)), 10.0)
with c2:
    eta_t = st.number_input("Turbine efficiency ηₜ (-)", 0.7, 1.0, float(st.session_state.get("eta_t", 0.90)), 0.01)
    P_design = st.number_input("Design power (MW)", 10.0, 5000.0, float(st.session_state.get("design_power", 500.0)), 10.0)
    P_max = st.number_input("Maximum power (MW)", 10.0, 6000.0, float(st.session_state.get("max_power", 600.0)), 10.0)

st.subheader("Friction factor mode")
mode_f = st.radio("Choose how to set Darcy friction factor f:",
                  ["Manual (slider)", "Compute from Moody (Swamee–Jain)"], index=1)

if mode_f == "Manual (slider)":
    # classic way
    f = st.slider("Friction factor f (Darcy)", 0.005, 0.03, 0.015, 0.001)
    rough_choice = "—"
    eps = None
    T_C = None
else:
    # Compute from Re and ε/D
    colA, colB, colC = st.columns(3)
    with colA:
        T_C = st.number_input("Water temperature (°C)", 0.0, 60.0, float(st.session_state.get("T_C", 20.0)), 0.5)
    with colB:
        rl = roughness_library()
        rough_choice = st.selectbox("Material / absolute roughness ε (m)", list(rl.keys()),
                                    index=list(rl.keys()).index(st.session_state.get("rough_choice","Concrete (smooth)")))
    with colC:
        if rough_choice == "Custom...":
            eps = st.number_input("Custom ε (m)", 0.0, 0.01, float(st.session_state.get("eps_custom", 0.00030)), 0.00001, format="%.5f")
        else:
            eps = rl[rough_choice]

# ------------------------------- Section 3: Losses & iteration -----------------------
st.header("3) Discharges, Velocities, Head Losses")

def compute_block(P_MW, h_span, hf_guess=30.0):
    A = area_circle(D_pen)
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

    # ❌ ERROR was here, using undefined "Ksum_global"
    hf = headloss_darcy(f_used, L_pen, D_pen, v, Ksum_global)

    return dict(
        f=f_used, Re=Re, v=v, Q_total=Q_total, Q_per=Q_per,
        h_net=h_net, hf=hf
    )


# local loss builder
st.subheader("Local loss components (ΣK)")
components = {
    "Entrance (bellmouth)": 0.15,
    "Entrance (square)": 0.50,
    "90° bend": 0.25,
    "45° bend": 0.15,
    "Gate valve (open)": 0.20,
    "Butterfly valve (open)": 0.30,
    "T-junction": 0.40,
    "Exit": 1.00
}
K_sum_global = 0.0
cols = st.columns(4)
i = 0
for comp, kval in components.items():
    default_on = comp in ["Entrance (bellmouth)", "90° bend", "Exit"]
    with cols[i % 4]:
        if st.checkbox(comp, value=default_on):
            K_sum_global += kval
    i += 1
st.metric("ΣK (selected)", f"{K_sum_global:.2f}")

# compute for design (gross head) and max (min head)
out_design = compute_block(P_design, gross_head, hf_guess=25.0)
out_max = compute_block(P_max, min_head, hf_guess=40.0)

# summary table (cloud-safe formatting)
results_basic = pd.DataFrame({
    "Case": ["Design", "Maximum"],
    "Net head h_net (m)": [out_design["h_net"], out_max["h_net"]],
    "Total Q (m³/s)": [out_design["Q_total"], out_max["Q_total"]],
    "Per-penstock Q (m³/s)": [out_design["Q_per"], out_max["Q_per"]],
    "Velocity v (m/s)": [out_design["v"], out_max["v"]],
    "Reynolds Re (-)": [out_design["Re"], out_max["Re"]],
    "Darcy f (-)": [out_design["f"], out_max["f"]],
    "Head loss h_f (m)": [out_design["hf"], out_max["hf"]],
})
st.dataframe(
    results_basic,
    use_container_width=True,
    column_config={
        "Net head h_net (m)": st.column_config.NumberColumn(format="%.2f"),
        "Total Q (m³/s)": st.column_config.NumberColumn(format="%.2f"),
        "Per-penstock Q (m³/s)": st.column_config.NumberColumn(format="%.2f"),
        "Velocity v (m/s)": st.column_config.NumberColumn(format="%.2f"),
        "Reynolds Re (-)": st.column_config.NumberColumn(format="%.0f"),
        "Darcy f (-)": st.column_config.NumberColumn(format="%.004f"),
        "Head loss h_f (m)": st.column_config.NumberColumn(format="%.2f"),
    }
)

# Velocity checks
st.subheader("Velocity checks (USBR guidance)")
v_design = out_design["v"]; v_max = out_max["v"]
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

# ------------------------------- Mini Moody plot ------------------------
if mode_f != "Manual (slider)":
    st.subheader("Mini Moody diagram (your operating point)")
    # Prepare f vs Re family for a few ε/D values
    Re_vals = np.logspace(3, 8, 300)
    epsD_list = [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]  # smooth → rough
    fig_m, axm = plt.subplots(figsize=(7.5, 5))
    for rr in epsD_list:
        f_line = [f_moody_swamee_jain(Re, rr) for Re in Re_vals]
        axm.plot(Re_vals, f_line, lw=1.5, label=f"ε/D={rr:g}")
    # add your two points
    if not np.isnan(out_design["Re"]):
        axm.scatter([out_design["Re"]], [out_design["f"]], c="tab:green", s=50, label="Design point")
    if not np.isnan(out_max["Re"]):
        axm.scatter([out_max["Re"]], [out_max["f"]], c="tab:red", s=50, label="Max point")
    axm.set_xscale("log"); axm.set_yscale("log")
    axm.set_xlabel("Reynolds number Re"); axm.set_ylabel("Darcy friction factor f")
    axm.set_title("Moody chart (Swamee–Jain approximation)")
    axm.grid(True, which="both", ls="--", alpha=0.35)
    axm.legend(loc="best", fontsize=9)
    st.pyplot(fig_m)

# ------------------------------- Section 4: System Curve ------------------
st.header("4) System Power Curve (didactic)")
Q_max_total = out_max["Q_total"]
if np.isnan(Q_max_total) or Q_max_total <= 0:
    Q_max_total = max(1.0, (P_max * 1e6) / (RHO * G * max(min_head, 1.0) * max(eta_t, 0.6)))

Q_grid = np.linspace(0, 1.2 * Q_max_total, 140)
h_net_design = out_design["h_net"]; h_net_min = out_max["h_net"]
if any(np.isnan([h_net_design, h_net_min])):
    h_net_design, h_net_min = max(gross_head - 10, 1.0), max(min_head - 10, 0.5)

# Quadratic head drop with Q is illustrative only
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

# ------------------------------- Section 5: Rock Cover & Lining -----------
st.header("5) Pressure Tunnel: Rock Cover & Lining Stress")
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

c1, c2 = st.columns(2)
c1.metric("Snowy vertical cover C_RV (m)", f"{CRV:.1f}")
c2.metric("Norwegian factor F_RV (-)", f"{FRV:.2f}")
st.markdown("**Target**: Typically F_RV ≥ 1.2–1.5 (site-dependent).")

st.subheader("Lining Hoop Stress (Lame solution)")
c1, c2, c3 = st.columns(3)
with c1:
    pi_MPa = st.number_input("Internal water pressure p_i (MPa)", 0.1, 20.0, 2.0, 0.1)
with c2:
    pext = st.number_input("External confinement p_ext (MPa)", 0.0, 20.0, 0.0, 0.1)
with c3:
    ft_MPa = st.number_input("Concrete tensile strength f_t (MPa)", 1.0, 10.0, 3.0, 0.1)

sigma_outer = hoop_stress(pi_MPa, pext, ri, re)   # evaluated at outer face (display)
pext_req = required_pext_for_ft(pi_MPa, ri, re, ft_MPa)

# Stress profile
r_plot = np.linspace(ri * 1.001, re, 200)
sigma_profile = hoop_stress(pi_MPa, pext, ri, r_plot)

fig_s, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(r_plot, sigma_profile, lw=2.2, label="σθ(r)")
ax.axhline(ft_MPa, color="g", ls="--", label=f"f_t = {ft_MPa:.1f} MPa")
ax.axvline(ri, color="k", ls=":", label=f"ri={ri:.2f} m")
ax.axvline(re, color="k", ls="--", label=f"re={re:.2f} m")
ax.fill_between(r_plot, sigma_profile, ft_MPa, where=(sigma_profile > ft_MPa), color="red", alpha=0.2,
                label="Cracking risk")
ax.set_xlabel("Radius r (m)"); ax.set_ylabel("Hoop stress σθ (MPa)")
ax.set_title("Lining hoop stress distribution")
ax.grid(True, linestyle="--", alpha=0.35); ax.legend(loc="best")
st.pyplot(fig_s)

c1, c2, c3 = st.columns(3)
c1.metric("σθ @ outer face (MPa)", f"{sigma_outer:.1f}")
c2.metric("Required p_ext (MPa)", f"{pext_req:.2f}")
c3.metric("Status", "⚠️ Cracking likely" if sigma_outer > ft_MPa else "✅ OK",
          help=("Stress exceeds tensile strength; increase thickness or confinement."
                if sigma_outer > ft_MPa else "Within tensile capacity at outer face."))

# ------------------------------- Section 6: Optional hf = k·Q^n Fit ------
st.header("6) (Optional) Loss Curve Fit  h_f = k·Qⁿ  from Anchors")
c1, c2, c3 = st.columns(3)
with c1:
    P1 = st.number_input("Anchor P₁ (MW)", 10.0, 5000.0, 1000.0, 10.0)
with c2:
    hf1 = st.number_input("h_f at P₁ (m)", 0.0, 500.0, 28.0, 0.1)
with c3:
    P2 = st.number_input("Anchor P₂ (MW)", 10.0, 5000.0, 2000.0, 10.0)
hf2 = st.number_input("h_f at P₂ (m)", 0.0, 500.0, 70.0, 0.1)

def fit_hf_k_n_from_anchors(h_gross, eta_t, anchors):
    (P1_, hf1_), (P2_, hf2_) = anchors
    Q1 = Q_from_power(P1_, h_gross - hf1_, eta_t)
    Q2 = Q_from_power(P2_, h_gross - hf2_, eta_t)
    if any(np.isnan([Q1, Q2])) or Q1 <= 0 or Q2 <= 0 or hf1_ <= 0 or hf2_ <= 0:
        return float("nan"), float("nan")
    n = math.log(hf2_ / hf1_) / math.log(Q2 / Q1)
    k = hf1_ / (Q1**n)
    return k, n

k_fit, n_fit = fit_hf_k_n_from_anchors(gross_head, eta_t, [(P1, hf1), (P2, hf2)])
if not (np.isnan(k_fit) or np.isnan(n_fit)):
    st.info(f"Fitted curve:  h_f ≈ {k_fit:.6g} · Q^{n_fit:.3f}   (Q in m³/s, h_f in m)")
    Q_show = np.linspace(0.1, max(1.2 * out_max["Q_total"], 10.0), 200)
    hf_show = k_fit * Q_show**n_fit
    fig_fit = make_subplots(specs=[[{"secondary_y": False}]])
    fig_fit.add_trace(go.Scatter(x=Q_show, y=hf_show, name="h_f(Q) fit", line=dict(width=3)))
    fig_fit.add_trace(go.Scatter(x=[out_design["Q_total"]], y=[out_design["hf"]],
                                 mode="markers", name="Design point", marker=dict(size=10)))
    fig_fit.add_trace(go.Scatter(x=[out_max["Q_total"]], y=[out_max["hf"]],
                                 mode="markers", name="Max point", marker=dict(size=10)))
    fig_fit.update_layout(title="Fitted head-loss curve (didactic)",
                          xaxis_title="Total Q (m³/s)", yaxis_title="h_f (m)", height=420)
    st.plotly_chart(fig_fit, use_container_width=True)
else:
    st.caption("Provide sensible anchors to view a fitted h_f = k·Qⁿ curve.")

# ------------------------------- Section 7: Surge Tank -------------------
st.header("7) Surge Tank — First Cut")
Ah = area_circle(D_pen)  # per conduit; for multiple branches, use local area at the tank location
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
    st.markdown("#### Continuity")
    st.latex(r"Q = A \, v")
    st.markdown("#### Bernoulli (with losses)")
    st.latex(r"\frac{P_1}{\rho g} + \frac{v_1^2}{2g} + z_1 = \frac{P_2}{\rho g} + \frac{v_2^2}{2g} + z_2 + h_f")
    st.markdown("#### Turbine Power")
    st.latex(r"P = \rho g Q H_{\text{net}} \eta_t")
    st.markdown("#### Darcy–Weisbach Head Loss (with local losses)")
    st.latex(r"h_f = \left(f \frac{L}{D} + \sum K \right) \frac{v^2}{2g}")

with tabM:
    st.markdown("#### Lame (thick-walled cylinder) — hoop stress")
    st.latex(r"\sigma_\theta(r) = \frac{p_i (r^2 + r_i^2) - 2 p_e r^2}{r^2 - r_i^2}")
    st.markdown("#### Required external confinement (didactic inner-fibre check)")
    st.latex(r"p_{e,\text{req}} \approx \frac{(p_i - f_t) (r_o^2 - r_i^2)}{2 r_o^2}")
    st.markdown("#### Snowy vertical cover")
    st.latex(r"C_{RV} = \frac{h_s \, \gamma_w}{\gamma_R}")
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
    "surge": {"Ah": area_circle(D_pen), **surge}
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
    "v_design_ms": out_design["v"], "hf_design_m": out_design["hf"],
    "hnet_max_m": out_max["h_net"], "Q_total_max_m3s": out_max["Q_total"],
    "v_max_ms": out_max["v"], "hf_max_m": out_max["hf"],
    "T_C": (T_C if mode_f != "Manual (slider)" else None),
    "eps_m": (eps if mode_f != "Manual (slider)" else None),
    "rel_rough": (out_design["rel_rough"] if mode_f != "Manual (slider)" else None),
}
csv_bytes = pd.DataFrame([flat]).to_csv(index=False).encode("utf-8")
st.download_button("Download CSV (parameters)", data=csv_bytes, file_name="phes_parameters.csv")

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
