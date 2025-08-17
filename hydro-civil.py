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

# --- Visualisation ---
fig_res, ax = plt.subplots(figsize=(8, 5))
# storage bars
ax.bar(["Upper"], [HWL_u - LWL_u], bottom=LWL_u, color="#3498DB", alpha=0.75, width=0.4)
ax.bar(["Lower"], [HWL_l - TWL_l], bottom=TWL_l, color="#2ECC71", alpha=0.75, width=0.4)
# NWL line (spans both bars)
ax.hlines(NWL_u, -0.4, 1.4, colors="#34495E", linestyles="--", linewidth=2, label="NWL (upper)")
# gross head arrow (NWL to TWL)
ax.annotate("", xy=(1.0, NWL_u), xytext=(1.0, TWL_l),
            arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=2))
ax.text(1.05, (NWL_u + TWL_l)/2, f"Hg ≈ {gross_head:.1f} m", color="#E74C3C", va="center")
# min head arrow (LWL to HWL_l)
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

# Head fluctuation criterion (lower limit)
st.markdown("**Head fluctuation rate**")
st.latex(r"\text{HFR} = \frac{LWL - TWL}{HWL - TWL}")
crit_col1, crit_col2 = st.columns([2, 1])
with crit_col1:
    turbine_choice = st.selectbox(
        "Criterion (lower limit, choose turbine type or none):",
        ["None (no check)", "Francis (≥ 0.70)", "Kaplan (≥ 0.55)"],
        index=1
    )
with crit_col2:
    custom_limit = st.number_input("Custom lower limit (optional)", 0.0, 1.0, 0.70, 0.01)

if   turbine_choice.startswith("Francis"): limit = 0.70
elif turbine_choice.startswith("Kaplan"):  limit = 0.55
elif turbine_choice.startswith("None"):    limit = None
else:                                       limit = None
if limit is not None and custom_limit is not None:
    limit = float(custom_limit)

if limit is not None and not np.isnan(head_fluct_ratio):
    st.markdown(f"**HFR:** {head_fluct_ratio:.3f}  •  **Lower limit:** {limit:.2f}")
    if head_fluct_ratio >= limit:
        st.success("Meets the recommended **minimum** (HFR ≥ limit).")
    else:
        st.error("Below the recommended **minimum** head fluctuation. "
                 "Consider **raising LWL** or **increasing HWL − TWL**.")

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

st.markdown(r"""
### What is ε/D?

Relative roughness $ \varepsilon / D $ compares the wall roughness height $ \varepsilon $ to the pipe diameter $ D $.  
It is dimensionless and is used with the Reynolds number $ Re $ to determine the Darcy friction factor $ f $ on a Moody diagram.

$$
\frac{\varepsilon}{D}, \qquad 
f \;\text{depends on}\; (Re, \, \varepsilon / D) \;\text{in turbulent flow.}
$$

Typical $ \varepsilon $ values (order of magnitude):

- PVC/HDPE: $1.5 \times 10^{-6}\,\text{m}$  
- New steel (welded): $4.5 \times 10^{-5}\,\text{m}$  
- Concrete (smooth): $3.0 \times 10^{-4}\,\text{m}$  
- Rock tunnel (good lining): $1.0 \times 10^{-3}\,\text{m}$  

*Teaching sources: USBR (1987), AWWA, ASCE/USACE typical roughness tables.*
""")



# -------------------- Section 3: Discharges & Velocities (no ΣK yet) -----------------
st.header("3) Discharges & Velocities")

def compute_block(P_MW, h_span, Ksum, hf_guess=30.0):
    """Two-pass iteration to refine f and h_f for given Ksum."""
    A = area_circle(D_pen)
    # pass 1
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

    # pass 2
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

    return dict(
        f=f_used2, Re=Re2, v=v2, Q_total=Q_total2, Q_per=Q_per2,
        h_net=h_net2, hf=hf2,
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
    st.latex(r"A = \frac{\pi D^{2}}{4}")
    st.latex(r"Q_p = \frac{Q_\text{total}}{N_\text{pen}}")
    st.latex(r"v = \frac{Q_p}{A}")
    st.latex(r"\mathrm{Re} = \frac{v D}{\nu}")
    st.latex(r"\boxed{\;\mathrm{Re}=\dfrac{4\,Q_\text{total}}{\pi\,N_\text{pen}\,D\,\nu}\;}")

# ---------- Mini Moody chart (AFTER flows exist) ----------
if (mode_f != "Manual (slider)"):
    st.subheader("Mini Moody diagram (your operating points)")
    Re_vals = np.logspace(3, 8, 300)
    epsD_list = [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]

    fig_m, axm = plt.subplots(figsize=(7.5, 5))
    for rr in epsD_list:
        f_line = [f_moody_swamee_jain(Re, rr) for Re in Re_vals]
        axm.plot(Re_vals, f_line, lw=1.5, label=f"ε/D={rr:g}")

    for label, out_pt in (("Design point", out_design_flow), ("Max point", out_max_flow)):
        Re_pt = out_pt.get("Re", np.nan)
        f_pt  = out_pt.get("f",  np.nan)
        if (not np.isnan(Re_pt)) and (not np.isnan(f_pt)):
            axm.scatter([Re_pt], [f_pt], s=60, zorder=5, label=label)

    axm.set_xscale("log"); axm.set_yscale("log")
    axm.set_xlabel("Reynolds number Re")
    axm.set_ylabel("Darcy friction factor f")
    axm.set_title("Moody chart (Swamee–Jain approximation)")
    axm.grid(True, which="both", ls="--", alpha=0.35)
    axm.legend(loc="best", fontsize=9)
    st.pyplot(fig_m)

# Table for Q & v only
results_flow = pd.DataFrame({
    "Case": ["Design", "Maximum"],
    "Net head h_net (m)": [out_design_flow["h_net"], out_max_flow["h_net"]],
    "Total Q (m³/s)": [out_design_flow["Q_total"], out_max_flow["Q_total"]],
    "Per-penstock Q (m³/s)": [out_design_flow["Q_per"], out_max_flow["Q_per"]],
    "Velocity v (m/s)": [out_design_flow["v"], out_max_flow["v"]],
    "Reynolds Re (-)": [out_design_flow["Re"], out_max_flow["Re"]],
})
st.dataframe(
    results_flow, use_container_width=True,
    column_config={
        "Net head h_net (m)": st.column_config.NumberColumn(format="%.2f"),
        "Total Q (m³/s)": st.column_config.NumberColumn(format="%.2f"),
        "Per-penstock Q (m³/s)": st.column_config.NumberColumn(format="%.2f"),
        "Velocity v (m/s)": st.column_config.NumberColumn(format="%.2f"),
        "Reynolds Re (-)": st.column_config.NumberColumn(format="%.0f"),
    }
)

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
with st.expander("Show equations used (Section 3)"):
    st.latex(r"A = \frac{\pi D^{2}}{4} \quad (\text{m}^2)")
    st.latex(r"Q_p = \frac{Q_\text{total}}{N_\text{pen}} \quad (\text{m}^3/\text{s})")
    st.latex(r"v = \frac{Q_p}{A} \quad (\text{m/s})")
    st.markdown(
        """
        **Velocity guidance (USBR):**
        - Low-pressure steel penstocks: *3 – 5 m/s*  
        - Medium-pressure steel penstocks: *5 – 7 m/s*  
        - High-pressure steel penstocks: *7 – 10 m/s*  
        *Reference: USBR, Design of Small Dams, 3rd Ed. (1987), Ch. 10 – Penstocks.*
        """
    )

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
