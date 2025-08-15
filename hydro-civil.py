# PHES Design Teaching App (with Equations & Reference Tables)
# ----------------------------------------------------------------
# Reservoir head, penstock hydraulics, lining stress, losses, surge tanks
# Cloud-friendly, robust formatting, didactic annotations, equations & references

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
RHO = 1000.0    # kg/m³

# ------------------------------- Helpers -------------------------------
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

def hoop_stress(pi, pe, ri, r):
    """
    Lame solution (thick-walled cylinder, elastic):
    σθ(r) = [pi*(r^2 + ri^2) - 2*pe*r^2] / (r^2 - ri^2)
    Accepts scalars or numpy arrays for r.
    """
    r_arr = np.array([r]) if np.isscalar(r) else np.array(r)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (pi * (r_arr**2 + ri**2) - 2 * pe * r_arr**2) / (r_arr**2 - ri**2)
    s[r_arr <= ri] = np.nan  # not physical inside the inner radius
    return s.item() if np.isscalar(r) else s

def required_pext_for_ft(pi_MPa, ri, re, ft_MPa):
    """
    External confinement to keep σθ at the inner fibre ≤ f_t (simple conservative rearrangement).
    A common classroom approximation:
        p_ext,req ≈ (pi - f_t) * (re^2 - ri^2) / (2 re^2)
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

# Optional: empirical hf = k·Q^n curve fitting from two anchor points
def fit_hf_k_n_from_anchors(h_gross, eta_t, anchors):
    """
    anchors: [(P1, hf1), (P2, hf2)] in MW and meters.
    Returns k, n so that hf ≈ k·Q^n (Q is total discharge).
    """
    (P1, hf1), (P2, hf2) = anchors
    Q1 = Q_from_power(P1, h_gross - hf1, eta_t)
    Q2 = Q_from_power(P2, h_gross - hf2, eta_t)
    if any(np.isnan([Q1, Q2])) or Q1 <= 0 or Q2 <= 0 or hf1 <= 0 or hf2 <= 0:
        return float("nan"), float("nan")
    n = math.log(hf2 / hf1) / math.log(Q2 / Q1)
    k = hf1 / (Q1**n)
    return k, n

# ------------------------------- App Shell -------------------------------
st.set_page_config(page_title="PHES Design Teaching App", layout="wide")
st.title("Pumped Hydro Energy Storage — Design Teaching App")
st.caption("Interactive classroom tool: reservoir head, penstock hydraulics, lining stress, losses, and surge tanks. "
           "Values are illustrative; final design requires detailed vendor data + transient analysis.")

# ------------------------------- Presets -------------------------------
with st.sidebar:
    st.header("Presets")
    preset = st.selectbox("Project", ["Custom", "Snowy 2.0 · Plateau", "Kidston"])
    if st.button("Apply preset"):
        if preset == "Snowy 2.0 · Plateau":
            st.session_state.update(dict(
                HWL_u=1100.0, LWL_u=1080.0, HWL_l=450.0, TWL_l=420.0,
                N_penstocks=6, D_pen=4.8, design_power=1000.0, max_power=2000.0,
                f_material="Concrete (smooth)", f=0.015, L_penstock=15000.0,
                eta_t=0.90
            ))
        elif preset == "Kidston":
            st.session_state.update(dict(
                HWL_u=500.0, LWL_u=490.0, HWL_l=230.0, TWL_l=220.0,
                N_penstocks=2, D_pen=3.2, design_power=250.0, max_power=500.0,
                f_material="New steel (welded)", f=0.012, L_penstock=800.0,
                eta_t=0.90
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

# ------------------------------- Section 2: Penstock Inputs -------------------------------
st.header("2) Penstock Geometry & Efficiencies")
c1, c2 = st.columns(2)
with c1:
    N_pen = st.number_input("Number of penstocks", 1, 16, int(st.session_state.get("N_penstocks", 2)))
    D_pen = st.number_input("Penstock diameter D (m)", 0.5, 12.0, float(st.session_state.get("D_pen", 3.5)), 0.1)
    L_pen = st.number_input("Penstock length L (m)", 10.0, 50000.0, float(st.session_state.get("L_penstock", 500.0)), 10.0)
with c2:
    eta_t = st.number_input("Turbine efficiency ηₜ (-)", 0.7, 1.0, float(st.session_state.get("eta_t", 0.90)), 0.01)
    P_design = st.number_input("Design power (MW)", 10.0, 5000.0, float(st.session_state.get("design_power", 500.0)), 10.0)
    P_max = st.number_input("Maximum power (MW)", 10.0, 6000.0, float(st.session_state.get("max_power", 600.0)), 10.0)

# friction & local losses
st.subheader("Head Loss Parameters (Darcy–Weisbach + local)")
c1, c2 = st.columns(2)
with c1:
    f_options = {
        "New steel (welded)": 0.012,
        "New steel (riveted)": 0.017,
        "Concrete (smooth)": 0.015,
        "Concrete (rough)": 0.022,
        "PVC/Plastic": 0.009
    }
    f_material = st.selectbox("Penstock material", list(f_options.keys()),
                              index=2 if "Concrete" in st.session_state.get("f_material","Concrete (smooth)") else 0)
    f = st.slider("Friction factor f (Darcy)", 0.005, 0.03, float(st.session_state.get("f", f_options[f_material])), 0.001)
    st.caption(f"Tip: For {f_material} a typical f ≈ {f_options[f_material]:.3f}.")
with c2:
    st.markdown("**Local loss components (ΣK)**")
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
    K_sum = 0.0
    for comp, kval in components.items():
        default_on = comp in ["Entrance (bellmouth)", "90° bend", "Exit"]
        if st.checkbox(comp, value=default_on):
            K_sum += kval
    st.metric("ΣK (selected)", f"{K_sum:.2f}")

auto_hf = st.checkbox("Auto-compute head losses from velocities", True)

# ------------------------------- Section 3: Computations -------------------------------
st.header("3) Discharges, Velocities, Head Losses")

# first pass guess for hf (for iteration)
hf_design_guess = 25.0
hf_max_guess = 40.0

# iterative two-pass to converge hf with v
def compute_block(P_MW, hf_guess):
    h_span = gross_head if P_MW == P_design else min_head
    h_net = h_span - hf_guess
    Q_total = Q_from_power(P_MW, h_net, eta_t)
    Q_per = safe_div(Q_total, N_pen)
    A = area_circle(D_pen)
    v = safe_div(Q_per, A)
    hf = headloss_darcy(f, L_pen, D_pen, v, K_sum)
    # updated net head using hf
    h_net2 = h_span - hf
    Q_total2 = Q_from_power(P_MW, h_net2, eta_t)
    Q_per2 = safe_div(Q_total2, N_pen)
    v2 = safe_div(Q_per2, A)
    hf2 = headloss_darcy(f, L_pen, D_pen, v2, K_sum)
    return dict(
        h_net=h_net2, Q_total=Q_total2, Q_per=Q_per2, v=v2, hf=hf2
    )

if auto_hf:
    out_design = compute_block(P_design, hf_design_guess)
    out_max = compute_block(P_max, hf_max_guess)
else:
    # manual hf entries
    hf_design = st.number_input("Design head loss h_f,design (m)", 0.0, 500.0, 25.0, 0.1)
    hf_max = st.number_input("Max head loss h_f,max (m)", 0.0, 500.0, 40.0, 0.1)
    def compute_fixed(P_MW, hf_fixed, head_span):
        h_net = head_span - hf_fixed
        Q_total = Q_from_power(P_MW, h_net, eta_t)
        Q_per = safe_div(Q_total, N_pen)
        v = safe_div(Q_per, area_circle(D_pen))
        return dict(h_net=h_net, Q_total=Q_total, Q_per=Q_per, v=v, hf=hf_fixed)
    out_design = compute_fixed(P_design, hf_design, gross_head)
    out_max = compute_fixed(P_max, hf_max, min_head)

# summary table (Streamlit column_config instead of Styler to avoid cloud errors)
results_basic = pd.DataFrame({
    "Case": ["Design", "Maximum"],
    "Net head h_net (m)": [out_design["h_net"], out_max["h_net"]],
    "Total Q (m³/s)": [out_design["Q_total"], out_max["Q_total"]],
    "Per-penstock Q (m³/s)": [out_design["Q_per"], out_max["Q_per"]],
    "Velocity v (m/s)": [out_design["v"], out_max["v"]],
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
        "Head loss h_f (m)": st.column_config.NumberColumn(format="%.2f"),
    }
)

# velocity guidance
st.subheader("Velocity checks (USBR guidance)")
v_design = out_design["v"]
v_max = out_max["v"]
c1, c2 = st.columns(2)
with c1:
    st.metric("v_design (m/s)", f"{v_design:.2f}")
    st.metric("v_max (m/s)", f"{v_max:.2f}")
with c2:
    st.markdown("- **Recommended range:** 4–6 m/s (concrete penstocks)")
    st.markdown("- **Absolute max:** ~7 m/s short duration")
if v_max > 7.0:
    st.error("⚠️ Dangerous velocity (exceeds ~7 m/s). Revisit D or layout.")
elif v_max > 6.0:
    st.warning("⚠️ Above recommended 6 m/s. Acceptable only for short periods.")
elif v_max >= 4.0:
    st.success("✓ Within recommended 4–6 m/s range.")
else:
    st.info("ℹ️ Low velocity (<4 m/s): safe but potentially uneconomic (oversized).")

# ------------------------------- Section 4: System Curve -------------------------------
st.header("4) System Power Curve")
# Smooth operating curve (illustrative parabolic hf change)
Q_max_total = out_max["Q_total"]
if np.isnan(Q_max_total) or Q_max_total <= 0:
    Q_max_total = max(1.0, (P_max * 1e6) / (RHO * G * max(min_head, 1.0) * max(eta_t, 0.6)))

Q_grid = np.linspace(0, 1.2 * Q_max_total, 140)
# Let head drop quadratically with flow (didactic)
h_net_design = out_design["h_net"]
h_net_min = out_max["h_net"]
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

# ------------------------------- Section 5: Pressure Tunnel & Lining -------------------------------
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

# Lining stress
st.subheader("Lining Hoop Stress (Lame solution)")
c1, c2, c3 = st.columns(3)
with c1:
    pi_MPa = st.number_input("Internal water pressure p_i (MPa)", 0.1, 20.0, 2.0, 0.1)
with c2:
    pext = st.number_input("External confinement p_ext (MPa)", 0.0, 20.0, 0.0, 0.1)
with c3:
    ft_MPa = st.number_input("Concrete tensile strength f_t (MPa)", 1.0, 10.0, 3.0, 0.1)

sigma_outer = hoop_stress(pi_MPa, pext, ri, re)   # stress evaluated at outer face for display
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
ax.set_xlabel("Radius r (m)")
ax.set_ylabel("Hoop stress σθ (MPa)")
ax.set_title("Lining hoop stress distribution")
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="best")
st.pyplot(fig_s)

c1, c2, c3 = st.columns(3)
c1.metric("σθ @ outer face (MPa)", f"{sigma_outer:.1f}")
c2.metric("Required p_ext (MPa)", f"{pext_req:.2f}")
c3.metric("Status", "⚠️ Cracking likely" if sigma_outer > ft_MPa else "✅ OK",
          help=("Stress exceeds tensile strength; increase thickness or confinement."
                if sigma_outer > ft_MPa else "Within tensile capacity at outer face."))

# ------------------------------- Section 6: Optional hf = k·Q^n Fit -------------------------------
st.header("6) (Optional) Loss Curve Fit  h_f = k·Qⁿ  from Anchors")
c1, c2, c3 = st.columns(3)
with c1:
    P1 = st.number_input("Anchor P₁ (MW)", 10.0, 5000.0, 1000.0, 10.0)
with c2:
    hf1 = st.number_input("h_f at P₁ (m)", 0.0, 500.0, 28.0, 0.1)
with c3:
    P2 = st.number_input("Anchor P₂ (MW)", 10.0, 5000.0, 2000.0, 10.0)
hf2 = st.number_input("h_f at P₂ (m)", 0.0, 500.0, 70.0, 0.1)

k_fit, n_fit = fit_hf_k_n_from_anchors(gross_head, eta_t, [(P1, hf1), (P2, hf2)])
if not (np.isnan(k_fit) or np.isnan(n_fit)):
    st.info(f"Fitted curve:  h_f ≈ {k_fit:.6g} · Q^{n_fit:.3f}   (Q in m³/s, h_f in m)")

    # Show curve with current operating points
    Q_show = np.linspace(0.1, max(1.2 * out_max["Q_total"], 10.0), 200)
    hf_show = k_fit * Q_show**n_fit
    fig_fit = make_subplots(specs=[[{"secondary_y": False}]])
    fig_fit.add_trace(go.Scatter(x=Q_show, y=hf_show, name="h_f(Q) fit", line=dict(width=3)))
    if not np.isnan(out_design["Q_total"]):
        fig_fit.add_trace(go.Scatter(x=[out_design["Q_total"]], y=[out_design["hf"]],
                                     mode="markers", name="Design point", marker=dict(size=10)))
    if not np.isnan(out_max["Q_total"]):
        fig_fit.add_trace(go.Scatter(x=[out_max["Q_total"]], y=[out_max["hf"]],
                                     mode="markers", name="Max point", marker=dict(size=10)))
    fig_fit.update_layout(title="Fitted head-loss curve (didactic)",
                          xaxis_title="Total Q (m³/s)", yaxis_title="h_f (m)", height=420)
    st.plotly_chart(fig_fit, use_container_width=True)
else:
    st.caption("Provide sensible anchors to view a fitted h_f = k·Qⁿ curve.")

# ------------------------------- Section 7: Surge Tank -------------------------------
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

# ------------------------------- Section 8A: Equations -------------------------------
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

# ------------------------------- Section 8B: Reference Tables -------------------------------
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

# ------------------------------- Section 9: Downloads & Bibliography -------------------------------
st.header("10) Downloads & Bibliography")

bundle = {
    "reservoirs": {"upper": {"HWL": HWL_u, "LWL": LWL_u}, "lower": {"HWL": HWL_l, "TWL": TWL_l}},
    "penstock": {"N": N_pen, "D": D_pen, "L": L_pen, "f": f, "K_sum": K_sum, "material": f_material},
    "efficiency": {"eta_t": eta_t},
    "operating": {"design": out_design, "max": out_max},
    "tunnel": {
        "ri": ri, "t": t, "re": re, "pi_MPa": pi_MPa, "pext": pext, "ft_MPa": ft_MPa,
        "CRV": CRV, "FRV": FRV, "sigma_outer": sigma_outer, "pext_req": pext_req
    },
    "surge": surge,
    "hf_fit": {"k": k_fit, "n": n_fit}
}
st.download_button("Download JSON", data=json.dumps(bundle, indent=2), file_name="phes_results.json")

# CSV params (flat)
flat = {
    "HWL_u": HWL_u, "LWL_u": LWL_u, "HWL_l": HWL_l, "TWL_l": TWL_l,
    "N_pen": N_pen, "D_pen": D_pen, "L_pen": L_pen, "f": f, "K_sum": K_sum,
    "eta_t": eta_t, "P_design_MW": P_design, "P_max_MW": P_max,
    "hnet_design_m": out_design["h_net"], "Q_total_design_m3s": out_design["Q_total"],
    "v_design_ms": out_design["v"], "hf_design_m": out_design["hf"],
    "hnet_max_m": out_max["h_net"], "Q_total_max_m3s": out_max["Q_total"],
    "v_max_ms": out_max["v"], "hf_max_m": out_max["hf"],
    "ri_m": ri, "re_m": re, "pi_MPa": pi_MPa, "pext_MPa": pext, "ft_MPa": ft_MPa,
    "CRV_m": CRV, "FRV": FRV, "A_h_m2": area_circle(D_pen), "A_s_m2": surge["As"], "Tn_s": surge["Tn"],
    "k_fit": k_fit, "n_fit": n_fit
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
