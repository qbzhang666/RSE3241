# PHES / Hydropower Design Teaching App (Snowy 2.0 & Kidston)
# Run: streamlit run app.py

import io
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================ App setup & style ============================
st.set_page_config(page_title="PHES Design Teaching App", layout="wide")
plt.style.use("default")
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "axes.edgecolor": "0.15",
    "axes.linewidth": 1.0,
})

st.title("Pumped Hydro / PHES Design – Snowy 2.0 & Kidston (Sections 5–9)")

# ============================ Constants & helpers ============================
g = 9.81
rho = 1000.0

def power_MW(Q, h_net, eta):
    return rho * g * Q * h_net * eta / 1e6

def discharge_from_power_MW(P_MW, h_net, eta):
    return P_MW * 1e6 / (rho * g * h_net * eta)

def segment_headloss(L, D, lam, Ksum, Q):
    """Darcy–Weisbach per segment with local losses."""
    A = math.pi * D**2 / 4
    v = Q / A if A > 0 else float('nan')
    hf_fric = lam * (L / D) * (v**2 / (2*g)) if D > 0 else float('nan')
    hf_local = Ksum * (v**2 / (2*g))
    return dict(A=A, v=v, hf_fric=hf_fric, hf_local=hf_local, hf=hf_fric + hf_local)

def fit_hf_k_n_from_anchors(h_gross, eta_t, anchors):
    """Fit h_f = k Q^n using two anchor points (P, h_f)."""
    (P1, hf1), (P2, hf2) = anchors
    Q1 = discharge_from_power_MW(P1, h_gross - hf1, eta_t)
    Q2 = discharge_from_power_MW(P2, h_gross - hf2, eta_t)
    n = math.log(hf2 / hf1) / math.log(Q2 / Q1)
    k = hf1 / (Q1 ** n)
    return k, n

def solve_Q_hf_net(P_MW, h_gross, eta_t, k, n, q0=None, tol=1e-10, itmax=200):
    """Fixed-point iteration with h_net = h_gross − k Q^n."""
    if q0 is None:
        q0 = discharge_from_power_MW(P_MW, h_gross, eta_t)
    Q = q0
    for _ in range(itmax):
        hf = k * (Q ** n)
        hnet = h_gross - hf
        Q_new = discharge_from_power_MW(P_MW, hnet, eta_t)
        if abs(Q_new - Q) < tol:
            Q = Q_new
            break
        Q = 0.5*(Q + Q_new)
    hf = k * (Q ** n)
    return dict(Q=Q, hf=hf, h_net=h_gross - hf)

# Thick-cylinder (Lame) – correct forms
# σ_r(r) = A − B/r² ; σ_θ(r) = A + B/r²
# A = (p_i r_i² − p_e r_o²)/(r_o² − r_i²)
# B = (p_i − p_e) r_i² r_o² /(r_o² − r_i²)
def lame_A(pi_, pe_, ri, ro):
    return (pi_ * ri**2 - pe_ * ro**2) / (ro**2 - ri**2)

def lame_B(pi_, pe_, ri, ro):
    return (pi_ - pe_) * ri**2 * ro**2 / (ro**2 - ri**2)

def hoop_stress_Lame(pi_MPa, pe_MPa, ri, ro, r):
    A = lame_A(pi_MPa, pe_MPa, ri, ro)
    B = lame_B(pi_MPa, pe_MPa, ri, ro)
    return A + B / (r**2)

def hoop_stress_inner(pi_MPa, pe_MPa, ri, ro):
    return (pi_MPa * (ro**2 + ri**2) - 2.0 * pe_MPa * ro**2) / (ro**2 - ri**2)

def required_pext_for_ft(pi_MPa, ri, ro, ft_MPa):
    return (pi_MPa * (ro**2 + ri**2) - ft_MPa * (ro**2 - ri**2)) / (2.0 * ro**2)

# Rock cover criteria
def snowy_vertical_cover(hs, gamma_w=9.81, gamma_R=26.0):
    return (hs * gamma_w) / gamma_R

def norwegian_FRV(CRV, hs, alpha_deg, gamma_w=9.81, gamma_R=26.0):
    return (CRV * gamma_R * math.cos(math.radians(alpha_deg))) / (hs * gamma_w)

def norwegian_FRM(CRM, hs, beta_deg, gamma_w=9.81, gamma_R=26.0):
    return (CRM * gamma_R * math.cos(math.radians(beta_deg))) / (hs * gamma_w)

# Cavitation (Thoma)
def thoma_sigma_available(H_atm, H_vap, TWL_oper, runner_CL, h_losses_draft, H_net):
    submergence = TWL_oper - runner_CL
    return (H_atm - H_vap + submergence - h_losses_draft) / H_net if H_net > 0 else float('nan')

def velocity_hint(v):
    if v > 8.0:  return "⚠️ Very high; high losses/erosion & strong transients."
    if v > 6.0:  return "⚠️ High; check losses and transient loads."
    if v < 1.0:  return "ℹ️ Low; may imply oversized diameter."
    return "✅ Typical for pressure conduits."

# ============================ Section 1: System Parameters ============================
st.header("1) System Parameters")

with st.sidebar:
    st.header("Presets")
    preset = st.selectbox("Preset", ["None", "Snowy 2.0 · Plateau (NEW)", "Snowy 2.0 · Plateau (DET)", "Kidston (example)"])
    if st.button("Apply preset"):
        if preset == "Snowy 2.0 · Plateau (NEW)":
            st.session_state.update(dict(HWL_u=1100.0, LWL_u=1080.0, HWL_l=450.0, TWL_l=420.0,
                                         eta_t=0.90, N=6, hf1=28.0, hf2=70.0, P1=1000.0, P2=2000.0))
        elif preset == "Snowy 2.0 · Plateau (DET)":
            st.session_state.update(dict(HWL_u=1100.0, LWL_u=1080.0, HWL_l=450.0, TWL_l=420.0,
                                         eta_t=0.90, N=6, hf1=30.0, hf2=106.0, P1=1000.0, P2=2000.0))
        elif preset == "Kidston (example)":
            st.session_state.update(dict(HWL_u=500.0, LWL_u=490.0, HWL_l=230.0, TWL_l=220.0,
                                         eta_t=0.90, N=2, hf1=6.0, hf2=18.0, P1=250.0, P2=500.0))
    st.caption("Use presets then fine-tune below.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Hydraulic Levels")
    HWL_u = st.number_input("Upper HWL (m)", value=float(st.session_state.get("HWL_u", 1100.0)))
    LWL_u = st.number_input("Upper LWL (m)", value=float(st.session_state.get("LWL_u", 1080.0)))
    HWL_l = st.number_input("Lower HWL (m)", value=float(st.session_state.get("HWL_l", 450.0)))
    TWL_l = st.number_input("Lower TWL (m)", value=float(st.session_state.get("TWL_l", 420.0)))

with col2:
    st.subheader("Plant & Penstock")
    N_penstocks = st.number_input("Number of Penstocks", min_value=1, max_value=12, value=int(st.session_state.get("N", 6)))
    eta_t = st.number_input("Turbine Efficiency ηₜ", min_value=0.70, max_value=1.00, value=float(st.session_state.get("eta_t", 0.90)), step=0.01,
                             help="Overall hydraulic + mechanical efficiency.")
    D_pen = st.number_input("Penstock Diameter D (m)", value=3.5)
    design_power = st.number_input("Design Power (MW)", value=500.0)
    max_power = st.number_input("Maximum Power (MW)", value=600.0)

# Derived head range
H_max = HWL_u - TWL_l
H_min = LWL_u - HWL_l
head_ratio = (H_min / H_max) if H_max > 0 else float('nan')
NWL_u = HWL_u - (HWL_u - LWL_u)/3.0

# Visual of ranges
fig_head, ax = plt.subplots(figsize=(7.5, 5.0))
ax.bar(['Upper Res.'], [HWL_u - LWL_u], bottom=LWL_u, color='#3498DB', alpha=0.7, width=0.5)
ax.bar(['Lower Res.'], [HWL_l - TWL_l], bottom=TWL_l, color='#2ECC71', alpha=0.7, width=0.5)
ax.annotate('', xy=(0, HWL_u), xytext=(0, TWL_l), arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=2))
ax.text(0.05, (HWL_u + TWL_l)/2, f'H_max={H_max:.1f} m', ha='left', va='center')
ax.annotate('', xy=(0.2, LWL_u), xytext=(0.2, HWL_l), arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=2))
ax.text(0.25, (LWL_u + HWL_l)/2, f'H_min={H_min:.1f} m', ha='left', va='center')
ax.set_ylabel('Elevation (m)'); ax.set_title('Reservoir operation ranges')
st.pyplot(fig_head)

c1, c2, c3 = st.columns(3)
c1.metric("H_max = HWL_u − TWL_l (m)", f"{H_max:.1f}")
c2.metric("H_min = LWL_u − HWL_l (m)", f"{H_min:.1f}")
c3.metric("Head ratio H_min/H_max", f"{head_ratio:.3f}")

with st.expander("Design equations"):
    st.latex(r"H_{\max}=\text{HWL}_u-\text{TWL}_l,\quad H_{\min}=\text{LWL}_u-\text{HWL}_l,\quad \frac{H_{\min}}{H_{\max}}")
    st.latex(r"Q_{\text{total}}=\frac{P\cdot10^6}{\rho g h_{\text{net}} \eta_t},\quad v=\frac{4Q}{\pi D^2},\quad h_f=\left(\frac{fL}{D}+\sum K\right)\frac{v^2}{2g}")

# ============================ Head Loss Parameters (auto/manual) ============================
st.header("2) Head Loss Parameters")

col1, col2 = st.columns(2)
with col1:
    L_penstock = st.number_input("Penstock length L (m)", value=500.0)
    f_options = {"New steel (welded)": 0.012, "New steel (riveted)": 0.017, "Concrete (smooth)": 0.015,
                 "Concrete (rough)": 0.022, "PVC/Plastic": 0.009}
    f_material = st.selectbox("Penstock material", options=list(f_options.keys()), index=2)
    f = st.slider("Friction factor f", min_value=0.005, max_value=0.03, step=0.001, value=float(f_options[f_material]))
    st.caption(f"Typical for {f_material}: {f_options[f_material]:.3f}")

with col2:
    st.markdown("**Local loss coefficients ΣK (builder)**")
    components = {
        "Entrance (bellmouth)": 0.15, "Entrance (square)": 0.50, "90° bend": 0.25,
        "45° bend": 0.15, "Gate valve": 0.20, "Butterfly valve": 0.30,
        "T-junction": 0.40, "Exit": 1.00
    }
    K_sum = 0.0
    for comp, k_val in components.items():
        if st.checkbox(comp, value=(comp in ["Entrance (bellmouth)", "90° bend", "Exit"])):
            K_sum += k_val
    st.markdown(f"**Total ΣK = {K_sum:.2f}** (typical 2–5)")
    auto_hf = st.checkbox("Calculate h_f automatically (recommended)", value=True)

# First-pass automatic h_f (for design and max conditions)
hf_design, hf_max = 25.0, 40.0
if auto_hf:
    # Temporary net heads to get initial Q
    h_net_design_temp = H_max - hf_design
    h_net_min_temp = H_min - hf_max
    Q_design_total_temp = (design_power * 1e6) / (rho * g * h_net_design_temp * eta_t)
    Q_max_total_temp = (max_power * 1e6) / (rho * g * h_net_min_temp * eta_t)
    Q_design_temp = Q_design_total_temp / N_penstocks
    Q_max_temp = Q_max_total_temp / N_penstocks
    A_pen = math.pi * (D_pen/2)**2
    v_design = Q_design_temp / A_pen
    v_max = Q_max_temp / A_pen
    hf_design = (f * L_penstock/D_pen + K_sum) * (v_design**2)/(2*g)
    hf_max = (f * L_penstock/D_pen + K_sum) * (v_max**2)/(2*g)
    st.info(f"Auto h_f estimates → design: {hf_design:.2f} m, max: {hf_max:.2f} m")
else:
    hf_design = st.number_input("Design head-loss h_f,design (m)", value=hf_design)
    hf_max = st.number_input("Max head-loss h_f,max (m)", value=hf_max)

# ============================ 3) Calculations (Q, v, heads) ============================
st.header("3) Discharge, Velocity & Net Head")

# Net heads
h_net_design = H_max - hf_design
h_net_min = H_min - hf_max

# Discharges & velocities
Q_design_total = (design_power * 1e6) / (rho * g * h_net_design * eta_t)
Q_max_total = (max_power * 1e6) / (rho * g * h_net_min * eta_t)
Q_design = Q_design_total / N_penstocks
Q_max = Q_max_total / N_penstocks
A_pen = math.pi * (D_pen/2)**2
v_design = Q_design / A_pen
v_max = Q_max / A_pen

results_basic = pd.DataFrame({
    "Parameter": ["Total System", "Per Penstock"],
    "Design Discharge (m³/s)": [Q_design_total, Q_design],
    "Max Discharge (m³/s)": [Q_max_total, Q_max],
    "Design Velocity (m/s)": [Q_design_total/A_pen, v_design],
    "Max Velocity (m/s)": [Q_max_total/A_pen, v_max]
})
st.dataframe(results_basic.style.format("{:.2f}"), use_container_width=True)

# Velocity validation
st.subheader("Velocity Validation (USBR guidance)")
colv1, colv2 = st.columns(2)
with colv1:
    st.metric("Per-penstock v_design", f"{v_design:.2f} m/s")
    st.metric("Per-penstock v_max", f"{v_max:.2f} m/s")
with colv2:
    st.markdown("- **Recommended range:** 4–6 m/s")
    st.markdown("- **Absolute max (short durations):** ~7 m/s")
if v_max > 7.0:
    st.error("⚠️ Exceeds ~7 m/s: cavitation/erosion risk. Reconsider D, K, L, or power.")
elif v_max > 6.0:
    st.warning("⚠️ Above 6 m/s: acceptable only for short periods; check transients and lining.")
elif v_max >= 4.0:
    st.success("✓ 4–6 m/s: good balance of efficiency & safety.")
else:
    st.info("ℹ️ <4 m/s: very safe but potentially uneconomic (oversized D).")

# ============================ 4) Interactive System Curves (Plotly) ============================
st.header("4) System Characteristics (Interactive)")

Q_range = np.linspace(0.0, max(Q_max_total*1.2, 1e-6), 200)
# simple parabolic degradation of net head toward H_min (teaching model)
h_net_range = h_net_design - (h_net_design - h_net_min) * (Q_range / max(Q_max_total, 1e-6))**2
P_range = (rho * g * Q_range * h_net_range * eta_t) / 1e6

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=Q_range, y=P_range, name="Power Output", line=dict(width=3)))
# “Efficiency” proxy vs Q; (not true ηt curve—teaching visualization)
eff_proxy = np.where(Q_range>0, (P_range*1e6)/(rho*g*Q_range*h_net_range)*100, 0)
fig.add_trace(go.Scatter(x=Q_range, y=eff_proxy, name="System Efficiency (%)",
                         line=dict(width=2, dash="dot"), visible="legendonly"), secondary_y=True)

# Reference lines
fig.add_vline(x=Q_design_total, line=dict(color="green", dash="dash", width=2), annotation=dict(text="Design", xanchor="left"))
fig.add_vline(x=Q_max_total,    line=dict(color="red",   dash="dash", width=2), annotation=dict(text="Max",    xanchor="left"))

# Velocity markers (4, 6, 7 m/s)
for v_mark in [4, 6, 7]:
    Q_v = v_mark * A_pen * N_penstocks
    fig.add_vline(x=Q_v, line=dict(color="orange", width=1, dash="dot"),
                  annotation=dict(text=f"{v_mark} m/s", yanchor="bottom"))

fig.update_layout(
    title="Operating characteristics",
    xaxis_title="Total discharge Q (m³/s)",
    yaxis_title="Power (MW)",
    yaxis2_title="Efficiency (%)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=40, r=40, b=40, t=60),
)
fig.update_xaxes(rangeslider=dict(visible=True))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Chart controls"):
    colc1, colc2 = st.columns(2)
    with colc1:
        show_eff = st.checkbox("Show efficiency curve", value=False)
        show_vel = st.checkbox("Show velocity markers", value=True)
    with colc2:
        logx = st.checkbox("Log scale (Q axis)")
    fig.update_traces(selector={"name": "System Efficiency (%)"}, visible=show_eff)
    if logx: fig.update_xaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

# Operating point probe
st.subheader("Operating Point Analysis")
selected_Q = st.slider("Select discharge (m³/s)", 0.0, float(max(Q_max_total*1.2, 1.0)), float(Q_design_total), 0.1)
idx = np.abs(Q_range - selected_Q).argmin()
P_selected = P_range[idx]
h_net_selected = h_net_range[idx]
v_selected = (selected_Q / N_penstocks) / A_pen
colop1, colop2, colop3 = st.columns(3)
colop1.metric("Power Output", f"{P_selected:.1f} MW")
colop2.metric("Net Head", f"{h_net_selected:.1f} m")
colop3.metric("Flow Velocity", f"{v_selected:.1f} m/s", delta="Above limit" if v_selected>6 else None)

# ============================ 5) Waterway Profile & Runner CL ============================
st.header("5) Waterway Profile & Runner Position")

left, right = st.columns([2,1])
with left:
    st.markdown("**Option B — Chainage–Elevation Data**")
    csv_file = st.file_uploader("Upload CSV with columns: Chainage_m, Elevation_m", type=["csv"], key="profile_csv")
    if csv_file:
        df_profile = pd.read_csv(csv_file)
    else:
        df_profile = st.data_editor(pd.DataFrame({
            "Chainage_m": [0,1000,2000,3000,4000,5000],
            "Elevation_m": [1085,1083,1076,1040,980,920]
        }), num_rows="dynamic", use_container_width=True)
    if df_profile is not None and {"Chainage_m","Elevation_m"}.issubset(df_profile.columns):
        fig_profile = plt.figure(figsize=(9, 3.6))
        plt.plot(df_profile["Chainage_m"], df_profile["Elevation_m"], lw=2)
        plt.xlabel("Chainage (m)"); plt.ylabel("Elevation (m)")
        plt.title("Waterway long-section"); plt.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig_profile)

with right:
    st.markdown("**Option A — Upload profile image (PNG/JPG)**")
    uploaded_img = st.file_uploader("Profile image", type=["png","jpg","jpeg"], key="profile_img")
    if uploaded_img: st.image(uploaded_img, caption="Uploaded profile", use_column_width=True)

st.markdown("**Runner / draft-tube positioning (uses operating tailwater)**")
TWL_oper = st.number_input("Operating tailwater TWL_op (m)", value=float(TWL_l))
h_draft  = st.number_input("Draft head below TWL_op (m)", value=5.0, min_value=0.0, step=0.5)
runner_CL = TWL_oper - h_draft
st.info(f"Runner centreline = TWL_op − h_draft = **{runner_CL:.1f} m**")

with st.expander("Quick check: which level sets runner submergence?"):
    ans = st.radio("Pick one", ["Upper reservoir LWL", "Lower reservoir TWL at operation", "Upper reservoir HWL"], index=None)
    if ans:
        st.write("✅ Correct: Lower reservoir TWL at operation."
                 if ans=="Lower reservoir TWL at operation"
                 else "❌ Not quite — cavitation depends on the *tailwater* at the operating point.")

# ============================ 6) Pressure Tunnels & Lining ============================
st.header("6) Pressure Tunnels & Lining (Rock Cover + Lame)")

colrc1, colrc2, colrc3, colrc4 = st.columns(4)
with colrc1: hs = st.number_input("Hydrostatic head to crown h_s (m)", 0.0, 2000.0, 300.0, 1.0)
with colrc2: alpha = st.number_input("Tunnel inclination α (deg)", 0.0, 90.0, 20.0, 1.0)
with colrc3: beta  = st.number_input("Ground slope β (deg)", 0.0, 90.0, 40.0, 1.0)
with colrc4: gamma_R = st.number_input("Rock unit weight γ_R (kN/m³)", 15.0, 30.0, 26.0, 0.5)

CRV = snowy_vertical_cover(hs, gamma_w=9.81, gamma_R=gamma_R)
FRV = norwegian_FRV(CRV, hs, alpha, gamma_w=9.81, gamma_R=gamma_R)
CRM = 0.85 * CRV
FRM = norwegian_FRM(CRM, hs, beta, gamma_w=9.81, gamma_R=gamma_R)
cvr1, cvr2, cvr3 = st.columns(3)
cvr1.metric("C_RV (m)", f"{CRV:.1f}")
cvr2.metric("F_RV (−)", f"{FRV:.2f}")
cvr3.metric("F_RM (−)", f"{FRM:.2f}")
st.caption("Target **F_RV ≥ 1.2–1.5** (rule-of-thumb). Lower values → insufficient cover / higher hydraulic fracturing risk.")

st.markdown("**Lining stress (Lame thick-cylinder)**")
colL1, colL2, colL3, colL4 = st.columns(4)
with colL1: ri = st.number_input("Inner radius r_i (m)", 0.1, 10.0, 3.15, 0.05)
with colL2: t  = st.number_input("Lining thickness t (m)", 0.1, 2.0, 0.35, 0.01)
with colL3: pi_MPa = st.number_input("Internal pressure p_i (MPa)", 0.0, 10.0, 2.0, 0.1)
with colL4: pe_MPa = st.number_input("External pressure p_e (MPa)", 0.0, 10.0, 0.0, 0.1)
ro = ri + t
ft_MPa = st.number_input("Concrete tensile limit f_t (MPa)", 0.5, 10.0, 3.0, 0.1)

sigma_theta_ri = hoop_stress_inner(pi_MPa, pe_MPa, ri, ro)
pext_req = required_pext_for_ft(pi_MPa, ri, ro, ft_MPa)
Lr1, Lr2, Lr3 = st.columns(3)
Lr1.metric("σθ at inner (MPa)", f"{sigma_theta_ri:.2f}")
Lr2.metric("p_ext required (MPa)", f"{pext_req:.2f}")
Lr3.write("Status: " + ("✅ OK (σθ≤f_t)" if sigma_theta_ri <= ft_MPa else "⚠️ Exceeds f_t — add confinement / steel"))

if st.checkbox("Show σr/σθ distribution"):
    r_grid = np.linspace(ri*1.001, ro, 200)
    sig_t  = hoop_stress_Lame(pi_MPa, pe_MPa, ri, ro, r_grid)
    A_ = lame_A(pi_MPa, pe_MPa, ri, ro); B_ = lame_B(pi_MPa, pe_MPa, ri, ro)
    sig_r  = A_ - B_/(r_grid**2)
    fig_stress, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(r_grid, sig_t, label="σθ(r)")
    ax.plot(r_grid, sig_r, label="σr(r)")
    ax.set_xlabel("Radius r (m)"); ax.set_ylabel("Stress (MPa)")
    ax.set_title("Thick-cylinder stress distribution")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend()
    st.pyplot(fig_stress)

# ============================ 7) Head Loss via Anchors & Segments ============================
st.header("7) Head Loss: Anchor-fit & Segment Method")

colA, colB = st.columns(2)
with colA:
    condition = st.selectbox("Loss condition", ["Plateau NEW", "Plateau DETERIORATED", "Custom/Kidston"])
    if condition == "Plateau NEW":
        hf1_default, hf2_default = 28.0, 70.0
    elif condition == "Plateau DETERIORATED":
        hf1_default, hf2_default = 30.0, 106.0
    else:
        hf1_default, hf2_default = 6.0, 18.0
with colB:
    P1 = st.number_input("Anchor P1 (MW)", 100.0, 5000.0, 1000.0, 10.0)
    P2 = st.number_input("Anchor P2 (MW)", 100.0, 5000.0, 2000.0, 10.0)

h_gross_input = st.number_input("Gross rating head h_gross (m)", 10.0, 3000.0, float(H_max), 1.0)
hf1 = st.number_input("h_f at P1 (m)", 0.0, 500.0, hf1_default, 0.1)
hf2 = st.number_input("h_f at P2 (m)", 0.0, 500.0, hf2_default, 0.1)

k, n = fit_hf_k_n_from_anchors(h_gross_input, eta_t, [(P1, hf1), (P2, hf2)])
st.info(f"Fitted loss curve:  h_f = {k:.6f} · Q^{n:.3f}")

ratings = [2000.0, 2200.0, st.number_input("Custom rating P (MW)", 100.0, 5000.0, 1800.0, 10.0)]
results_list = []
for P in ratings:
    out = solve_Q_hf_net(P, h_gross_input, eta_t, k, n)
    results_list.append({"P_MW": P, **out})
df_res = pd.DataFrame(results_list)
st.dataframe(df_res.style.format({"Q":"{:.2f}", "hf":"{:.2f}", "h_net":"{:.2f}"}), use_container_width=True)

# hf–Q chart (matplotlib)
fig_hf, ax_hf = plt.subplots(figsize=(7.5, 3.4))
Q_grid_fit = np.linspace(max(1e-6, 0.5*df_res["Q"].min()), 1.5*df_res["Q"].max(), 200)
hf_grid_fit = k * Q_grid_fit**n
ax_hf.plot(Q_grid_fit, hf_grid_fit, lw=2)
ax_hf.scatter(df_res["Q"], df_res["hf"], c='r', s=45)
ax_hf.set_xlabel("Q (m³/s)"); ax_hf.set_ylabel("h_f (m)"); ax_hf.set_title("Fitted h_f = k·Q^n")
st.pyplot(fig_hf)

with st.expander("Sensitivity: roughness effect on net head"):
    Qg = np.linspace(0.5*df_res["Q"].min(), 1.5*df_res["Q"].max(), 150)
    hf_soft = (0.8*k) * Qg**n
    hf_hard = (1.2*k) * Qg**n
    hnet_soft = h_gross_input - hf_soft
    hnet_hard = h_gross_input - hf_hard
    fig_sens, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(Qg, hnet_soft, label="0.8k (smoother)")
    ax.plot(Qg, hnet_hard, label="1.2k (rougher)")
    ax.set_xlabel("Q (m³/s)"); ax.set_ylabel("Net head (m)")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend()
    st.pyplot(fig_sens)

# Segment method example (shared + penstock)
st.markdown("**Segment method (Darcy–Weisbach example)**")
cols = st.columns(4)
with cols[0]: L_shared = st.number_input("L_shared (m)", 0.0, 50000.0, 157.0, 1.0)
with cols[1]: D_shared = st.number_input("D_shared (m)", 0.1, 10.0, 4.8, 0.1)
with cols[2]: lam_shared = st.number_input("λ_shared (−)", 0.001, 0.10, 0.015, 0.001, format="%.3f")
with cols[3]: Ksum_shared = st.number_input("ΣK_shared (−)", 0.0, 20.0, 4.0, 0.1)

cols = st.columns(4)
with cols[0]: L_pen = st.number_input("L_penstock (m)", 0.0, 50000.0, 106.0, 1.0)
with cols[1]: D_pen_in = st.number_input("D_pen_in (m)", 0.1, 10.0, 2.7, 0.1)
with cols[2]: D_pen_out = st.number_input("D_pen_out (m)", 0.1, 10.0, 2.1, 0.1)
with cols[3]: lam_pen = st.number_input("λ_pen (−)", 0.001, 0.10, 0.018, 0.001, format="%.3f")

Q_total_fit = float(df_res.loc[0, "Q"]) if not df_res.empty else 0.0
Q_unit_fit  = Q_total_fit / N_penstocks if N_penstocks>0 else 0.0
Q_shared_fit = 2 * Q_unit_fit

seg_shared = segment_headloss(L_shared, D_shared, lam_shared, Ksum_shared, Q_shared_fit)
seg_pen    = segment_headloss(L_pen, 0.5*(D_pen_in + D_pen_out), lam_pen, 2.0, Q_unit_fit)

df_seg = pd.DataFrame([
    {"segment":"shared (2 units)", **seg_shared},
    {"segment":"penstock (per unit, avg D)", **seg_pen},
])
st.dataframe(df_seg.style.format({"A":"{:.2f}","v":"{:.2f}","hf_fric":"{:.2f}","hf_local":"{:.2f}","hf":"{:.2f}"}),
             use_container_width=True)
st.caption(f"Shared conduit velocity hint: {velocity_hint(seg_shared['v'])}")
st.caption(f"Penstock avg-D velocity hint: {velocity_hint(seg_pen['v'])}")

# ============================ 8) Cavitation (Thoma σ) ============================
st.header("8) Cavitation check (Thoma σ)")

colc1, colc2, colc3, colc4 = st.columns(4)
with colc1: H_atm = st.number_input("Atmospheric head H_atm (m)", 0.0, 15.0, 10.33, 0.01)
with colc2: H_vap = st.number_input("Vapour head H_vap (m)", 0.0, 2.0, 0.30, 0.01)
with colc3: h_losses_draft = st.number_input("Draft-tube losses (m)", 0.0, 10.0, 1.0, 0.1)
with colc4: sigma_req = st.number_input("σ_required (vendor)", 0.0, 1.0, 0.10, 0.01)

if not df_res.empty:
    H_net_ref = float(df_res.loc[0, "h_net"])
    sigma_av = thoma_sigma_available(H_atm, H_vap, TWL_oper, runner_CL, h_losses_draft, H_net_ref)
    st.write(f"σ_available @ {df_res.loc[0,'P_MW']:.0f} MW (H_net={H_net_ref:.1f} m): **{sigma_av:.3f}**")
    st.write("Status: " + ("✅ Meets requirement" if sigma_av >= sigma_req
                           else "⚠️ Increase submergence / reduce losses / raise H_net"))

# ============================ 9) Surge Tank (first cut) ============================
st.header("9) Surge Tank (first-cut sizing)")

Ah = math.pi * D_shared**2 / 4 if "D_shared" in locals() and D_shared>0 else float('nan')
Lh = st.number_input("Headrace length to surge tank L_h (m)", 100.0, 100000.0, 15000.0, 100.0)
ratio = st.number_input("Area ratio A_s/A_h", 1.0, 10.0, 4.0, 0.1)

def surge_tank_first_cut(Ah, Lh, ratio=4.0):
    As = ratio * Ah
    omega_n = math.sqrt(g * Ah / (Lh * As)) if (Lh>0 and As>0) else float('nan')
    Tn = 2*math.pi/omega_n if omega_n and omega_n>0 else float('nan')
    return dict(As=As, omega_n=omega_n, Tn=Tn)

first_cut = surge_tank_first_cut(Ah, Lh, ratio=ratio)
cs1, cs2, cs3 = st.columns(3)
cs1.metric("A_h (m²)", f"{Ah:.2f}")
cs2.metric("A_s (m²)", f"{first_cut['As']:.2f}")
cs3.metric("Natural period T_n (s)", f"{first_cut['Tn']:.1f}")
with st.expander("Equations & notes"):
    st.latex(r"\frac{A_s}{A_h}\approx 3\text{–}5,\quad \omega_n \approx \sqrt{\frac{g A_h}{L_h A_s}},\quad T_n = \frac{2\pi}{\omega_n}")
    st.caption("Confirm with transient surge analysis; add throttling/air-cushion where appropriate.")

# ============================ Downloads ============================
st.header("Downloads")

bundle = {
    "inputs": {
        "levels": {"HWL_u": HWL_u, "LWL_u": LWL_u, "HWL_l": HWL_l, "TWL_l": TWL_l, "TWL_op": TWL_oper},
        "plant": {"N_penstocks": N_penstocks, "eta_t": eta_t, "D_pen": D_pen,
                  "design_power": design_power, "max_power": max_power},
        "loss_params": {"L_penstock": L_penstock, "f": f, "K_sum": K_sum, "auto_hf": auto_hf,
                        "hf_design": hf_design, "hf_max": hf_max},
        "anchor_fit": {"P1": P1, "hf1": hf1, "P2": P2, "hf2": hf2, "h_gross": h_gross_input, "k": k, "n": n},
        "segments": {"shared": {"L": L_shared, "D": D_shared, "lam": lam_shared, "Ksum": Ksum_shared},
                     "penstock": {"L": L_pen, "D_in": D_pen_in, "D_out": D_pen_out, "lam": lam_pen}},
        "lining": {"ri": ri, "t": t, "ro": ro, "pi_MPa": pi_MPa, "pe_MPa": pe_MPa, "ft_MPa": ft_MPa},
        "surge": {"Lh": Lh, "ratio": ratio},
        "cavitation": {"H_atm": H_atm, "H_vap": H_vap, "h_losses_draft": h_losses_draft, "sigma_req": sigma_req},
    },
    "derived": {
        "head_range": {"H_max": H_max, "H_min": H_min, "Hmin_over_Hmax": head_ratio, "NWL_u": NWL_u},
        "basic_results": results_basic.to_dict(orient="records"),
        "anchor_results": df_res.to_dict(orient="records"),
        "segment_results": df_seg.to_dict(orient="records"),
        "lining": {"sigma_theta_ri": sigma_theta_ri, "p_ext_required": pext_req},
        "surge_first_cut": first_cut
    }
}

st.download_button("Download JSON", data=json.dumps(bundle, indent=2), file_name="phes_results.json")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    pd.DataFrame([bundle["inputs"]]).to_excel(writer, sheet_name="Inputs", index=False)
    pd.DataFrame(bundle["derived"]["basic_results"]).to_excel(writer, sheet_name="Basic", index=False)
    pd.DataFrame(bundle["derived"]["anchor_results"]).to_excel(writer, sheet_name="AnchorFit", index=False)
    pd.DataFrame(bundle["derived"]["segment_results"]).to_excel(writer, sheet_name="Segments", index=False)
    pd.DataFrame([bundle["derived"]["lining"]]).to_excel(writer, sheet_name="Lining", index=False)
    pd.DataFrame([bundle["derived"]["surge_first_cut"]]).to_excel(writer, sheet_name="SurgeFirstCut", index=False)
st.download_button("Download Excel", data=excel_buffer.getvalue(), file_name="phes_results.xlsx")

# ============================ References ============================
with st.expander("References & teaching notes"):
    st.markdown("""
- USBR Design Standards (Hydropower Penstocks), ICOLD Bulletins, ASCE/USACE guidance for **f** and **K**.
- **Teaching notes**: Runner submergence uses *tailwater at operation*; Lame stresses illustrate need for confinement/steel; velocities drive losses & erosion; aim for **F_RV ≥ 1.2–1.5**; confirm surge tank with transient analysis.
""")
