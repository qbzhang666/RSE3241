# PHES Design App · Snowy 2.0 & Kidston (Sections 5–9)
# Fully corrected version with proper imports and configurations

import io
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl  # Added this import
import streamlit as st

# Configure plotting style
plt.style.use('default')  # Using default style which is more stable
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "axes.edgecolor": "0.15",
    "axes.linewidth": 1.25,
})

# --------------------------------- Physics helpers ---------------------------------
g = 9.81
rho = 1000.0

def power_MW(Q, h_net, eta):
    return rho * g * Q * h_net * eta / 1e6

def discharge_from_power_MW(P_MW, h_net, eta):
    return P_MW * 1e6 / (rho * g * h_net * eta)

def segment_headloss(L, D, lam, Ksum, Q):
    A = math.pi * D**2 / 4
    v = Q / A if A > 0 else float('nan')
    hf_fric = lam * (L / D) * (v**2 / (2*g)) if D > 0 else float('nan')
    hf_local = Ksum * (v**2 / (2*g))
    return dict(A=A, v=v, hf_fric=hf_fric, hf_local=hf_local, hf=hf_fric + hf_local)

def fit_hf_k_n_from_anchors(h_gross, eta_t, anchors):
    (P1, hf1), (P2, hf2) = anchors
    Q1 = discharge_from_power_MW(P1, h_gross - hf1, eta_t)
    Q2 = discharge_from_power_MW(P2, h_gross - hf2, eta_t)
    n = math.log(hf2 / hf1) / math.log(Q2 / Q1)
    k = hf1 / (Q1 ** n)
    return k, n

def solve_Q_hf_net(P_MW, h_gross, eta_t, k, n, q0=None, tol=1e-10, itmax=200):
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
        Q = 0.5*(Q+Q_new)
    hf = k * (Q ** n)
    return dict(Q=Q, hf=hf, h_net=h_gross - hf)

def hoop_stress(pi, pe, ri, r):
    """Calculate hoop stress with safeguards"""
    r_array = np.array([r]) if np.isscalar(r) else np.array(r)
    with np.errstate(divide='ignore', invalid='ignore'):
        stress = (pi*(r_array**2 + ri**2) - 2*pe*r_array**2)/(r_array**2 - ri**2)
    stress[r_array <= ri] = np.nan
    return stress.item() if np.isscalar(r) else stress

def required_pext_for_ft(pi_MPa, ri, re, ft_MPa):
    return (ft_MPa - pi_MPa) * (re**2 - ri**2) / (2.0 * re**2)

def snowy_vertical_cover(hs, gamma_w=9.81, gamma_R=26.0):
    return (hs * gamma_w) / gamma_R

def norwegian_FRV(CRV, hs, alpha_deg, gamma_w=9.81, gamma_R=26.0):
    return (CRV * gamma_R * math.cos(math.radians(alpha_deg))) / (hs * gamma_w)

def norwegian_FRM(CRM, hs, beta_deg, gamma_w=9.81, gamma_R=26.0):
    return (CRM * gamma_R * math.cos(math.radians(beta_deg))) / (hs * gamma_w)

def surge_tank_first_cut(Ah, Lh, ratio=4.0):
    As = ratio * Ah
    omega_n = math.sqrt(g * Ah / (Lh * As))
    Tn = 2*math.pi/omega_n
    return dict(As=As, omega_n=omega_n, Tn=Tn)

def thoma_sigma_available(H_atm, H_vap, TWL, runner_CL, h_losses_draft, H_net):
    submergence = TWL - runner_CL
    return (H_atm - H_vap + submergence - h_losses_draft) / H_net if H_net > 0 else float('nan')

# --------------------------------- UI ---------------------------------
st.set_page_config(page_title="PHES Design (Snowy & Kidston)", layout="wide")
st.title("PHES Design App · Snowy 2.0 & Kidston (Sections 5–9)")


# Presets
with st.sidebar:
    st.header("Presets & Settings")
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

    eta_t = st.number_input("Turbine efficiency ηₜ", 0.70, 1.00, float(st.session_state.get("eta_t", 0.90)), 0.01)
    eta_p = st.number_input("Pump efficiency ηₚ (ref.)", 0.60, 1.00, 0.88, 0.01)
    N = st.number_input("Units (N)", 1, 20, int(st.session_state.get("N", 6)), 1)
    st.caption("All units SI; water ρ=1000 kg/m³, g=9.81 m/s².")

# ---------------------------- 5) Levels & rating head ----------------------------
st.subheader("5) Reservoir Levels & Head Analysis")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Upper reservoir**")
    HWL_u = st.number_input("Upper HWL (m)", 0.0, 3000.0, float(st.session_state.get("HWL_u", 1100.0)), 1.0)
    LWL_u = st.number_input("Upper LWL (m)", 0.0, 3000.0, float(st.session_state.get("LWL_u", 1080.0)), 1.0)
with col2:
    st.markdown("**Lower reservoir**")
    HWL_l = st.number_input("Lower HWL (m)", 0.0, 3000.0, float(st.session_state.get("HWL_l", 450.0)), 1.0)
    TWL_l = st.number_input("Lower TWL (m)", 0.0, 3000.0, float(st.session_state.get("TWL_l", 420.0)), 1.0)

# Calculate head parameters
gross_head = HWL_u - TWL_l
NWL_u = HWL_u - (HWL_u - LWL_u)/3.0
head_fluct_rate = (LWL_u - TWL_l)/(HWL_u - TWL_l) if (HWL_u - TWL_l) != 0 else float('nan')
max_head = HWL_u - TWL_l
min_head = LWL_u - HWL_l

# Visualization
fig_reservoir, ax = plt.subplots(figsize=(8, 6))
ax.bar(['Upper Reservoir'], [HWL_u - LWL_u], bottom=LWL_u, color='#3498DB', alpha=0.7)
ax.bar(['Lower Reservoir'], [HWL_l - TWL_l], bottom=TWL_l, color='#2ECC71', alpha=0.7)

ax.annotate('', xy=(0, HWL_u), xytext=(0, TWL_l),
           arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=2))
ax.text(0.1, (HWL_u + TWL_l)/2, f'Max Head = {max_head:.1f} m', ha='left', va='center')

ax.annotate('', xy=(0.2, LWL_u), xytext=(0.2, HWL_l),
           arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=2))
ax.text(0.3, (LWL_u + HWL_l)/2, f'Min Head = {min_head:.1f} m', ha='left', va='center')

ax.set_ylabel('Elevation (m)')
ax.set_title('Reservoir Operation Ranges')
ax.grid(True, linestyle='--', alpha=0.4)
st.pyplot(fig_reservoir)

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Gross head h_gross (m)", f"{gross_head:.1f}")
c2.metric("NWL (upper) (m)", f"{NWL_u:.1f}")
c3.metric("Head fluctuation rate", f"{head_fluct_rate:.3f}")

# ---------------------------- 6) Waterway profile ----------------------------
st.subheader("6) Waterway Profile")

left, right = st.columns([2,1])
with right:
    st.markdown("**Option A — Upload profile image**")
    uploaded_img = st.file_uploader("Profile image", type=["png","jpg","jpeg"], key="profile_img")
with left:
    st.markdown("**Option B — Provide chainage data**")
    csv_file = st.file_uploader("Upload CSV (Chainage_m, Elevation_m)", type=["csv"], key="profile_csv")
    if csv_file:
        df_profile = pd.read_csv(csv_file)
    else:
        df_profile = pd.DataFrame({
            "Chainage_m": [0, 1000, 2000, 3000, 4000, 5000],
            "Elevation_m": [1085, 1083, 1076, 1040, 980, 920],
        })
        df_profile = st.data_editor(df_profile, num_rows="dynamic", use_container_width=True)

if df_profile is not None and {"Chainage_m","Elevation_m"}.issubset(df_profile.columns):
    fig_profile = plt.figure(figsize=(10,4))
    plt.plot(df_profile["Chainage_m"], df_profile["Elevation_m"], 'b-', lw=2)
    plt.xlabel("Chainage (m)"); plt.ylabel("Elevation (m)")
    plt.title("Waterway Long-Section"); plt.grid(True)
    st.pyplot(fig_profile)

if uploaded_img:
    st.image(uploaded_img, caption="Uploaded profile", use_column_width=True)

# Runner CL
h_draft = st.number_input("Draft head below lower LWL (m)", 0.0, 200.0, 5.0, 0.5)
runner_CL = TWL_l - h_draft
st.write(f"Runner centreline elevation ≈ {runner_CL:.1f} m")

# ---------------------------- 7) Pressure tunnels ----------------------------
st.subheader("7) Pressure Tunnels & Shafts")

col1, col2, col3, col4 = st.columns(4)
with col1:
    hs = st.number_input("Hydrostatic head to crown h_s (m)", 0.0, 2000.0, 300.0, 1.0)
with col2:
    alpha = st.number_input("Tunnel inclination α (deg)", 0.0, 90.0, 20.0, 1.0)
with col3:
    beta = st.number_input("Ground slope β (deg)", 0.0, 90.0, 40.0, 1.0)
with col4:
    gamma_R = st.number_input("Rock unit weight γ_R (kN/m³)", 15.0, 30.0, 26.0, 0.5)

CRV = snowy_vertical_cover(hs, gamma_w=9.81, gamma_R=gamma_R)
FRV = norwegian_FRV(CRV, hs, alpha, gamma_w=9.81, gamma_R=gamma_R)
CRM = 0.85 * CRV
FRM = norwegian_FRM(CRM, hs, beta, gamma_w=9.81, gamma_R=gamma_R)

col1, col2, col3 = st.columns(3)
col1.metric("C_RV (m)", f"{CRV:.1f}")
col2.metric("F_RV (-)", f"{FRV:.2f}")
col3.metric("F_RM (-)", f"{FRM:.2f}")

# Lining stress analysis
st.markdown("**Lining Stress Analysis**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    ri = st.number_input("Inner radius r_i (m)", 0.1, 10.0, 3.15, 0.05)
with col2:
    t = st.number_input("Lining thickness t (m)", 0.1, 2.0, 0.35, 0.01)
with col3:
    pi_MPa = st.number_input("Internal pressure p_i (MPa)", 0.0, 10.0, 2.0, 0.1)
with col4:
    pext_manual = st.number_input("External pressure p_ext (MPa)", 0.0, 10.0, 0.0, 0.1)
re = ri + t
ft_MPa = st.number_input("Concrete tensile strength f_t (MPa)", 1.0, 10.0, 3.0, 0.1)

sigma_theta_i = hoop_stress(pi_MPa, pext_manual, ri, re)
pext_req = required_pext_for_ft(pi_MPa, ri, re, ft_MPa)

# Stress distribution plot
fig_stress, ax = plt.subplots(figsize=(8,5))
r_extended = np.linspace(ri*1.001, 6.0, 200)
sigma_t_extended = hoop_stress(pi_MPa, pext_manual, ri, r_extended)

ax.plot(r_extended, sigma_t_extended, 'b-', lw=2.5, label='Hoop Stress')
ax.axhline(ft_MPa, color='g', ls='-.', lw=2, label=f'Tensile Strength ({ft_MPa} MPa)')
ax.axvline(ri, color='k', ls=':', label=f'Inner Radius ({ri} m)')
ax.axvline(re, color='k', ls='--', label=f'Outer Radius ({re} m)')

ax.fill_between(r_extended, sigma_t_extended, ft_MPa, 
               where=(sigma_t_extended > ft_MPa), 
               color='red', alpha=0.2, label='Overstress Region')

ax.set_xlabel('Radius (m)'); ax.set_ylabel('Stress (MPa)')
ax.set_title('Lining Stress Distribution')
ax.set_xlim(ri-0.5, 6.0); ax.grid(True); ax.legend()
st.pyplot(fig_stress)

col1, col2, col3 = st.columns(3)
col1.metric("σ_θ(r_i) (MPa)", f"{sigma_theta_i:.2f}")
col2.metric("p_ext required (MPa)", f"{pext_req:.2f}")
status = "✅ OK (no cracking)" if sigma_theta_i <= ft_MPa else "⚠️ Cracking likely"
col3.metric("Status", status)

# ---------------------------- 8) Head losses ----------------------------
st.subheader("8) Head Loss Analysis")

colA, colB = st.columns(2)
with colA:
    condition = st.selectbox("Condition", ["Plateau NEW", "Plateau DETERIORATED", "Custom/Kidston"])
    if condition == "Plateau NEW":
        hf1_default, hf2_default = 28.0, 70.0
    elif condition == "Plateau DETERIORATED":
        hf1_default, hf2_default = 30.0, 106.0
    else:
        hf1_default, hf2_default = 6.0, 18.0
with colB:
    P1 = st.number_input("Anchor P1 (MW)", 100.0, 5000.0, 1000.0, 10.0)
    P2 = st.number_input("Anchor P2 (MW)", 100.0, 5000.0, 2000.0, 10.0)

h_gross_input = st.number_input("Gross head for rating (m)", 10.0, 3000.0, gross_head, 1.0)
hf1 = st.number_input("h_f at P1 (m)", 0.0, 500.0, hf1_default, 0.1)
hf2 = st.number_input("h_f at P2 (m)", 0.0, 500.0, hf2_default, 0.1)

k, n = fit_hf_k_n_from_anchors(h_gross_input, eta_t, [(P1, hf1), (P2, hf2)])
st.write(f"Fitted loss curve: h_f = {k:.6f}·Q^{n:.3f}")

ratings = [2000.0, 2200.0, st.number_input("Custom rating P (MW)", 100.0, 5000.0, 1800.0, 10.0)]
results = []
for P in ratings:
    out = solve_Q_hf_net(P, h_gross_input, eta_t, k, n)
    results.append({"P_MW": P, **out})

df_res = pd.DataFrame(results)
st.dataframe(df_res.style.format({"Q":"{:.2f}", "hf":"{:.2f}", "h_net":"{:.2f}"}))

# Plot hf vs Q
fig_hf, ax = plt.subplots(figsize=(8,4))
Q_grid = np.linspace(max(1.0, 0.5*df_res["Q"].min()), 1.5*df_res["Q"].max(), 200)
hf_grid = k * Q_grid**n
ax.plot(Q_grid, hf_grid, 'b-', lw=2)
ax.scatter(df_res["Q"], df_res["hf"], c='r', s=50)
ax.set_xlabel("Q (m³/s)"); ax.set_ylabel("h_f (m)"); ax.grid(True)
st.pyplot(fig_hf)

# ---------------------------- 9) Surge tanks ----------------------------
st.subheader("9) Surge Tanks (First Cut)")

Ah = math.pi * (4.8**2)/4  # Shared conduit area
Lh = st.number_input("Headrace length to surge tank L_h (m)", 100.0, 100000.0, 15000.0, 100.0)
ratio = st.number_input("Area ratio A_s/A_h", 1.0, 10.0, 4.0, 0.1)
first_cut = surge_tank_first_cut(Ah, Lh, ratio=ratio)

col1, col2, col3 = st.columns(3)
col1.metric("A_h (m²)", f"{Ah:.1f}")
col2.metric("A_s (m²)", f"{first_cut['As']:.1f}")
col3.metric("Natural period T_n (s)", f"{first_cut['Tn']:.1f}")

# ---------------------------- Downloads ----------------------------
st.subheader("Download Results")
json_data = json.dumps({
    "inputs": {
        "reservoirs": {"upper": {"HWL": HWL_u, "LWL": LWL_u}, "lower": {"HWL": HWL_l, "TWL": TWL_l}},
        "efficiencies": {"eta_t": eta_t, "eta_p": eta_p},
        "lining": {"ri": ri, "t": t, "pi_MPa": pi_MPa, "ft_MPa": ft_MPa},
        "head_loss": {"k": k, "n": n, "anchors": [(P1, hf1), (P2, hf2)]},
        "surge_tank": {"Lh": Lh, "ratio": ratio}
    },
    "results": {
        "head": {"gross": gross_head, "max": max_head, "min": min_head, "fluctuation_rate": head_fluct_rate},
        "lining": {"sigma_theta_i": sigma_theta_i, "pext_req": pext_req},
        "surge_tank": first_cut,
        "ratings": results
    }
}, indent=2)

st.download_button("Download JSON", data=json_data, file_name="phes_results.json")

# Excel export
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer) as writer:
    pd.DataFrame(results).to_excel(writer, sheet_name="Ratings")
    pd.DataFrame([{"Parameter": "k", "Value": k}, {"Parameter": "n", "Value": n}]).to_excel(writer, sheet_name="HeadLoss")
st.download_button("Download Excel", data=excel_buffer.getvalue(), file_name="phes_results.xlsx")

st.caption("Note: Final design should be validated with transient modeling and vendor data.")
