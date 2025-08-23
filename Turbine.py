# streamlit_turbine_selection_full.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Hydropower Turbine Selection", layout="wide")
st.title("🌊 Hydropower Turbine Selection Tool")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Project Parameters")

h = st.sidebar.number_input("Head (m)", min_value=1.0, max_value=2000.0, value=700.0, step=1.0)
Q = st.sidebar.number_input("Discharge (m³/s)", min_value=0.1, max_value=1000.0, value=400.0, step=0.1)
power_req = st.sidebar.number_input("Required Power Output (MW)", min_value=0.1, max_value=5000.0, value=2000.0, step=0.1)

# Constants
g = 9.81
rho = 1000

# ---------------- Power (initial, with guessed efficiency) ----------------
efficiency_guess = st.sidebar.slider("Expected Efficiency (η guess)", 0.1, 0.95, 0.85, step=0.01)
P = Q * h * g * rho * efficiency_guess / 1e6
st.sidebar.metric("Calculated Power Output (guess)", f"{P:.2f} MW")

# ---------------- Step 1: Turbine Type Selection ----------------
def select_turbine(h, Q):
    if h > 300 and Q < 50:
        return "Pelton"
    elif 50 <= h <= 700 and Q >= 50:
        return "Francis"
    elif h < 50 and Q >= 10:
        return "Kaplan"
    elif h < 20 and Q >= 20:
        return "Bulb"
    else:
        return "Francis"

recommended_turbine = select_turbine(h, Q)

col1, col2 = st.columns(2)

with col1:
    st.header("10) Turbine Recommendation")
    st.metric("Recommended Turbine Type", recommended_turbine)

    if recommended_turbine == "Pelton":
        st.info("Impulse turbine, high head (>300m), low flow. Efficiency ~80–90%.")
    elif recommended_turbine == "Francis":
        st.info("Reaction turbine, medium–high head (50–700m), medium–high flow. Efficiency ~90–94%.")
    elif recommended_turbine == "Kaplan":
        st.info("Reaction turbine, low head (<50m), high flow. Efficiency ~90–94%. Adjustable blades.")
    elif recommended_turbine == "Bulb":
        st.info("Reaction turbine, very low head (<20m), very high flow. Efficiency ~90–94%.")

with col2:
    st.header("Turbine Application Zones")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill([0.1, 50, 50, 0.1], [50, 50, 2000, 2000], alpha=0.3, color='red', label='Pelton')
    ax.fill([0.5, 100, 200, 10, 0.5], [20, 20, 100, 700, 700], alpha=0.3, color='blue', label='Francis')
    ax.fill([10, 1000, 1000, 10], [5, 5, 100, 100], alpha=0.3, color='green', label='Kaplan')
    ax.fill([50, 1000, 1000, 50], [5, 5, 20, 20], alpha=0.3, color='purple', label='Bulb')
    ax.plot(Q, h, 'ko', markersize=10, label='Your Project')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.1, 1000); ax.set_ylim(5, 2000)
    ax.set_xlabel('Discharge Q (m³/s)'); ax.set_ylabel('Head h (m)')
    ax.set_title('Turbine Selection Chart')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# ---------------- Step 11: Efficiency from Ns or Q/Qmax ----------------
st.header("11) Turbine Efficiency")

eta_turbine = efficiency_guess

# Digitised curves
Ns_vals = np.array([5, 10, 20, 40, 60, 80, 100])
eta_vals_francis = np.array([0.82, 0.88, 0.92, 0.93, 0.91, 0.88, 0.85])
Q_ratio = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
eta_pelton = np.array([0.70, 0.85, 0.90, 0.93, 0.94, 0.94])
eta_kaplan = np.array([0.60, 0.80, 0.90, 0.93, 0.94, 0.94])

if recommended_turbine == "Francis":
    N_rpm = st.number_input("Runner Speed N (rpm)", min_value=50, max_value=1500, value=375, step=25)
    P_gross = Q * h * g * rho / 1e6
    Ns = N_rpm * np.sqrt(P_gross) / (h**1.25)
    eta_turbine = np.interp(Ns, Ns_vals, eta_vals_francis)

    st.write(f"Specific Speed Ns = {Ns:.1f}")
    st.success(f"Turbine Efficiency from Ns curve: {eta_turbine*100:.1f}%")

    fig2, ax2 = plt.subplots()
    ax2.plot(Ns_vals, eta_vals_francis*100, 'b-', lw=2, label="Francis Ns–η Curve")
    ax2.plot(Ns, eta_turbine*100, 'ro', label="Your Design Point")
    ax2.set_xlabel("Specific Speed Ns"); ax2.set_ylabel("Efficiency (%)")
    ax2.set_title("Francis Efficiency vs Ns"); ax2.legend(); ax2.grid(True, ls="--", alpha=0.5)
    st.pyplot(fig2)

else:
    Qmax = st.number_input("Assumed Maximum Q (m³/s)", min_value=Q, max_value=2000.0, value=Q*1.5)
    q_ratio = Q / Qmax

    if recommended_turbine == "Pelton":
        eta_turbine = np.interp(q_ratio, Q_ratio, eta_pelton)
        curve = eta_pelton; label = "Pelton η–Q/Qmax Curve"
    elif recommended_turbine in ["Kaplan", "Bulb"]:
        eta_turbine = np.interp(q_ratio, Q_ratio, eta_kaplan)
        curve = eta_kaplan; label = f"{recommended_turbine} η–Q/Qmax Curve"

    st.write(f"Q/Qmax = {q_ratio:.2f}")
    st.success(f"Turbine Efficiency from Q/Qmax curve: {eta_turbine*100:.1f}%")

    fig3, ax3 = plt.subplots()
    ax3.plot(Q_ratio, curve*100, 'g-', lw=2, label=label)
    ax3.plot(q_ratio, eta_turbine*100, 'ro', label="Your Design Point")
    ax3.set_xlabel("Q/Qmax"); ax3.set_ylabel("Efficiency (%)")
    ax3.set_title(label); ax3.legend(); ax3.grid(True, ls="--", alpha=0.5)
    st.pyplot(fig3)

# ---------------- Overall Efficiency ----------------
st.subheader("Overall Efficiency")
eta_gen = st.slider("Generator Efficiency η_G", 0.90, 0.98, 0.96)
eta_tr = st.slider("Transformer Efficiency η_T", 0.98, 0.995, 0.99)
eta_total = eta_turbine * eta_gen * eta_tr
st.success(f"Overall Efficiency η_total = {eta_total*100:.1f}%")

# ---------------- Step 12: Power Generation & Pumping ----------------
st.header("12) Power Generation and Pumping")
P_gen = Q * h * g * rho * eta_total / 1e6
eta_pump = st.slider("Pumping Efficiency η_Pump", 0.70, 0.90, 0.80)
P_pump = (Q * h * g * rho) / (eta_pump*1e6)

col3, col4 = st.columns(2)
with col3: st.metric("Power Generation", f"{P_gen:.1f} MW")
with col4: st.metric("Pumping Power", f"{P_pump:.1f} MW")

# ---------------- Case Study ----------------
st.header("📌 Case Study: Kidston PHES")
st.markdown("""
- Head H = 218 m  
- Discharge Q = 240 m³/s  
- Turbine: Francis (Reversible)  
- PG ≈ 410 MW at η ≈ 0.80  
- PP ≈ 641 MW  

✅ Matches real project → validates workflow
""")
# ---------------- Equations ----------------
st.header("📐 Reference Equations")
with st.expander("Show Equations Used"):
    st.latex(r"P_{hydraulic} = \rho g Q H \quad \text{(Hydraulic Power, W)}")
    st.latex(r"P_{gen} = \rho g Q H \eta_{total} \,/\, 10^6 \quad \text{(Power Generation, MW)}")
    st.latex(r"\eta_{total} = \eta_{turbine} \cdot \eta_{generator} \cdot \eta_{transformer}")
    st.latex(r"N_s = \frac{N \sqrt{P}}{H^{5/4}} \quad \text{(Specific Speed, metric)}")
    st.latex(r"E_{annual} = P_{gen} \cdot T \quad \text{(Annual Energy, MWh)}")
    st.latex(r"P_{pump} = \frac{\rho g Q H}{\eta_{pump} \cdot 10^6} \quad \text{(Pumping Power, MW)}")
