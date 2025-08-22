# streamlit_turbine_selection_clean.py

import math
import streamlit as st

# -------------------------------
# Constants
# -------------------------------
g = 9.81       # m/s²
rho = 1000.0   # kg/m³

st.title("🌊 Turbine Selection & Energy Generation")
st.markdown("Teaching tool for turbine selection, efficiency, and energy balance.")

# ---------------------------------------------------
# STEP 1: User Inputs
# ---------------------------------------------------
st.header("1. Define Inputs")

P_target_MW = st.number_input("Power per Turbine (MW)", value=125.0, step=5.0)
H_effective = st.number_input("Effective Head Hₑ (m)", value=218.0, step=1.0)
Q_design = st.number_input("Design Discharge Q (m³/s)", value=240.0, step=10.0)

# efficiencies
eta_turbine = st.slider("Turbine Efficiency (η_turbine)", 0.70, 0.98, 0.90)
eta_generator = st.slider("Generator Efficiency (η_generator)", 0.90, 0.99, 0.96)
eta_transformer = st.slider("Transformer Efficiency (η_transformer)", 0.98, 0.995, 0.99)

# ---------------------------------------------------
# STEP 2: Overall Efficiency
# ---------------------------------------------------
st.header("2. Overall Efficiency")
eta_total = eta_turbine * eta_generator * eta_transformer
st.write(f"⚙️ **Overall Efficiency η_total = {eta_total:.3f}**")

# ---------------------------------------------------
# STEP 3: Power Generation
# ---------------------------------------------------
st.header("3. Power Generation")
P = rho * g * Q_design * H_effective * eta_total
P_MW = P / 1e6
st.success(f"Net Power Output = {P_MW:.1f} MW")

# ---------------------------------------------------
# STEP 4: Pumping Power (Reverse Mode)
# ---------------------------------------------------
st.header("4. Pumping Power")
eta_pump = st.slider("Pumping Efficiency η_pump", 0.7, 0.9, 0.8)
P_pump = (rho * g * Q_design * H_effective) / eta_pump
st.write(f"Required Pumping Power = {P_pump/1e6:.1f} MW")

# ---------------------------------------------------
# STEP 5: Minimum Flow Condition
# ---------------------------------------------------
st.header("5. Minimum Flow Check")
turbine_choice = st.selectbox("Choose Turbine Type", ["Francis", "Kaplan", "Pelton"])

if turbine_choice == "Francis":
    Q_min = 0.4 * Q_design
elif turbine_choice == "Kaplan":
    Q_min = 0.2 * Q_design
elif turbine_choice == "Pelton":
    Q_min = 0.1 * Q_design
else:
    Q_min = 0.0

st.write(f"Minimum Flow for {turbine_choice} ≈ {Q_min:.1f} m³/s")
if Q_design < Q_min:
    st.error("❌ Flow below minimum – turbine shutdown required!")
else:
    st.success("✅ Flow is above minimum – safe operation.")

# ---------------------------------------------------
# STEP 6: Specific Speed (nq)
# ---------------------------------------------------
st.header("6. Specific Speed")
N_rpm = st.number_input("Runner Speed N (rpm)", value=375, step=25)
nq = N_rpm * (Q_design**0.5) / (H_effective**0.75)
st.write(f"Specific Speed n_q = {nq:.1f}")

# ---------------------------------------------------
# STEP 7: Summary
# ---------------------------------------------------
st.header("7. Summary")
st.markdown(f"""
- **Effective Head (Hₑ):** {H_effective:.2f} m  
- **Discharge Q:** {Q_design:.1f} m³/s  
- **Net Power Output:** {P_MW:.1f} MW  
- **Pumping Power:** {P_pump/1e6:.1f} MW  
- **Selected Turbine:** {turbine_choice}  
- **Minimum Flow Requirement:** {Q_min:.1f} m³/s  
- **Specific Speed (nq):** {nq:.1f}  
""")
