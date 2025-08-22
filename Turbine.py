import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up the page
st.set_page_config(page_title="Hydropower Turbine Selection", layout="wide")
st.title("Hydropower Turbine Selection Tool")
st.markdown("Based on RSE3241 Week 8: Turbine Selection and Energy")

# Sidebar for user inputs
st.sidebar.header("Project Parameters")

# Input parameters
h = st.sidebar.number_input("Head (m)", min_value=1.0, max_value=2000.0, value=100.0, step=1.0)
Q = st.sidebar.number_input("Discharge (m³/s)", min_value=0.1, max_value=1000.0, value=10.0, step=0.1)
power_req = st.sidebar.number_input("Required Power Output (MW)", min_value=0.1, max_value=5000.0, value=50.0, step=0.1)
efficiency = st.sidebar.slider("Expected Efficiency (η)", min_value=0.1, max_value=0.95, value=0.85, step=0.01)

# Constants
g = 9.81  # m/s²
rho = 1000  # kg/m³

# Calculate power
P = Q * h * g * rho * efficiency / 1e6  # Convert to MW

# Display calculated power
st.sidebar.metric("Calculated Power Output", f"{P:.2f} MW")

# Turbine selection logic
def select_turbine(h, Q):
    if h > 300 and Q < 50:
        return "Pelton"
    elif 50 <= h <= 300 and Q < 100:
        return "Francis"
    elif h < 50 and Q >= 10:
        return "Kaplan"
    elif h < 20 and Q >= 20:
        return "Bulb"
    else:
        return "Cross-flow or Special Application"

# Get turbine recommendation
recommended_turbine = select_turbine(h, Q)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Turbine Recommendation")
    st.metric("Recommended Turbine Type", recommended_turbine)
    
    # Display turbine information
    if recommended_turbine == "Pelton":
        st.info("""
        **Pelton Turbine Characteristics:**
        - Type: Impulse
        - Best for: High head (>300m), low flow
        - Efficiency: 80-90%
        - Good for fluctuating discharges
        """)
    elif recommended_turbine == "Francis":
        st.info("""
        **Francis Turbine Characteristics:**
        - Type: Reaction
        - Best for: Medium head (50-300m), medium flow
        - Efficiency: 90-94%
        - Most common type worldwide
        """)
    elif recommended_turbine == "Kaplan":
        st.info("""
        **Kaplan Turbine Characteristics:**
        - Type: Reaction
        - Best for: Low head (<50m), high flow
        - Efficiency: 90-94%
        - Adjustable blades for variable flow
        """)
    elif recommended_turbine == "Bulb":
        st.info("""
        **Bulb Turbine Characteristics:**
        - Type: Reaction
        - Best for: Very low head (<20m), very high flow
        - Efficiency: 90-94%
        - Compact design for low-head applications
        """)
    else:
        st.info("""
        **Special Application:**
        - Consider cross-flow or other specialized turbines
        - May need custom design
        - Consult with turbine manufacturers
        """)

with col2:
    st.header("Turbine Application Ranges")
    
    # Create a simple visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define turbine application areas (simplified)
    h_range = np.logspace(0, 3, 100)
    
    # Pelton range
    pelton_q = 50 * np.ones_like(h_range)
    ax.fill_betweenx(h_range, 0, pelton_q, where=(h_range>300), alpha=0.3, label='Pelton', color='red')
    
    # Francis range
    francis_q = np.where(h_range<50, 100, np.where(h_range>300, 50, 100))
    ax.fill_betweenx(h_range, 0, francis_q, where=((h_range>=50) & (h_range<=300)), alpha=0.3, label='Francis', color='blue')
    
    # Kaplan range
    kaplan_q = 100 * np.ones_like(h_range)
    ax.fill_betweenx(h_range, 0, kaplan_q, where=(h_range<50), alpha=0.3, label='Kaplan', color='green')
    
    # Bulb range
    bulb_q = 100 * np.ones_like(h_range)
    ax.fill_betweenx(h_range, 20, bulb_q, where=(h_range<20), alpha=0.3, label='Bulb', color='purple')
    
    # Plot current point
    ax.plot(Q, h, 'ko', markersize=10, label='Your Project')
    
    ax.set_xlabel('Discharge Q (m³/s)')
    ax.set_ylabel('Head h (m)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Turbine Selection Chart')
    ax.legend()
    ax.grid(True, which="both", ls="-")
    
    st.pyplot(fig)

# Energy calculation section
st.header("Energy Calculations")
col3, col4 = st.columns(2)

with col3:
    operating_hours = st.number_input("Annual Operating Hours", min_value=100, max_value=8760, value=4000, step=100)
    annual_energy = P * 1000 * operating_hours  # Convert to kWh
    st.metric("Annual Energy Generation", f"{annual_energy/1e6:.2f} MWh")

with col4:
    round_trip_efficiency = st.slider("Round-trip Efficiency (for pumped storage)", 
                                    min_value=0.5, max_value=0.9, value=0.75, step=0.01)
    pumping_energy = annual_energy / round_trip_efficiency
    st.metric("Pumping Energy Required", f"{pumping_energy/1e6:.2f} MWh")

# Additional information
st.header("Additional Information")
with st.expander("Turbine Efficiency Curves"):
    st.image("https://github.com/qbzhang666/RSE3241/blob/main/Turbine%20Selection.png", 
             caption="Typical turbine efficiency curves (Source: Wikipedia)")

with st.expander("Formulas Used"):
    st.latex(r"P(MW) = Q \times h \times g \times \rho \times \eta / 10^6")
    st.latex(r"E(MWh) = P(MW) \times T(h)")
    st.latex(r"E_{pumping}(MWh) = E_{generation}(MWh) / \alpha")

# References
st.header("References")
st.markdown("""
- RSE3241 Week 8: Turbine Selection and Energy
- IRENA (2012) Hydroelectric power generation
- IEC 61116:1992, Electromechanical equipment guide for small hydroelectric installations
- Giesecke, J.; Heimerl, S.; Mosonyi, E. Wasserkraftanlagen: Planung, Bau und Betrieb
""")
