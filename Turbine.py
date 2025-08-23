import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

# Set up the page
st.set_page_config(page_title="Hydropower Turbine Selection", layout="wide")
st.title("Hydropower Turbine Selection Tool")

# Sidebar for user inputs
st.sidebar.header("Project Parameters")

# Input parameters with updated values
h = st.sidebar.number_input("Head (m)", min_value=1.0, max_value=2000.0, value=700.0, step=1.0)
Q = st.sidebar.number_input("Discharge (m³/s)", min_value=0.1, max_value=1000.0, value=400.0, step=0.1)
power_req = st.sidebar.number_input("Required Power Output (MW)", min_value=0.1, max_value=5000.0, value=2000.0, step=0.1)
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
    elif 50 <= h <= 700 and Q >= 50:
        return "Francis"
    elif h < 50 and Q >= 10:
        return "Kaplan"
    elif h < 20 and Q >= 20:
        return "Bulb"
    else:
        return "Francis"  # Default for high head, high flow

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
        - Best for: Medium to high head (50-700m), medium to high flow
        - Efficiency: 90-94%
        - Most common type worldwide (60% of global capacity)
        - Suitable for your parameters: 700m head, 400 m³/s flow
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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

st.header("Turbine Application Ranges")

fig, ax = plt.subplots(figsize=(10, 8))

# -----------------------
# Pelton (red polygon)
# -----------------------
pelton_coords = [
    (0.1, 2000), (0.2, 1000), (0.5, 500), (2, 200), (5, 100), (10, 80),
    (20, 60), (50, 50), (0.1, 50)  # back to base
]
pelton_poly = patches.Polygon(pelton_coords, closed=True, color='red', alpha=0.3, label="Pelton")
ax.add_patch(pelton_poly)

# -----------------------
# Francis (blue polygon)
# -----------------------
francis_coords = [
    (0.5, 700), (5, 700), (20, 600), (50, 500), (100, 400), (200, 300),
    (400, 200), (500, 150), (200, 50), (50, 50), (5, 100), (0.5, 200)
]
francis_poly = patches.Polygon(francis_coords, closed=True, color='blue', alpha=0.3, label="Francis")
ax.add_patch(francis_poly)

# -----------------------
# Kaplan (green polygon)
# -----------------------
kaplan_coords = [
    (5, 50), (20, 40), (50, 30), (200, 20), (1000, 10), (1000, 5),
    (5, 5)
]
kaplan_poly = patches.Polygon(kaplan_coords, closed=True, color='green', alpha=0.3, label="Kaplan")
ax.add_patch(kaplan_poly)

# -----------------------
# Bulb (purple polygon)
# -----------------------
bulb_coords = [
    (20, 20), (100, 20), (500, 15), (1000, 10), (1000, 5), (20, 5)
]
bulb_poly = patches.Polygon(bulb_coords, closed=True, color='purple', alpha=0.3, label="Bulb")
ax.add_patch(bulb_poly)

# -----------------------
# Design Point
# -----------------------
ax.plot(Q, h, 'ko', markersize=10, label='Your Project')

# -----------------------
# Formatting
# -----------------------
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 1000)
ax.set_ylim(5, 2000)

ax.set_xlabel("Discharge Q (m³/s)")
ax.set_ylabel("Head h (m)")
ax.set_title("Turbine Selection Chart")

# Labels inside zones
ax.text(0.2, 800, "Pelton", color="red", fontsize=12)
ax.text(30, 300, "Francis", color="blue", fontsize=12)
ax.text(100, 15, "Kaplan", color="green", fontsize=12)
ax.text(400, 8, "Bulb", color="purple", fontsize=12)

ax.grid(True, which="both", ls="-", alpha=0.3)
ax.legend()

st.pyplot(fig)


# Energy calculation section
st.header("Energy Calculations")
col3, col4 = st.columns(2)

with col3:
    operating_hours = st.number_input("Annual Operating Hours", min_value=100, max_value=8760, value=6000, step=100)
    annual_energy = P * 1000 * operating_hours  # Convert to kWh
    st.metric("Annual Energy Generation", f"{annual_energy/1e6:.2f} MWh")

with col4:
    round_trip_efficiency = st.slider("Round-trip Efficiency (for pumped storage)", 
                                    min_value=0.5, max_value=0.9, value=0.75, step=0.01)
    pumping_energy = annual_energy / round_trip_efficiency
    st.metric("Pumping Energy Required", f"{pumping_energy/1e6:.2f} MWh")

# Francis turbine specific information
st.header("Francis Turbine Details")
st.markdown("""
For your project parameters (700m head, 400 m³/s flow), a Francis turbine is recommended. Here are some key considerations:

**Design Parameters:**
- Runner speed: ~250-400 RPM (depending on specific speed)
- Penstock diameter: ~3-4 meters
- Specific speed (Ns): ~40-80

**Efficiency Considerations:**
- Peak efficiency: 93-95%
- Good efficiency over a wide operating range
- Maintains efficiency down to ~40% of design flow

**Installation Requirements:**
- Vertical shaft configuration
- Spiral casing and stay vanes
- Draft tube for energy recovery
""")

# Additional information
st.header("Additional Information")
with st.expander("Turbine Efficiency Curves"):
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Turbine_efficiency_curves.svg/1200px-Turbine_efficiency_curves.svg.png", 
             caption="Typical turbine efficiency curves (Source: Wikipedia)")

with st.expander("Formulas Used"):
    st.latex(r"P(MW) = Q \times h \times g \times \rho \times \eta / 10^6")
    st.latex(r"E(MWh) = P(MW) \times T(h)")
    st.latex(r"E_{pumping}(MWh) = E_{generation}(MWh) / \alpha")
    st.latex(r"N_s = \frac{N\sqrt{P}}{H^{5/4}} \quad \text{(Specific Speed)}")

# References
st.header("References")
st.markdown("""
- RSE3241 Week 8: Turbine Selection and Energy
- IRENA (2012) Hydroelectric power generation
- IEC 61116:1992, Electromechanical equipment guide for small hydroelectric installations
- Giesecke, J.; Heimerl, S.; Mosonyi, E. Wasserkraftanlagen: Planung, Bau und Betrieb
""")
