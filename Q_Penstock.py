import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt

# Constants
PHO = 1000  # Density of water (kg/m³)
G = 9.8     # Gravity (m/s²)

def friction_factor(Re, d_h, k):
    """Calculate friction factor using Swamee-Jain approximation"""
    if Re < 2000:
        return 64.0 / Re
    rr = k / d_h
    return 0.25 / (math.log10(rr/3.7 + 5.74/Re**0.9))**2

def calculate_head_loss(L, d_h, Q, k, ν):
    """Calculate head loss using Darcy-Weisbach equation"""
    A = math.pi * d_h**2 / 4
    v = Q / A
    Re = v * d_h / ν
    f = friction_factor(Re, d_h, k)
    return f * L * v**2 / (d_h * 2 * G)

def calculate_power(Q, H_g, eff_turb, eff_gener, head_loss):
    """Calculate electrical power output"""
    return (eff_turb/100 * eff_gener/100 * PHO * G * Q * (H_g - head_loss)) / 1000

def bisection_solver(f, a, c, tol=1e-8, max_iter=100):
    """Bisection method for root finding"""
    for _ in range(max_iter):
        b = (a + c) / 2
        if abs(c - a) < tol:
            return b
        if f(a) * f(b) < 0:
            c = b
        else:
            a = b
    return b

def main():
    st.set_page_config(
        page_title="Optimal Flow and Penstock Diameter",
        layout="wide",
        page_icon="💧"
    )
    
    # Header section
    st.title("Optimal Flow Discharge and Penstock Diameter")
    st.subheader("For Impulse and Reaction Turbines")
    st.write("**Author:** Arturo Leon, Oregon State University")
    st.write("**Email:** arturo.leon@oregonstate.edu")
    st.warning("This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.")
    
    # Create tabs for input and results
    tab1, tab2 = st.tabs(["Input Data", "Results & Visualization"])
    
    # Initialize variables
    flow_discharge = electrical_power = penstock_diameter = None
    results = {}
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Turbine Specifications")
            turbine_type = st.selectbox(
                "Specify the type of turbine",
                ["Impulse Turbine", "Reaction Turbine"]
            )
            
            known_variable = st.selectbox(
                "Enter variable that is known",
                [
                    "Only flow discharge (Optimal)",
                    "Only electrical power (Optimal)",
                    "Penstock Diameter and Flow (No optimal)"
                ]
            )
            
            if known_variable == "Only flow discharge (Optimal)":
                flow_discharge = st.number_input(
                    "Flow discharge [m³/s]",
                    min_value=0.01,
                    value=1.0
                )
            elif known_variable == "Only electrical power (Optimal)":
                electrical_power = st.number_input(
                    "Electrical Power [kW]",
                    min_value=1.0,
                    value=100.0
                )
            else:  # Penstock Diameter and Flow
                flow_discharge = st.number_input(
                    "Flow discharge [m³/s]",
                    min_value=0.01,
                    value=1.0
                )
                penstock_diameter = st.number_input(
                    "Penstock diameter (not optimal) [m]",
                    min_value=0.1,
                    value=1.0
                )
            
            st.header("Penstock Parameters")
            penstock_length = st.number_input(
                "Penstock length [m]",
                min_value=1.0,
                value=500.0
            )
            gross_head = st.number_input(
                "Gross head [m]",
                min_value=1.0,
                value=200.0
            )
            ratio_areas = st.number_input(
                "Ratio of penstock area to nozzle area [no units]" if turbine_type == "Impulse Turbine" 
                else "Ratio of penstock area to draft tube area [no units]",
                min_value=1.0,
                value=16.0
            )
            
        with col2:
            st.header("Loss Parameters")
            nozzle_coeff = st.number_input(
                "Nozzle velocity coefficient [no units]",
                min_value=0.9,
                max_value=1.0,
                value=0.985,
                disabled=(turbine_type == "Reaction Turbine")
            )
            sum_losses = st.number_input(
                "Sum of local losses (entrance, bends, etc.) [no units]",
                min_value=0.0,
                value=1.5
            )
            roughness = st.number_input(
                "Roughness height [mm]",
                min_value=0.001,
                value=0.045
            )
            viscosity = st.number_input(
                "Kinematic Viscosity [m²/s]",
                min_value=1e-7,
                format="%e",
                value=1.000e-06
            )
            
            st.header("Efficiency Parameters")
            turbine_eff = st.number_input(
                "Turbine efficiency (%)",
                min_value=1.0,
                max_value=100.0,
                value=82.0
            )
            generator_eff = st.number_input(
                "Generator efficiency (%)",
                min_value=1.0,
                max_value=100.0,
                value=90.0
            )
    
    # Calculation logic
    with tab2:
        if st.button("Calculate and Plot", type="primary"):
            # Convert inputs
            k_m = roughness / 1000  # mm to m
            eta = turbine_eff/100 * generator_eff/100  # combined efficiency
            
            # Determine turbine coefficients
            if turbine_type == "Impulse Turbine":
                K_N = 1/nozzle_coeff**2 - 1
            else:  # Reaction Turbine
                K_N = 1
            
            try:
                # Mode 1: Only flow discharge (Optimal)
                if known_variable == "Only flow discharge (Optimal)":
                    Q = flow_discharge
                    # Define function for bisection solver
                    const = 14.0/45.0 * G * gross_head / Q**2
                    
                    def f1(D):
                        A = math.pi * D**2 / 4
                        v = Q / A
                        Re = v * D / viscosity
                        f = friction_factor(Re, D, k_m)
                        term = (f * penstock_length/D + sum_losses + K_N * ratio_areas**2) / A**2
                        return term - const
                    
                    # Find optimal diameter
                    D_opt = bisection_solver(f1, 1e-10, 100.0)
                    head_loss = calculate_head_loss(penstock_length, D_opt, Q, k_m, viscosity)
                    power = calculate_power(Q, gross_head, turbine_eff, generator_eff, head_loss)
                    head_loss_percent = head_loss / gross_head * 100
                    
                    results = {
                        "power": power,
                        "head_loss_percent": head_loss_percent,
                        "diameter_mm": D_opt * 1000,
                        "flow_lps": Q * 1000
                    }
                
                # Mode 2: Only electrical power (Optimal)
                elif known_variable == "Only electrical power (Optimal)":
                    P_el = electrical_power * 1000  # Convert to watts
                    
                    # Calculate optimal flow
                    Q = (45/38) * P_el / (eta * PHO * G * gross_head)
                    
                    # Define function for bisection solver
                    const = 14.0/45.0 * G * gross_head / Q**2
                    
                    def f1(D):
                        A = math.pi * D**2 / 4
                        v = Q / A
                        Re = v * D / viscosity
                        f = friction_factor(Re, D, k_m)
                        term = (f * penstock_length/D + sum_losses + K_N * ratio_areas**2) / A**2
                        return term - const
                    
                    # Find optimal diameter
                    D_opt = bisection_solver(f1, 1e-10, 100.0)
                    head_loss = calculate_head_loss(penstock_length, D_opt, Q, k_m, viscosity)
                    power = calculate_power(Q, gross_head, turbine_eff, generator_eff, head_loss)
                    head_loss_percent = head_loss / gross_head * 100
                    
                    results = {
                        "power": power,
                        "head_loss_percent": head_loss_percent,
                        "diameter_mm": D_opt * 1000,
                        "flow_lps": Q * 1000
                    }
                
                # Mode 3: Penstock Diameter and Flow (No optimal)
                else:
                    Q = flow_discharge
                    D = penstock_diameter
                    head_loss = calculate_head_loss(penstock_length, D, Q, k_m, viscosity)
                    power = calculate_power(Q, gross_head, turbine_eff, generator_eff, head_loss)
                    head_loss_percent = head_loss / gross_head * 100
                    
                    results = {
                        "power": power,
                        "head_loss_percent": head_loss_percent,
                        "diameter_mm": D * 1000,
                        "flow_lps": Q * 1000
                    }
                
                # Display results
                st.success("Calculation completed successfully!")
                st.subheader("Results")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Electrical Power [kW]", f"{results['power']:.2f}")
                col2.metric("Head Loss [%]", f"{results['head_loss_percent']:.2f}")
                col3.metric("Diameter [mm]", f"{results['diameter_mm']:.2f}")
                col4.metric("Flow [Liters/s]", f"{results['flow_lps']:.2f}")
                
                # Generate plot
                generate_plot(gross_head, Q, results['power'], 
                              turbine_eff, generator_eff, K_N, 
                              ratio_areas, head_loss)
                
            except Exception as e:
                st.error(f"Calculation error: {str(e)}")

def generate_plot(H_g, Q, P, eff_turb, eff_gener, K_N, AP_AN, head_loss):
    """Generate dimensionless power vs flow plot"""
    eta = eff_turb/100 * eff_gener/100
    ref_diameter = 0.3968  # Reference diameter from example
    ref_area = math.pi * ref_diameter**2 / 4
    A3 = ref_area / AP_AN  # Reference area
    
    # Calculate reference values
    P_ref = (4/3) * PHO * G * H_g * A3 * math.sqrt((1/3)*G*H_g) / 1000
    Q_ref = 2 * A3 * math.sqrt((1/3)*G*H_g)
    
    Qplus_actual = Q / Q_ref
    Pplus_actual = P / P_ref
    
    # Calculate CL and Beta values
    A = math.pi * (ref_diameter)**2 / 4
    CL = (2 * G * H_g * head_loss) / Q**2 * A**2
    Beta = [CL/(2*AP_AN**2), CL/AP_AN**2, 2*CL/AP_AN**2]
    
    # Prepare plot data
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['k', 'b', 'm']
    labels = [f'β = {Beta[0]:.5f}', f'β = {Beta[1]:.5f} (Data)', f'β = {Beta[2]:.5f}']
    
    for i in range(3):
        # Calculate maximum Qplus for this Beta
        Qplus_max = math.sqrt(1.5/(3*Beta[i]))
        Qplus = np.linspace(0, Qplus_max*1.1, 100)
        Pplus = eta * (1.5 * Qplus - Beta[i] * Qplus**3)
        ax.plot(Pplus, Qplus, colors[i], label=labels[i], linewidth=2.5)
    
    # Calculate points for special markers
    Q_eff_0 = math.sqrt(1.5/(3*Beta[0]))
    P_eff_0 = eta * (1.5 * Q_eff_0 - Beta[0] * Q_eff_0**3)
    
    Q_eff_08 = math.sqrt(0.7/(3*Beta[0]))
    P_eff_08 = eta * (1.5 * Q_eff_08 - Beta[0] * Q_eff_08**3)
    
    # Add markers and annotations
    ax.plot(P_eff_0, Q_eff_0, 'sg', markersize=8, label='dP₊/dQ₊ = 0')
    ax.plot(P_eff_08, Q_eff_08, 'sc', markersize=8, label='dP₊/dQ₊ = 0.8η')
    ax.plot(Pplus_actual, Qplus_actual, 'pr', markersize=12, label='Your Data Point')
    
    # Format plot
    ax.set_xlabel('P₊ = P/P₀', fontsize=14)
    ax.set_ylabel('Q₊ = Q/Q₀', fontsize=14)
    ax.set_title('Dimensionless Power vs Flow', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    ax.annotate(f'η = {eta*100:.1f}%', 
                xy=(0.4, 0.1), 
                xycoords='axes fraction',
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()
