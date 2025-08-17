import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import bisect

# Constants
G = 9.81  # Gravity acceleration (m/s²)
PHO = 1000  # Water density (kg/m³)

# Custom styling
st.set_page_config(
    page_title="Penstock Diameter Estimator",
    layout="wide",
    page_icon="💧"
)

st.markdown("""
<style>
    .header {
        background-color: #0e3b5e;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .section {
        background-color: #f0f5ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #0e3b5e;
    }
    .result-box {
        background-color: #e6f0ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #0e3b5e;
    }
    .warning {
        background-color: #fff8e6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #ffcc00;
    }
    .equation {
        font-family: "Times New Roman", Times, serif;
        font-size: 18px;
        text-align: center;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

def friction_factor(Re, d_h, k):
    """Calculate friction factor using Swamee-Jain approximation"""
    if Re < 2000:
        return 64.0 / Re
    rr = k / d_h
    return 0.25 / (math.log10(rr/3.7 + 5.74/Re**0.9))**2

def estimate_penstock_diameter(Q, H_g, L, target_loss_percent=10, 
                               roughness=0.045e-3, nu=1e-6, g=9.81):
    """
    Estimates optimal penstock diameter using iterative hydraulic calculation
    
    Parameters:
    Q : Flow rate (m³/s)
    H_g : Gross head (m)
    L : Penstock length (m)
    target_loss_percent : Target head loss percentage
    roughness : Absolute roughness (m)
    nu : Kinematic viscosity (m²/s)
    g : Gravity acceleration (m/s²)
    
    Returns:
    dict: Results including diameter, head loss, and velocity
    """
    # Convert target loss to absolute value
    target_loss = H_g * target_loss_percent / 100
    
    # Initial diameter estimate (empirical formula)
    D = 0.5 * (Q**2 * L / (g * H_g * target_loss))**0.2
    
    # Iterative solution
    tolerance = 0.001
    max_iterations = 100
    history = []
    
    for i in range(max_iterations):
        # Calculate flow velocity
        A = math.pi * D**2 / 4
        v = Q / A
        
        # Reynolds number
        Re = v * D / nu
        
        # Friction factor (Swamee-Jain)
        f = friction_factor(Re, D, roughness)
        
        # Head loss calculation
        h_f = f * L/D * v**2/(2*g)
        loss_percent = h_f / H_g * 100
        
        # Store iteration history
        history.append({
            "iteration": i+1,
            "diameter": D,
            "velocity": v,
            "reynolds": Re,
            "friction": f,
            "head_loss": h_f,
            "loss_percent": loss_percent
        })
        
        # Check convergence
        if abs(loss_percent - target_loss_percent) < tolerance:
            break
            
        # Adjust diameter (proportional adjustment)
        D *= (loss_percent / target_loss_percent)**0.2
    else:
        st.warning("Diameter estimation did not fully converge within 100 iterations")
    
    return {
        "diameter_m": D,
        "diameter_mm": D * 1000,
        "head_loss_m": h_f,
        "head_loss_percent": loss_percent,
        "velocity_mps": v,
        "iterations": i+1,
        "reynolds": Re,
        "friction_factor": f,
        "history": history
    }

def main():
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.title("Penstock Diameter Estimator")
    st.subheader("Optimal Sizing for Hydroelectric Systems")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning">
    This educational tool estimates penstock diameter based on hydraulic principles.
    Actual designs should be verified by a qualified engineer.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        ### Engineering Principles
        Penstock diameter optimization balances:
        - **Capital cost** (larger diameter = more expensive)
        - **Energy losses** (smaller diameter = more friction loss)
        - **Velocity constraints** (3-6 m/s typical)
        
        The calculation uses:
        - Darcy-Weisbach equation for head loss
        - Swamee-Jain friction factor approximation
        - Iterative solution to meet target head loss
        """)
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Hydroelectric_dam.svg/1200px-Hydroelectric_dam.svg.png", 
                 caption="Hydroelectric System Components")
    
    st.markdown("""
    <div class="section">
    <h2>System Parameters</h2>
    <p>Enter your hydroelectric system characteristics:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Q = st.number_input(
            "Flow Rate (Q) [m³/s]",
            min_value=0.1,
            value=1.0,
            step=0.1,
            help="Volumetric flow rate of water"
        )
        H_g = st.number_input(
            "Gross Head (H_g) [m]",
            min_value=1.0,
            value=200.0,
            step=10.0,
            help="Vertical distance from water source to turbine"
        )
        target_loss = st.slider(
            "Target Head Loss [% of gross head]",
            min_value=1.0,
            max_value=30.0,
            value=15.56,
            step=0.5,
            help="Recommended: 4-15% (15.56% from textbook example)"
        )
        
    with col2:
        L = st.number_input(
            "Penstock Length (L) [m]",
            min_value=1.0,
            value=500.0,
            step=10.0,
            help="Length of the penstock pipe"
        )
        roughness = st.number_input(
            "Roughness Height [mm]",
            min_value=0.001,
            value=0.045,
            step=0.01,
            help="Surface roughness of penstock material"
        )
        material = st.selectbox(
            "Penstock Material",
            ["Steel (smooth)", "Concrete", "HDPE", "Custom"],
            index=0,
            help="Typical roughness values: Steel=0.045mm, Concrete=0.3-3mm"
        )
        
        # Set roughness based on material selection
        if material == "Steel (smooth)":
            roughness = 0.045
        elif material == "Concrete":
            roughness = 1.0
        elif material == "HDPE":
            roughness = 0.0015
    
    with col3:
        nu = st.number_input(
            "Kinematic Viscosity (ν) [m²/s]",
            min_value=1e-7,
            format="%e",
            value=1.000e-06,
            step=1e-7,
            help="Water viscosity at operating temperature"
        )
        temp = st.slider(
            "Water Temperature [°C]",
            min_value=0,
            max_value=40,
            value=15,
            step=1,
            help="Affects water viscosity (0°C: ν=1.79e-6, 20°C: ν=1.00e-6)"
        )
        
        # Update viscosity based on temperature
        if temp == 0:
            nu = 1.79e-6
        elif temp == 10:
            nu = 1.30e-6
        elif temp == 20:
            nu = 1.00e-6
        elif temp == 30:
            nu = 0.80e-6
        elif temp == 40:
            nu = 0.66e-6
    
    if st.button("Calculate Optimal Diameter", type="primary", use_container_width=True):
        # Convert roughness to meters
        roughness_m = roughness / 1000
        
        # Perform calculation
        with st.spinner("Calculating optimal diameter..."):
            results = estimate_penstock_diameter(
                Q=Q,
                H_g=H_g,
                L=L,
                target_loss_percent=target_loss,
                roughness=roughness_m,
                nu=nu
            )
        
        # Display results
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal Diameter", f"{results['diameter_mm']:.1f} mm", f"{results['diameter_m']:.3f} m")
        col2.metric("Head Loss", f"{results['head_loss_percent']:.2f}%", f"{results['head_loss_m']:.2f} m")
        col3.metric("Flow Velocity", f"{results['velocity_mps']:.2f} m/s", 
                   "Good" if 3 <= results['velocity_mps'] <= 6 else "Check")
        
        st.caption("Velocity guide: 3-6 m/s (concrete), 4-8 m/s (steel)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot iteration history
        history = results['history']
        iterations = [h['iteration'] for h in history]
        diameters = [h['diameter_mm'] for h in history]
        losses = [h['loss_percent'] for h in history]
        velocities = [h['velocity'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Diameter convergence
        ax1.plot(iterations, diameters, 'bo-')
        ax1.axhline(y=results['diameter_mm'], color='r', linestyle='--')
        ax1.set_title('Diameter Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Diameter (mm)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Head loss convergence
        ax2.plot(iterations, losses, 'go-')
        ax2.axhline(y=target_loss, color='r', linestyle='--', label='Target')
        ax2.set_title('Head Loss Convergence')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Head Loss (%)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Velocity vs Diameter plot
        diameters_range = np.linspace(results['diameter_m']*0.5, results['diameter_m']*1.5, 50)
        velocities_range = [Q / (math.pi * d**2 / 4) for d in diameters_range]
        
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot([d*1000 for d in diameters_range], velocities_range, 'b-')
        ax.axvline(x=results['diameter_mm'], color='r', linestyle='--', label='Optimal')
        ax.axhline(y=3, color='g', linestyle='-.', label='Min Recommended')
        ax.axhline(y=6, color='g', linestyle='-.', label='Max Recommended')
        ax.set_title('Velocity vs Diameter')
        ax.set_xlabel('Diameter (mm)')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig2)
        
        # Engineering formulas
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Engineering Formulas")
        
        st.markdown("""
        **1. Darcy-Weisbach Head Loss Equation:**
        """)
        st.latex(r'''
        h_f = f \frac{L}{D} \frac{v^2}{2g}
        ''')
        
        st.markdown("""
        **2. Flow Velocity:**
        """)
        st.latex(r'''
        v = \frac{Q}{A} = \frac{4Q}{\pi D^2}
        ''')
        
        st.markdown("""
        **3. Reynolds Number:**
        """)
        st.latex(r'''
        Re = \frac{vD}{\nu}
        ''')
        
        st.markdown("""
        **4. Swamee-Jain Friction Factor (Turbulent Flow):**
        """)
        st.latex(r'''
        f = \frac{0.25}{\left[ \log_{10} \left( \frac{\varepsilon / D}{3.7} + \frac{5.74}{Re^{0.9}} \right) \right]^2}
        ''')
        
        st.markdown("""
        **5. Iterative Solution Process:**
        1. Start with initial diameter estimate
        2. Calculate velocity, Reynolds number, friction factor
        3. Compute head loss
        4. Compare with target head loss
        5. Adjust diameter and repeat until convergence
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Material reference table
        st.markdown("""
        <div class="section">
        <h3>Penstock Material Reference</h3>
        
        | Material | Roughness (mm) | Typical Use | Velocity Range (m/s) |
        |----------|---------------|-------------|----------------------|
        | Steel (smooth) | 0.045 | Large projects | 4-8 |
        | Concrete | 0.3-3.0 | Pressure tunnels | 3-6 |
        | HDPE | 0.0015 | Small projects | 2-4 |
        | Wood stave | 0.5-1.0 | Historical | 2-5 |
        | PVC | 0.0015 | Small installations | 2-4 |
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
