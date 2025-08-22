# turbine_selection_app.py
# Streamlit app for Turbine Selection, Efficiency, and Scenario Comparison

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -------------------------------
# Constants
# -------------------------------
g = 9.81
rho = 1000.0

st.title("🌊 Turbine Selection & Energy Analysis")
st.markdown("Interactive teaching tool for turbine selection, efficiency, and performance evaluation.")

# ---------------------------------------------------
# Session state for scenarios
# ---------------------------------------------------
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

# ---------------------------------------------------
# STEP 1: Define a New Scenario
# ---------------------------------------------------
st.header("1. Define a New Scenario")

col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Scenario Name", value=f"Case {len(st.session_state.scenarios)+1}")
    turbine_choice = st.selectbox("Turbine Type", ["Francis", "Kaplan", "Pelton", "Bulb"])
with col2:
    H_effective = st.number_input("Effective Head Hₑ (m)", value=218.0, step=1.0)
    Q_design = st.number_input("Design Discharge Q (m³/s)", value=240.0, step=10.0)

operating_fraction = st.slider("Operating Flow (Q/Qmax)", 0.1, 1.0, 0.8)
N_rpm = st.number_input("Runner Speed N (rpm)", value=375, step=25)

eta_generator = st.slider("Generator Efficiency (η_gen)", 0.90, 0.99, 0.96)
eta_transformer = st.slider("Transformer Efficiency (η_tr)", 0.98, 0.995, 0.99)

# ---------------------------------------------------
# STEP 2: Efficiency Curves (η vs Q/Qmax)
# ---------------------------------------------------
Q_fraction = np.linspace(0.05, 1.0, 100)
eta_francis = 0.9 - 0.5*(Q_fraction-0.9)**2
eta_kaplan  = 0.92 - 0.2*(Q_fraction-0.8)**2
eta_pelton  = 0.91 - 0.7*(Q_fraction-0.95)**2
eta_bulb    = 0.88 - 0.3*(Q_fraction-0.7)**2

if turbine_choice == "Francis":
    eta_turbine = 0.9 - 0.5*(operating_fraction-0.9)**2
elif turbine_choice == "Kaplan":
    eta_turbine = 0.92 - 0.2*(operating_fraction-0.8)**2
elif turbine_choice == "Pelton":
    eta_turbine = 0.91 - 0.7*(operating_fraction-0.95)**2
else:
    eta_turbine = 0.88 - 0.3*(operating_fraction-0.7)**2

eta_total = eta_turbine * eta_generator * eta_transformer
P = rho * g * Q_design * H_effective * eta_total
P_MW = P / 1e6
nq = N_rpm * (Q_design**0.5) / (H_effective**0.75)

# ---------------------------------------------------
# STEP 3: Add Scenario
# ---------------------------------------------------
if st.button("➕ Add Scenario"):
    st.session_state.scenarios.append({
        "Name": name,
        "Turbine": turbine_choice,
        "Hₑ (m)": H_effective,
        "Q (m³/s)": Q_design,
        "Q/Qmax": operating_fraction,
        "η_turbine": eta_turbine,
        "η_gen": eta_generator,
        "η_tr": eta_transformer,
        "η_total": eta_total,
        "P (MW)": P_MW,
        "nq": nq
    })

# ---------------------------------------------------
# STEP 4: Efficiency Curves Plot
# ---------------------------------------------------
st.header("2. Turbine Efficiency Curves (η–Q/Qmax)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=Q_fraction, y=eta_francis, mode="lines", name="Francis", line=dict(color="red")))
fig.add_trace(go.Scatter(x=Q_fraction, y=eta_kaplan, mode="lines", name="Kaplan", line=dict(color="green")))
fig.add_trace(go.Scatter(x=Q_fraction, y=eta_pelton, mode="lines", name="Pelton", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=Q_fraction, y=eta_bulb, mode="lines", name="Bulb", line=dict(color="orange")))

for sc in st.session_state.scenarios:
    fig.add_trace(go.Scatter(
        x=[sc["Q/Qmax"]],
        y=[sc["η_turbine"]],
        mode="markers+text",
        text=[f"{sc['Name']} ({sc['η_turbine']*100:.1f}%)"],
        textposition="top center",
        marker=dict(size=12, symbol="circle"),
        name=sc["Name"]
    ))

fig.update_layout(
    title="Turbine Efficiency Curves (η vs Q/Qmax)",
    xaxis_title="Flow Rate (Q/Qmax)",
    yaxis_title="Turbine Efficiency η",
    yaxis=dict(range=[0.6, 1.0]),
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# STEP 5: Turbine Selection Map (Q vs H)
# ---------------------------------------------------
st.header("3. Turbine Selection Map (Q vs H)")

fig_map = go.Figure()
# Rough polygons for turbine zones (simplified)
fig_map.add_shape(type="rect", x0=0.1, x1=20, y0=50, y1=2000,
                  fillcolor="orange", opacity=0.2, line=dict(color="orange"), name="Pelton")
fig_map.add_shape(type="rect", x0=0.5, x1=500, y0=20, y1=1000,
                  fillcolor="blue", opacity=0.2, line=dict(color="blue"), name="Francis")
fig_map.add_shape(type="rect", x0=1, x1=1000, y0=5, y1=200,
                  fillcolor="yellow", opacity=0.2, line=dict(color="yellow"), name="Kaplan")
fig_map.add_shape(type="rect", x0=10, x1=2000, y0=2, y1=50,
                  fillcolor="green", opacity=0.2, line=dict(color="green"), name="Bulb")

for sc in st.session_state.scenarios:
    fig_map.add_trace(go.Scatter(
        x=[sc["Q (m³/s)"]],
        y=[sc["Hₑ (m)"]],
        mode="markers+text",
        text=[sc["Name"]],
        textposition="top center",
        marker=dict(size=12, symbol="x"),
        name=sc["Name"]
    ))

fig_map.update_layout(
    xaxis_type="log", yaxis_type="log",
    xaxis_title="Discharge Q (m³/s)", yaxis_title="Head h (m)",
    title="Turbine Selection Chart: Operating Points",
    template="plotly_white"
)
st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------------------------------
# STEP 6: Specific Speed vs Efficiency
# ---------------------------------------------------
st.header("4. Specific Speed vs Efficiency")

Ns_range = np.linspace(5, 120, 100)
eta_curve = 0.75 + 0.2*np.exp(-((Ns_range-40)/20)**2)  # simple bell-shaped curve

fig_ns = go.Figure()
fig_ns.add_trace(go.Scatter(x=Ns_range, y=eta_curve*100, mode="lines", name="Efficiency Curve"))

for sc in st.session_state.scenarios:
    fig_ns.add_trace(go.Scatter(
        x=[sc["nq"]],
        y=[sc["η_turbine"]*100],
        mode="markers+text",
        text=[sc["Name"]],
        textposition="top center",
        marker=dict(size=12, symbol="diamond"),
        name=sc["Name"]
    ))

fig_ns.update_layout(
    title="Specific Speed (Ns) vs Efficiency",
    xaxis_title="Specific Speed Ns",
    yaxis_title="Efficiency (%)",
    template="plotly_white"
)
st.plotly_chart(fig_ns, use_container_width=True)

# ---------------------------------------------------
# STEP 7: Comparison Table
# ---------------------------------------------------
st.header("5. Scenario Comparison Table")

if st.session_state.scenarios:
    df = pd.DataFrame(st.session_state.scenarios)
    st.dataframe(df.style.format({
        "η_turbine": "{:.3f}",
        "η_gen": "{:.3f}",
        "η_tr": "{:.3f}",
        "η_total": "{:.3f}",
        "P (MW)": "{:.1f}",
        "nq": "{:.1f}"
    }))
else:
    st.info("Add scenarios above to see the comparison table.")
