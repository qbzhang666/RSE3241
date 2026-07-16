from __future__ import annotations

import html
import io
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


G = 9.81
RHO = 1000.0
APP_DIR = Path(__file__).resolve().parent

# Monash brand palette. Supporting blues are reserved for neutral data series;
# orange and green retain their warning/pass semantics throughout the app.
MONASH_BLUE = "#006DAE"
MONASH_BLACK = "#000000"
MONASH_GREY_1 = "#5A5A5A"
MONASH_GREY_2 = "#969696"
MONASH_GREY_3 = "#E6E6E6"
MONASH_ORANGE = "#F86700"
MONASH_GREEN = "#83A00A"
MONASH_WHITE = "#FFFFFF"
MONASH_HERITAGE_BLUE = "#ABF5F9"
MONASH_ELECTRIC_BLUE = "#285AFF"
MONASH_BLUEBERRY = "#121256"


NEW_PROJECT_PRESET = "New student project"

DISCHARGE_BASIS_POWER = "Size Q from the power target"
DISCHARGE_BASIS_DECLARED = "Use a declared operating Q"
DISCHARGE_BASIS_OPTIONS = [DISCHARGE_BASIS_POWER, DISCHARGE_BASIS_DECLARED]


PRESETS = {
    "Snowy 2.0 - Plateau": {
        "project_name": "Snowy Plateau teaching model",
        "design_power_mw": 2000.0,
        "max_power_mw": 2000.0,
        "units": 6,
        "penstocks": 6,
        "upper_hwl": 1228.7,
        "upper_lwl": 1205.8,
        "lower_hwl": 543.2,
        "lower_twl": 538.8,
        "reservoir_volume_m3": 240_000_000.0,
        "operation_hours": 201.0,
        "sizing_efficiency": 0.90,
        "discharge_basis": DISCHARGE_BASIS_DECLARED,
        "operating_discharge_m3_s": 373.2,
        "teaching_effective_head_m": 606.0,
        "draft_head_m": 72.0,
        "other_head_loss_m": 0.0,
        "eta_turbine": 0.935,
        "eta_generator": 0.97,
        "eta_transformer": 0.99,
        "eta_pump": 0.90,
        "penstock_length_m": 18_000.0,
        "penstock_diameter_m": 10.0,
        "unit_penstock_diameter_m": 2.4,
        "roughness_material": "Custom",
        "roughness_m": 0.00070,
        "flow_area_mode": "Shared conduit",
        "runner_speed_rpm": 428,
        "reservoir_arrangement": "Existing reservoir reuse",
        "valley_geometry": "Existing storage / modified dam",
        "foundation_quality": "Good rock",
        "construction_material": "Existing asset",
        "dam_foundation_rl": 1190.0,
        "dam_crest_length_m": 500.0,
        "dam_crest_width_m": 8.0,
        "freeboard_m": 3.0,
        "wave_allowance_m": 1.0,
        "settlement_allowance_m": 0.5,
        "upstream_slope_hv": 2.5,
        "downstream_slope_hv": 2.0,
    },
    "Snowy 2.0 - Ravine": {
        "project_name": "Snowy Ravine teaching alternative",
        "design_power_mw": 2000.0,
        "max_power_mw": 2000.0,
        "units": 6,
        "penstocks": 6,
        "upper_hwl": 1228.7,
        "upper_lwl": 1205.8,
        "lower_hwl": 543.2,
        "lower_twl": 534.4,
        "reservoir_volume_m3": 240_000_000.0,
        "operation_hours": 201.0,
        "sizing_efficiency": 0.90,
        "discharge_basis": DISCHARGE_BASIS_DECLARED,
        "operating_discharge_m3_s": 373.2,
        "teaching_effective_head_m": 606.0,
        "draft_head_m": 72.0,
        "other_head_loss_m": 0.0,
        "eta_turbine": 0.935,
        "eta_generator": 0.97,
        "eta_transformer": 0.99,
        "eta_pump": 0.90,
        "penstock_length_m": 18_000.0,
        "penstock_diameter_m": 10.0,
        "unit_penstock_diameter_m": 2.4,
        "roughness_material": "Custom",
        "roughness_m": 0.00070,
        "flow_area_mode": "Shared conduit",
        "runner_speed_rpm": 428,
        "reservoir_arrangement": "Existing reservoir reuse",
        "valley_geometry": "Existing storage / modified dam",
        "foundation_quality": "Good rock",
        "construction_material": "Existing asset",
        "dam_foundation_rl": 1190.0,
        "dam_crest_length_m": 500.0,
        "dam_crest_width_m": 8.0,
        "freeboard_m": 3.0,
        "wave_allowance_m": 1.0,
        "settlement_allowance_m": 0.5,
        "upstream_slope_hv": 2.5,
        "downstream_slope_hv": 2.0,
    },
    "Kidston PHES": {
        "project_name": "Kidston-inspired mine-pit teaching model",
        "design_power_mw": 250.0,
        "max_power_mw": 500.0,
        "units": 2,
        "penstocks": 2,
        "upper_hwl": 500.0,
        "upper_lwl": 490.0,
        "lower_hwl": 230.0,
        "lower_twl": 220.0,
        "reservoir_volume_m3": 30_000_000.0,
        "operation_hours": 8.0,
        "sizing_efficiency": 0.80,
        "discharge_basis": DISCHARGE_BASIS_DECLARED,
        "operating_discharge_m3_s": 108.0,
        "teaching_effective_head_m": 270.0,
        "draft_head_m": 12.0,
        "other_head_loss_m": 5.284,
        "eta_turbine": 0.90,
        "eta_generator": 0.97,
        "eta_transformer": 0.99,
        "eta_pump": 0.80,
        "penstock_length_m": 1200.0,
        "penstock_diameter_m": 4.8,
        "unit_penstock_diameter_m": 3.8,
        "roughness_material": "New steel (welded)",
        "roughness_m": 0.000045,
        "flow_area_mode": "Per penstock",
        "runner_speed_rpm": 300,
        "reservoir_arrangement": "Mine pit or quarry reuse",
        "valley_geometry": "Existing storage / modified dam",
        "foundation_quality": "Good rock",
        "construction_material": "Rockfill available",
        "dam_foundation_rl": 455.0,
        "dam_crest_length_m": 700.0,
        "dam_crest_width_m": 8.0,
        "freeboard_m": 2.5,
        "wave_allowance_m": 0.8,
        "settlement_allowance_m": 0.5,
        "upstream_slope_hv": 2.0,
        "downstream_slope_hv": 2.0,
    },
    "Wivenhoe": {
        "project_name": "Wivenhoe teaching model",
        "design_power_mw": 500.0,
        "max_power_mw": 500.0,
        "units": 2,
        "penstocks": 2,
        "upper_hwl": 156.0,
        "upper_lwl": 146.0,
        "lower_hwl": 82.0,
        "lower_twl": 76.0,
        "reservoir_volume_m3": 220_000_000.0,
        "operation_hours": 10.0,
        "sizing_efficiency": 0.80,
        "discharge_basis": DISCHARGE_BASIS_POWER,
        "teaching_effective_head_m": 70.0,
        "draft_head_m": 8.0,
        "other_head_loss_m": 0.0,
        "eta_turbine": 0.88,
        "eta_generator": 0.97,
        "eta_transformer": 0.99,
        "eta_pump": 0.80,
        "penstock_length_m": 800.0,
        "penstock_diameter_m": 6.0,
        "unit_penstock_diameter_m": 8.5,
        "roughness_material": "New steel (welded)",
        "roughness_m": 0.000045,
        "flow_area_mode": "Per penstock",
        "runner_speed_rpm": 150,
        "reservoir_arrangement": "Existing reservoir reuse",
        "valley_geometry": "Existing storage / modified dam",
        "foundation_quality": "Good rock",
        "construction_material": "Existing asset",
        "dam_foundation_rl": 120.0,
        "dam_crest_length_m": 1500.0,
        "dam_crest_width_m": 8.0,
        "freeboard_m": 2.5,
        "wave_allowance_m": 0.8,
        "settlement_allowance_m": 0.5,
        "upstream_slope_hv": 2.5,
        "downstream_slope_hv": 2.0,
    },
}


PRESET_LABELS = {
    NEW_PROJECT_PRESET: NEW_PROJECT_PRESET,
    "Snowy 2.0 - Plateau": "Snowy Plateau teaching model",
    "Snowy 2.0 - Ravine": "Snowy Ravine teaching alternative",
    "Kidston PHES": "Kidston-inspired mine-pit teaching model",
    "Wivenhoe": "Wivenhoe teaching model",
}


PRESET_EVIDENCE_NOTES = {
    "Snowy 2.0 - Plateau": {
        "public_layer": "Snowy 2.0 public reference: 2,200 MW, about 350 GWh, six units and about 27 km of waterways; Snowy Hydro states completion by the end of 2028.",
        "model_layer": "The 2,000 MW, 240 GL, level, route, diameter and 373.2 m3/s values below are course-model assumptions, not the current public project specification.",
        "source": "Snowy Hydro, About Snowy 2.0 (checked 2026-07-11): https://www.snowyhydro.com.au/snowy-20/about/",
    },
    "Snowy 2.0 - Ravine": {
        "public_layer": "Snowy 2.0 public reference: 2,200 MW, about 350 GWh, six units and about 27 km of waterways; Snowy Hydro states completion by the end of 2028.",
        "model_layer": "Ravine is a course-defined comparison alternative. Its geometry and performance must not be presented as the adopted public Snowy 2.0 design without primary evidence.",
        "source": "Snowy Hydro, About Snowy 2.0 (checked 2026-07-11): https://www.snowyhydro.com.au/snowy-20/about/",
    },
    "Kidston PHES": {
        "public_layer": "Kidston public reference: 250 MW, 2,000 MWh, eight-hour storage, two 125 MW units and a published gross-head range of about 181-218 m.",
        "model_layer": "The 270 m first-pass head, 30 GL volume, 1.2 km conduits and 108 m3/s operating point are course-model assumptions and deliberately expose a power-target shortfall.",
        "source": "Genex, 250MW Kidston Pumped Storage Hydro Project (checked 2026-07-11): https://genexpower.com.au/250mw-kidston-pumped-storage-hydro-project/",
    },
    "Wivenhoe": {
        "public_layer": "This preset is retained as a teaching benchmark; current public-project facts have not been embedded in this app release.",
        "model_layer": "Treat every preset value as a course assumption and replace it with a dated primary source before making a project claim.",
        "source": "No current primary project source is attached to this teaching preset.",
    },
}


# Design inputs start empty (None). Nothing calculated is shown until the
# user provides them in Step 1 onwards, or applies a benchmark preset.
DESIGN_NONE_KEYS = [
    "design_power_mw",
    "max_power_mw",
    "operation_hours",
    "upper_hwl",
    "upper_lwl",
    "lower_hwl",
    "lower_twl",
    "reservoir_volume_m3",
    "teaching_effective_head_m",
    "operating_discharge_m3_s",
    "draft_head_m",
    "penstock_length_m",
    "penstock_diameter_m",
    "unit_penstock_diameter_m",
    "dam_foundation_rl",
]


# Non-design UI state: safe structural defaults that carry no project claim.
UI_STATE_DEFAULTS = {
    "project_name": "",
    "active_preset": NEW_PROJECT_PRESET,
    "units": 2,
    "penstocks": 2,
    "sizing_efficiency": 0.85,
    "discharge_basis": DISCHARGE_BASIS_POWER,
    "other_head_loss_m": 0.0,
    "summary_cost_source": "",
    "summary_benefit_source": "",
    "schedule_price_source": "",
    "eta_turbine": 0.90,
    "eta_generator": 0.97,
    "eta_transformer": 0.99,
    "eta_pump": 0.85,
    "temperature_c": 20.0,
    "roughness_material": "Concrete (finished)",
    "roughness_m": 0.00060,
    "flow_area_mode": "Shared conduit",
    "target_velocity": 5.0,
    "runner_speed_rpm": 300,
    "qmax_factor": 1.5,
    "cover_depth_m": 300.0,
    "rock_unit_weight": 26.0,
    "lining_inner_radius_m": 2.5,
    "lining_thickness_m": 0.45,
    "lining_tensile_strength_mpa": 3.0,
    "lining_external_pressure_mpa": 1.0,
    "lining_static_head_m": 100.0,
    "lining_transient_surcharge_m": 0.0,
    "reservoir_arrangement": "New off-river upper reservoir",
    "valley_geometry": "Broad valley",
    "foundation_quality": "Unknown",
    "construction_material": "Rockfill available",
    "dam_crest_length_m": 500.0,
    "dam_crest_width_m": 8.0,
    "freeboard_m": 3.0,
    "wave_allowance_m": 1.0,
    "settlement_allowance_m": 0.5,
    "upstream_slope_hv": 2.5,
    "downstream_slope_hv": 2.0,
}


# "New student project" starts blank: only a name, all design inputs empty.
NEW_PROJECT_DEFAULTS = {
    "project_name": "New PHES project",
}


PRESET_OPTIONS = [NEW_PROJECT_PRESET] + list(PRESETS.keys())


STEP4_HEAD_BASIS_GROSS = "Use Step 3 gross head"
STEP4_HEAD_BASIS_REDUCED = "Use gross head minus first-pass allowance"
STEP4_HEAD_BASIS_MANUAL = "Use preset/manual effective head"
STEP4_HEAD_BASIS_OPTIONS = [
    STEP4_HEAD_BASIS_GROSS,
    STEP4_HEAD_BASIS_REDUCED,
    STEP4_HEAD_BASIS_MANUAL,
]
STEP4_HEAD_BASIS_LEGACY = {
    "Use Step 3 gross head H_g": STEP4_HEAD_BASIS_GROSS,
    "Use preset/manual effective head H_e": STEP4_HEAD_BASIS_MANUAL,
}


ROUGHNESS = {
    "PVC/HDPE": 0.0000015,
    "New steel (welded)": 0.000045,
    "New steel (riveted)": 0.00015,
    "Ductile iron": 0.00026,
    "Concrete (smooth)": 0.00030,
    "Concrete (finished)": 0.00060,
    "Concrete (rough)": 0.00150,
    "Rock tunnel (good lining)": 0.00100,
    "Rock tunnel (rough)": 0.00300,
    "Custom": None,
}


LOSS_COMPONENT_DETAILS = [
    {"component": "Bellmouth entrance", "k": 0.15, "default_count": 1, "default_use": True, "guidance": "Smooth rounded intake entrance."},
    {"component": "Square entrance", "k": 0.50, "default_count": 0, "default_use": False, "guidance": "Sharp-edged entrance; use instead of bellmouth if intake is not rounded."},
    {"component": "Trash rack / screen", "k": 0.20, "default_count": 1, "default_use": True, "guidance": "First-pass allowance for intake screen losses."},
    {"component": "Intake transition", "k": 0.10, "default_count": 1, "default_use": True, "guidance": "Contraction or transition from intake to tunnel."},
    {"component": "45 degree bend", "k": 0.15, "default_count": 0, "default_use": False, "guidance": "Moderate bend in tunnel/penstock alignment."},
    {"component": "90 degree bend", "k": 0.25, "default_count": 2, "default_use": True, "guidance": "Large-radius bend; increase K for tight bends."},
    {"component": "Reducer / expansion", "k": 0.20, "default_count": 1, "default_use": True, "guidance": "Area change at transition, manifold, or valve bay."},
    {"component": "Gate valve", "k": 0.20, "default_count": 1, "default_use": False, "guidance": "Fully open isolation gate or spherical valve allowance."},
    {"component": "Butterfly valve", "k": 0.30, "default_count": 0, "default_use": False, "guidance": "Fully open butterfly valve allowance."},
    {"component": "Branch or tee", "k": 0.40, "default_count": 1, "default_use": True, "guidance": "Bifurcation, manifold, or branch junction."},
    {"component": "Draft tube / outlet transition", "k": 0.30, "default_count": 1, "default_use": True, "guidance": "Transition into draft tube, tailrace, or outlet works."},
    {"component": "Outlet or draft exit", "k": 1.00, "default_count": 1, "default_use": True, "guidance": "Exit loss to reservoir/tailwater where kinetic energy is not recovered."},
    {"component": "Minor fittings allowance", "k": 0.50, "default_count": 1, "default_use": True, "guidance": "Scoping allowance for fittings not yet drawn on the layout."},
]

LOSS_COMPONENTS = {item["component"]: item["k"] for item in LOSS_COMPONENT_DETAILS}


LOSS_COMPONENT_PRESETS = {
    "Single penstock + one turbine": {
        "Bellmouth entrance": 1,
        "Trash rack / screen": 1,
        "Intake transition": 1,
        "90 degree bend": 2,
        "Reducer / expansion": 1,
        "Gate valve": 1,
        "Draft tube / outlet transition": 1,
        "Outlet or draft exit": 1,
    },
    "Shared tunnel + unit branch": {
        "Bellmouth entrance": 1,
        "Trash rack / screen": 1,
        "Intake transition": 1,
        "90 degree bend": 2,
        "Reducer / expansion": 2,
        "Branch or tee": 1,
        "Gate valve": 1,
        "Draft tube / outlet transition": 1,
        "Outlet or draft exit": 1,
        "Minor fittings allowance": 1,
    },
    "Conservative early concept": {
        "Bellmouth entrance": 1,
        "Trash rack / screen": 1,
        "Intake transition": 1,
        "45 degree bend": 2,
        "90 degree bend": 3,
        "Reducer / expansion": 2,
        "Gate valve": 1,
        "Branch or tee": 1,
        "Draft tube / outlet transition": 1,
        "Outlet or draft exit": 1,
        "Minor fittings allowance": 2,
    },
}


EFFICIENCY_GUIDANCE = [
    ["0.45-0.60", "Stress-test or poor early assumption", "Use only to show sensitivity when losses/equipment are highly uncertain."],
    ["0.60-0.75", "Conservative concept sizing", "Useful for old plant, part-load operation, rough head estimate, or broad feasibility checks."],
    ["0.75-0.90", "Typical teaching first pass", "Often reasonable for preliminary generation-mode hydraulic sizing before Step 9 detail."],
    ["0.90-0.96", "Optimistic modern equipment / best point", "Use only if Step 9 efficiency chain and vendor-style operating point support it."],
]


LOSS_LAYOUT_GUIDANCE = [
    ["One penstock and one turbine", "Usually 5-8 components", "Entrance, trash rack, intake transition, one or two bends, valve, reducer/draft tube, outlet exit."],
    ["Shared headrace with branches", "Usually 8-12 components", "Shared intake losses plus bifurcation/branch, valves, reducers, unit transitions, and outlet losses."],
    ["Early concept with no detailed layout", "Use a minor fittings allowance", "Keep the allowance visible and remove it once the fittings are drawn reach by reach."],
]


VELOCITY_GUIDANCE = [
    ["Headrace / low-pressure tunnel", "2-5 m/s", "Lower velocity reduces friction; larger excavation increases civil cost."],
    ["Steel penstock / pressure shaft", "3.5-7 m/s", "Medium-head hydro penstocks are commonly around 3.7-5.5 m/s; high-head designs can justify higher values."],
    ["Intake and trash rack approach", "0.5-1.5 m/s", "Lower velocities reduce debris, vortices, and fish/ecology risk."],
    ["Tailrace tunnel", "2-5 m/s", "Avoid high velocities that drive tailrace losses and transient drawdown."],
]


HEAD_LOSS_ALLOWANCE_GUIDANCE = [
    ["Strict / efficient waterway", "about 1-3% of gross head", "Use when energy efficiency is important, the waterway is long, or students want a conservative low-loss design."],
    ["Typical first screening", "about 3-5% of gross head", "Good starting point for Step 5 before cost optimisation; then check whether the required diameter is realistic."],
    ["Broad concept upper screen", "about 5-10% of gross head", "Use only if excavation/penstock cost is likely to dominate and the lost head is acceptable in Step 4/9."],
    ["Usually revise", "above 10% of gross head", "Losses are large enough to materially change net head, discharge, turbine selection, and economics."],
]


BRANCH_SELECTION_GUIDANCE = [
    ["Units", "Start from plant rating divided into practical reversible pump-turbine units.", "More units reduce unit flow and may move the turbine point back into a chart zone, but add complexity and cost."],
    ["Penstocks", "Teaching default: one high-pressure branch per unit, or a shared headrace that bifurcates near the powerhouse.", "Use separate penstocks where isolation, transients, or constructability justify them."],
    ["Intake/outlet openings", "Usually one per penstock or one intake tower with separate gated passages.", "Each opening needs approach velocity, trash rack, gate, vortex/submergence, and local-loss checks."],
    ["Branch local K", "Use 1-3 for a smooth branch/valve/transition group at concept stage.", "Increase if the branch has tight bends, abrupt transitions, or poor manifolding."],
]


STEP6_NUMERIC_INPUT_GUIDANCE = [
    ["Unit branch length", "Measure the individual branch centreline from the bifurcation/header or manifold to the turbine inlet/spiral case. If no branch is drawn, start with about 5-20% of the Step 5 pressure-waterway length, commonly 100-500 m for a concept layout.", "Replace with the Step 4 long-section chainage once the branch is drawn."],
    ["Unit branch local-loss coefficient", "Sum the local losses in the unit branch only: branch/tee, bends, valve, reducer/transition, and turbine inlet transition. Smooth concept branches are often about 1-3; complex branches can be 3-6 or higher.", "Do not double count losses already included in Step 5 shared-conduit K."],
    ["Wave speed", "Select from the lining/material table. Use 800-1000 m/s for an unknown concrete/steel concept, 900-1250 m/s for steel, and test low/high sensitivity before final transient modelling.", "This is for the Joukowsky sense check only; Step 8 should revisit it with closure time and surge protection."],
]


UNIT_BRANCH_K_GUIDANCE = [
    ["Very smooth short branch", "0.5-1.5", "One gradual bifurcation/transition and no tight bends; rare unless the layout is very clean."],
    ["Typical concept unit branch", "1-3", "One branch/tee, valve, reducer/transition, and one or two large-radius bends."],
    ["Complex or compact powerhouse manifold", "3-6", "Several bends, abrupt transitions, closely spaced valves, or constrained geometry near the powerhouse."],
    ["Uncertain layout", "Use 2 first, then test 1 and 4", "Report as sensitivity until the branch arrangement is drawn."],
]


WAVE_SPEED_GUIDANCE = [
    ["Steel penstock", "900-1250 m/s", "Stiff steel and thick walls give high wave speeds and larger Joukowsky pressure rise."],
    ["Concrete-lined tunnel", "800-1200 m/s", "Use a middle value until lining, rock restraint, and joints are known."],
    ["Unlined rock tunnel", "1000-1400 m/s", "Can be high, but leakage/joint compliance and air content matter."],
    ["GRP / PVC / PE", "200-700 m/s", "More flexible pipes have lower wave speed but may have lower pressure rating."],
    ["Unknown concept", "800-1000 m/s", "Use as a sensitivity input, then replace with a material/lining calculation."],
]


RESERVOIR_ARRANGEMENTS = [
    "Existing reservoir reuse",
    "One existing and one new reservoir",
    "New off-river upper reservoir",
    "New lower reservoir",
    "Closed-loop PHES",
    "Open-loop PHES",
    "Mine pit or quarry reuse",
    "Ring-dike / turkey-nest reservoir",
]


VALLEY_GEOMETRIES = [
    "Existing storage / modified dam",
    "Broad valley",
    "Narrow valley",
    "Narrow gorge",
    "Plateau or ridge top",
    "Mine pit or quarry",
]


FOUNDATION_QUALITIES = [
    "Very strong rock",
    "Good rock",
    "Fair rock",
    "Soil / weathered foundation",
    "Unknown",
]


CONSTRUCTION_MATERIALS = [
    "Existing asset",
    "Rockfill available",
    "Clay / earthfill available",
    "RCC / concrete aggregate available",
    "Limited local materials",
]


DAM_TYPE_GUIDANCE = {
    "Existing reservoir/dam reuse": "Use this for Snowy 2.0-style teaching cases. Identify operating constraints, dam safety limits, intake/outlet modifications, and environmental approvals rather than redesigning the dam.",
    "Earthfill embankment": "Best suited to broad valleys and low-to-moderate dam heights where clay, filters, drains, and earthfill materials are locally available.",
    "Rockfill embankment": "A common PHES screening choice for broad valleys, moderate-to-high embankments, and projects with strong rockfill supply from excavation.",
    "Concrete faced rockfill dam (CFRD)": "Useful where a rockfill embankment is attractive but seepage control needs a low-permeability upstream face.",
    "RCC / concrete gravity dam": "Suited to narrower valleys with competent rock foundation and reliable concrete aggregate supply.",
    "Arch dam": "Only plausible for a narrow gorge with very strong abutments. Efficient in concrete volume but demanding in geology.",
    "Ring-dike / turkey-nest reservoir": "Relevant for off-river PHES on plateau or ridge-top terrain, where storage is formed by a perimeter embankment.",
    "Further site investigation required": "The inputs do not support a confident screening choice. Ask students to identify the missing survey, geology, hydrology, or material data.",
}


DAM_SELECTION_CRITERIA = [
    ["Reservoir arrangement", "Decides whether the app is selecting a new dam type or recognising an existing storage/reuse concept.", "Existing reservoir or mine-pit reuse normally means dam-safety review and intake/outlet modification, not a new dam family."],
    ["Valley or storage geometry", "Controls whether an embankment, gravity/RCC, arch, ring-dike or reuse concept is physically plausible.", "Broad valleys favour embankments; narrow competent valleys favour concrete options; plateau/ridge sites can favour ring-dikes."],
    ["Foundation and abutment quality", "Controls sliding, seepage, settlement, arch action and dam-safety risk.", "Unknown foundation blocks a confident new-dam selection; very strong rock is required for arch dams."],
    ["Available construction material", "Controls whether earthfill, rockfill/CFRD or RCC is a practical first-pass option.", "Local rockfill favours rockfill/CFRD; clay/earthfill favours earthfill; concrete aggregate favours RCC/gravity."],
    ["Dam height", "Separates low/moderate embankment concepts from high-head, high-consequence structures that need stronger verification.", "At about 60 m or higher, the app becomes cautious about small/simple embankment choices."],
    ["Active storage volume", "Large storage usually increases consequence class, embankment quantity, seepage importance and outlet/safety requirements.", "Large concept storage pushes the app toward robust rockfill/CFRD-style screening unless other criteria govern first."],
]


EVIDENCE_SOURCE_LINKS = [
    ["USACE hydropower plant structures manual", "https://www.publications.usace.army.mil/portals/76/publications/engineermanuals/em_1110-2-3001.pdf", "Official planning/design reference for hydropower structures, including penstocks, intakes, surge tanks, and powerhouse layout."],
    ["USBR dam design manuals and monographs", "https://www.usbr.gov/tsc/techreferences/hydraulics_lab/pubs/manuals_monographs.html", "Official portal to Design of Small Dams, Gravity Dams, Arch Dams, and related dam-design references."],
    ["FERC engineering guidelines: water conveyance", "https://www.ferc.gov/sites/default/files/2020-04/chap12.pdf", "Inspection and monitoring context for hydropower water-conveyance safety; useful for risk and evidence discussions."],
    ["Global closed-loop PHES atlas study", "https://www.sciencedirect.com/science/article/pii/S2542435120305596", "Peer-reviewed DEM/GIS resource assessment for off-river closed-loop PHES."],
    ["GIS-AHP PHES site selection, Northern Queensland", "https://www.sciencedirect.com/science/article/pii/S0306261923002787", "Peer-reviewed Australian example combining environmental and technical GIS criteria with AHP and LCOE scenarios."],
    ["GIS-MCDA PHES placement study", "https://link.springer.com/article/10.1007/s13201-025-02524-z", "Open-access MCDA example using head, slope, head-to-distance, TWI, lineaments, grid, roads, and rivers."],
    ["NLR pumped-storage supply-curve geospatial data", "https://www.nlr.gov/gis/psh-supply-curves", "Practice dataset fields include storage duration, paired volume, capacity, head, reservoir distance, transmission spurline, and cost."],
    ["GHD site-selection practice note", "https://www.ghd.com/en-ph/insights/good-site-selection---the-cornerstone-of-successful-pumped-hydro", "Industry perspective on topography, geology, access, grid integration, environmental sustainability, and social benefit."],
    ["ANU RE100 pumped hydro atlases", "https://re100.eng.anu.edu.au/pumped_hydro_atlas/", "Useful Australian/global reference atlas; use as one source among several, not as the app method."],
]


OPEN_SOURCE_GIS_LINKS = [
    ["QGIS", "https://github.com/qgis/QGIS", "Desktop GIS platform for map preparation, vector/raster processing, layouts, and exports."],
    ["QGIS hydrological analysis guide", "https://docs.qgis.org/latest/en/docs/training_manual/processing/hydro.html", "Official tutorial for DEM-based channel network, watershed, and terrain-statistics workflow."],
    ["GRASS GIS", "https://github.com/OSGeo/grass", "Open-source geoprocessing engine; useful for contours, hydrology, terrain analysis, and scripted batch checks."],
    ["QGIS GRASS integration", "https://docs.qgis.org/latest/en/docs/user_manual/grass_integration/grass_integration.html", "Official QGIS documentation for using GRASS modules through QGIS."],
    ["WhiteboxTools", "https://github.com/jblindsay/whitebox-tools", "Open-source terrain/hydrology backend with flow accumulation, watershed, stream network, sink removal, and terrain-index tools."],
    ["ANU PHES_Searching", "https://github.com/ANU-RE100/PHES_Searching", "DEM-based PHES search code. GPL-3.0 and C++/GDAL-heavy; adopt only after licence and dependency review."],
]


RESERVOIR_SOURCE_CATEGORIES = [
    "Greenfield",
    "Bluefield",
    "Brownfield",
    "Ocean",
    "Seasonal",
    "Turkey's Nest",
    "Custom / field-mapped",
]


STORAGE_DURATION_CLASSES = [
    "2 GWh / 6 h",
    "5 GWh / 18 h",
    "15 GWh / 18 h",
    "50 GWh / 18-50 h",
    "150 GWh / 50-168 h",
    "500 GWh / 50-168 h",
    "1500 GWh / 60-168 h",
    "5000 GWh / 200 h",
    "Custom",
]


SCREENING_COST_CLASSES = ["AAA", "AA", "A", "B", "C", "D", "E", "Unknown"]


SCREENING_COST_CLASS_GUIDANCE = [
    ["AAA", "Most favourable desktop cost class", "Treat as a priority to investigate, not as proof of feasibility."],
    ["AA", "Very favourable desktop cost class", "Check whether the short waterway, high head, or high storage ratio survives local constraints."],
    ["A", "Favourable desktop cost class", "Good concept-stage ranking; still needs project-specific quantities and exclusions."],
    ["B", "Moderate desktop cost class", "Compare against nearby alternatives before committing to detailed work."],
    ["C", "Higher desktop cost class", "Civil works, route length, or storage geometry may be weakening the option."],
    ["D", "Poor desktop cost class", "Usually only worth continuing if grid value or reuse opportunity is unusually strong."],
    ["E", "Least favourable desktop cost class", "Use mainly as a reject/benchmark case unless there is compelling strategic value."],
]


SCREENING_CRITERIA = [
    ["Topographic head", "GIS-MCDA and industry practice consistently treat elevation difference as a primary technical/economic driver.", "Carry upper/lower levels into Step 3 head and Step 4 discharge."],
    ["Head-to-distance ratio / slope", "Published MCDA examples rank head-to-distance and slope highly because waterways, losses, access, and cost depend on route length.", "Calculate head divided by separation and compare alternatives before selecting a corridor."],
    ["Storage volume and duration", "Supply-curve datasets and PHES studies report paired volume, storage duration, capacity, and energy as core fields.", "Check that active storage can support the project MW and target hours."],
    ["Reservoir/source category", "Greenfield, existing-reservoir, mine/quarry, coastal, and ring-dike concepts have different approval and civil assumptions.", "Select the concept family before dam type, water source, and environmental risk are discussed."],
    ["Civil efficiency", "Dam height, dam length, fill/rock volume, and water-to-rock ratio are practical proxies for civil quantity and cost exposure.", "Use water-to-rock and dam-volume checks as screening metrics, not final quantities."],
    ["Geology and lineaments", "Scientific MCDA studies and practice guidance include faults, lineaments, rock mass, and foundation conditions as feasibility controls.", "Flag weak rock, fault crossings, cavern risk, hydraulic jacking risk, and dam foundation uncertainty."],
    ["Grid, road, and construction access", "Transmission distance, road access, portals, spoil, and constructability often decide whether technically attractive sites can proceed.", "Record distances and access assumptions; move detailed checks to Steps 4, 7, and 10."],
    ["Environmental, heritage, and land constraints", "Closed-loop and off-river screening still excludes or downgrades protected, urban, culturally sensitive, and high-impact areas.", "Use constraint layers early; a fatal constraint can override strong head or cost metrics."],
    ["Data confidence", "DEM resolution, vertical datum, contour interval, storage curve quality, and field validation determine whether a desktop site is credible.", "Attach source/date/confidence notes to every mapped candidate."],
    ["Relative cost class", "Desktop cost ranks are useful for sorting alternatives, but practice requires project-specific quantities, rates, risk allowance, and contingency.", "Use AAA-E or similar ranks only as a qualitative benchmark until a cost model is built."],
]


QGIS_SCREENING_WORKFLOW = [
    ["1", "Set project CRS and study boundary", "Use a projected CRS in metres; clip DEM and vectors to a buffered search area.", "Boundary map, CRS, vertical datum, DEM source and resolution."],
    ["2", "Build terrain base layers", "Generate hillshade, slope, contours, local relief, and candidate elevation bands from the DEM.", "Contour/hillshade map and candidate upper/lower elevation zones."],
    ["3", "Prepare hydrology and exclusions", "Run sink filling/flow accumulation where useful, then overlay waterways, protected areas, urban areas, heritage, land tenure, and high-impact exclusions.", "Constraint map with hard exclusions and soft risk flags."],
    ["4", "Map candidate reservoirs", "Digitise potential upper/lower footprints or use contour polygons; estimate area, active depth, and usable volume.", "Reservoir-pair table with area, volume, elevation, and confidence."],
    ["5", "Measure pair geometry", "Use shortest feasible alignment, not just straight-line distance; record head, separation, slope, and head-to-distance ratio.", "Candidate longlist with head, route length, and head-to-distance."],
    ["6", "Screen access and grid", "Measure road, portal, construction access, substation/transmission distance, and likely corridor conflicts.", "Access/grid map and distance table."],
    ["7", "Rank alternatives", "Apply transparent weights or a simple pass/review/reject matrix for topography, storage, civil quantity, grid, access, environment, geology, and data quality.", "Shortlist with reasoned ranking and sensitivity notes."],
    ["8", "Export evidence to this app", "Export map layouts and CSV/GeoPackage attributes; transfer selected levels, volume, route length, cost rank, constraints, and confidence.", "Inputs for Steps 2-4 plus appendable report figures."],
]


QGIS_TOOL_COOKBOOK = [
    ["Project setup", "QGIS Browser; Reproject layer; Warp (reproject); Clip raster by mask layer; Clip", "DEM, boundary polygon, vector constraints", "Common CRS project, clipped DEM, clipped vector layers", "Check CRS units are metres before measuring head-to-distance or area."],
    ["Terrain view", "Hillshade; Slope; Aspect; Contour", "DEM", "Hillshade, slope raster, contour lines", "Use hillshade plus 10-50 m contours to teach why steep short routes matter."],
    ["Hydrology check", "GRASS r.watershed; GRASS r.fill.dir; Whitebox Fill Depressions; Whitebox D8 Flow Accumulation; Channel Network", "DEM", "Flow accumulation, indicative drainage lines, catchment hints", "Use for environmental/waterway awareness, not as a substitute for hydrology design."],
    ["Candidate elevation bands", "Raster calculator; Reclassify by table; Polygonize; Dissolve", "DEM, target upper/lower elevation ranges", "Upper/lower candidate polygons", "A simple undergraduate exercise: isolate land between two contour levels and polygonise it."],
    ["Reservoir footprint and storage", "Digitising tools; Field Calculator; Raster surface volume; Zonal statistics", "Candidate polygon, DEM, target HWL/LWL", "Area, approximate active volume, elevation statistics", "Use the app's Step 3 to recompute energy after QGIS estimates the active volume."],
    ["Constraint overlay", "Buffer; Difference; Intersection; Extract by location; Select by location", "Protected areas, waterways, heritage, urban/land-use, candidate polygons", "Pass/review/reject flags and conflict area", "Good lab task: compare the same candidate before and after exclusions."],
    ["Road/access data", "QuickOSM; OpenStreetMap basemap; Clip; Join attributes by nearest", "Study boundary or map extent", "Road/access layer and nearest-road distance", "QuickOSM can fetch highway and track features quickly for tutorial demos."],
    ["Transmission/grid proxy", "QuickOSM; Distance to nearest hub; Join attributes by nearest", "Power lines/substations from OSM or official network data", "Distance to nearest line/substation; grid evidence note", "Use official network data where available; OSM is a teaching proxy."],
    ["Waterway corridor", "Shortest path (point to point); Shortest path (point to layer); GRASS r.cost; Least-cost path tools", "Candidate intake/outlet points, roads/cost surface/avoidance areas", "Indicative route length and corridor conflicts", "Compare straight-line distance with feasible route length."],
    ["Alternative ranking", "Field Calculator; Refactor fields; Join attributes by location; Basic statistics", "Candidate table with head, length, volume, constraints", "Ranked shortlist and summary table", "Keep ranking transparent: weights and assumptions must be visible."],
    ["Repeatable model", "QGIS Model Designer; Batch Processing", "DEM, boundary, constraints, candidate points/polygons", "Reusable screening model for tutorials", "Turn the tutorial into a one-click model after students understand each step."],
    ["Evidence export", "Layout Manager; Export map; Save features as GeoPackage/CSV", "Final candidate layers and maps", "PNG/PDF map figures, CSV candidate table", "Export figures and CSV for upload or manual transfer into this Streamlit app."],
]


QGIS_TUTORIAL_DEMOS = [
    ["Demo A: terrain-only pair screen", "DEM plus a rectangular study boundary", "Hillshade -> Contour -> Raster calculator elevation bands -> Polygonize -> Field Calculator area", "Two candidate reservoir polygons, estimated area, upper/lower elevation, head", "Use in Week 1-2 before discussing environmental constraints."],
    ["Demo B: constraint overlay", "Candidate polygons plus waterways, protected area, land-use, and heritage layers", "Buffer waterways -> Intersect candidates with exclusions -> Difference to create remaining feasible area", "Pass/review/reject flag and lost reservoir area", "Shows why a high-head site can still be rejected."],
    ["Demo C: head-to-distance comparison", "Two or three candidate upper/lower pairs", "Create intake/outlet points -> Shortest path or least-cost route -> Field Calculator head/length", "Alternative table ranked by head, route length, and head-to-distance", "Directly populates Step 2 and Step 4 route-length inputs."],
    ["Demo D: Kidston-style mine/quarry reuse", "Satellite imagery/OSM context plus a digitised pit polygon", "Digitise pit/reservoir footprint -> Field Calculator area -> assume active depth -> estimate volume", "Reuse concept with area, active volume, source category, confidence note", "Links the map exercise to the Kidston preset already in the app."],
    ["Demo E: Snowy-style existing reservoir pair", "Existing reservoir polygons, route sketch, protected-area layer", "Measure reservoir levels from sources -> draw route corridor -> overlay constraints -> export plan map", "Existing-storage reuse evidence and route/conflict map", "Links qualitative siting risk to the Snowy Plateau/Ravine case-study tab."],
    ["Demo F: one-click model builder", "Any completed Demo A-C layers", "Rebuild the steps in QGIS Model Designer with DEM, boundary, constraints, and candidate points as inputs", "Reusable model plus a CSV output template", "Assesses repeatability and scientific traceability rather than one-off map drawing."],
]


QGIS_CANDIDATE_TEMPLATE_COLUMNS = [
    "option",
    "source_category",
    "latitude",
    "longitude",
    "upper_elevation_m",
    "lower_elevation_m",
    "head_m",
    "route_length_km",
    "storage_gwh",
    "usable_volume_gl",
    "reservoir_area_km2",
    "dam_height_m",
    "dam_length_m",
    "dam_fill_volume_million_m3",
    "water_to_rock_ratio",
    "desktop_cost_rank",
    "constraint_flag",
    "data_confidence",
    "source_notes",
]


QGIS_STARTER_DATA_ITEMS = [
    ["DEM", "Elevation raster loaded and clipped to the study area"],
    ["Boundary", "Study boundary drawn or imported"],
    ["Contours / hillshade", "Terrain layers created from the DEM"],
    ["Reservoir sketch", "At least one upper and one lower footprint or point marked"],
    ["Constraints", "Waterways, protected areas, land use, heritage, or other exclusions added"],
    ["Access / grid", "Road and transmission/substation proxy layers added"],
]


CONSTRAINT_RESULT_GUIDE = [
    [
        "No obvious fatal flag",
        "Use only after the candidate reservoir footprints, dam axis, waterway corridor, powerhouse area, access route and grid path avoid mapped hard exclusions at desktop scale.",
        "Protected-area and heritage layers do not intersect the works; land access looks plausible; waterway/ecology conflicts are minor or avoidable; at least one route remains.",
        "Continue to Step 2, but keep normal approvals, survey, geology and water-rights checks in the risk register.",
    ],
    [
        "Review required",
        "Use when the option may still work, but a mapped layer or missing dataset could materially change the layout.",
        "Buffer overlap with a waterway or sensitive habitat; uncertain land tenure or cultural heritage status; route crosses difficult access/geology; DEM or storage data are low confidence.",
        "Keep the candidate as conditional. Add a note describing the missing evidence and compare at least one alternative route or reservoir pair.",
    ],
    [
        "Possible fatal constraint",
        "Use when a hard exclusion or unavoidable conflict could stop the scheme, even if head and storage look good.",
        "Works are inside a no-go protected/heritage area; no feasible access or grid path; dam/powerhouse footprint is legally or physically blocked; known geology or water-rights issue cannot be bypassed.",
        "Do not carry the option forward as selected. Reroute, change reservoir pair, or document it as rejected evidence.",
    ],
]


NEW_PROJECT_CONSTRAINT_CHECKS = [
    ["1", "Collect layers", "DEM/contours, reservoirs or candidate basins, waterways, protected areas, heritage, land tenure, roads, transmission/substations, geology if available."],
    ["2", "Draw the project footprint", "Upper/lower reservoir footprint, dam axis, waterway corridor, powerhouse/cavern area, portal/access route and grid path."],
    ["3", "Overlay constraints", "Use QGIS select/intersect/buffer tools to identify direct intersections and near misses. Record affected area, route length or crossing count where possible."],
    ["4", "Classify the flag", "No obvious fatal flag if no hard exclusion is mapped; Review required if evidence is incomplete or conflicts look manageable; Possible fatal constraint if an unavoidable no-go conflict appears."],
    ["5", "Carry evidence forward", "Attach the map/table in Step 2. If the result is review or fatal, record the mitigation, alternative, or rejection reason in Step 10 risk/recommendation."],
]


STEP2_EVIDENCE_ITEMS = [
    ["Reservoir footprints", "Upper and lower polygons or points are digitised with area and elevation notes.", "Step 3 levels and storage"],
    ["Operating levels", "Upper/lower HWL and LWL plus representative levels come from storage curves, operating rules or a clearly labelled benchmark assumption.", "Step 3 representative head and simultaneous head envelope"],
    ["Storage estimate", "Active volume or area-depth estimate is recorded with confidence.", "Step 3 energy and duration"],
    ["Dam axis", "Candidate crest line, abutments, foundation RL, and crest length are mapped.", "Step 3 dam sizing"],
    ["Waterway corridor", "A feasible route, not only a straight line, has length and conflict notes.", "Steps 4-6 losses"],
    ["Powerhouse location", "Surface or underground location is marked with access and geology assumptions.", "Step 7 cavern checks"],
    ["Constraint screen", "Protected areas, waterways, heritage, land tenure, roads, and grid access are checked.", "Step 10 risk"],
    ["Alternative comparison", "At least one rejected or backup option is documented with the reason.", "Step 10 recommendation"],
]


DAM_CONCEPT_DEFINITIONS = [
    ["Earthfill embankment", "Compacted earth with filters, drains, and often a clay core.", "Broad valleys, moderate heights, local clay/earthfill.", "Slope stability, seepage, filters, settlement, erosion protection."],
    ["Rockfill embankment", "Rockfill shell with impervious core or upstream sealing system.", "Large volumes of sound excavated rock; common PHES concept option.", "Rockfill quality, settlement, seepage, face/core detailing."],
    ["Concrete faced rockfill dam (CFRD)", "Rockfill embankment sealed by an upstream concrete face slab.", "Rockfill available and seepage control is important.", "Face slab joints, plinth/foundation treatment, deformation."],
    ["RCC / gravity dam", "Concrete mass resists water load mainly by its own weight.", "Narrower valleys and competent rock foundations.", "Sliding, overturning, uplift, foundation shear, thermal cracking."],
    ["Arch dam", "Curved concrete dam transfers load to strong abutments.", "Narrow gorge with very strong rock abutments.", "Abutment stability, arch stresses, foundation deformation."],
    ["Ring-dike / turkey-nest", "Perimeter embankment creates storage on a ridge or plateau.", "Off-river upper reservoir on relatively flat high ground.", "Long embankment length, seepage, settlement, spillway/freeboard."],
    ["Existing dam/reservoir reuse", "Use or modify an existing storage instead of designing a new dam.", "Brownfield projects such as existing hydro reservoirs or mine pits.", "Dam safety limits, intake/outlet works, approvals, operating constraints."],
]


DAM_DIMENSION_INPUT_GUIDANCE = [
    ["Foundation RL", "Use the lowest competent foundation elevation along the dam axis, not just the water edge.", "From contours, geology map, boreholes, or a clearly stated assumption."],
    ["Freeboard", "Concept allowance above HWL for wind setup, wave run-up, flood routing uncertainty, and operation.", "Use a conservative teaching value, then replace with hydrology/wave design."],
    ["Crest length", "Measure along the mapped dam axis between abutments or around a ring-dike.", "This drives fill/concrete volume more than students often expect."],
    ["Crest width", "Allow access, construction, safety, and operation; road access usually requires a wider crest.", "Keep as a concept proxy until dam type and access class are known."],
    ["Upstream/downstream slopes", "Use steeper concrete faces and flatter embankment slopes; final values require geotechnical design.", "Slope choices directly affect fill volume and stability."],
    ["Active volume", "Use a storage-elevation curve where possible; area x active depth is a first teaching estimate.", "Loop back to Step 2 if the mapped footprint cannot support the target GWh."],
]


DRAFT_HEAD_GUIDANCE = [
    ["No turbine-setting evidence yet", "Use 0 m only as a temporary geometry placeholder, then revisit after Step 9 cavitation/submergence checks.", "Do not subtract this value from gross or selected hydraulic head."],
    ["Known turbine centreline", "Use tailwater level minus turbine centreline when the reversible pump-turbine centreline is set below tailwater.", "This locates the machine and draft tube; it is not an energy loss."],
    ["Benchmark or vendor basis", "Use the setting from a cited case study, vendor pre-selection, cavitation check, or previous design stage.", "Record the tailwater condition, datum and whether the value is minimum submergence or a physical centreline offset."],
    ["Cavitation sensitivity", "Test turbine settings against simultaneous minimum tailwater and the pump-mode operating envelope.", "Carry the result into powerhouse geometry and cavitation risk, not into Darcy/minor head loss."],
]


DAM_MODELLING_VERIFICATION = [
    ["2D slope-stability / limit equilibrium", "Embankment upstream and downstream slopes; rapid drawdown and steady seepage.", "Factor of safety for circular/non-circular slips and sensitivity to material strength."],
    ["Seepage model", "Earthfill, rockfill, CFRD plinth, abutments, and foundation.", "Pore pressures, uplift, exit gradients, filter/drain requirements, leakage paths."],
    ["2D/3D finite-element model", "High dams, complex foundations, RCC/gravity dams, or unusual geometry.", "Stress, deformation, settlement, concrete cracking risk, foundation interaction."],
    ["Sliding/overturning/uplift checks", "Gravity, RCC, spillway blocks, intake structures, and concrete sections.", "Load combinations for normal, flood, seismic, and construction stages."],
    ["Dynamic/seismic model", "Projects in seismic areas or with high consequence classification.", "Deformation, liquefaction screening, post-earthquake stability, freeboard adequacy."],
    ["Independent dam-safety review", "Any concept progressing beyond teaching/design-screening level.", "Design basis, hydrology, geology, materials, monitoring, and emergency planning."],
]


CAVERN_DIMENSION_GUIDANCE = [
    ["Machine hall width", "Start from turbine-generator envelope plus crane, erection, access, and rock support space.", "Screening values often sit around 20-35 m for large underground hydro caverns; verify with equipment layout."],
    ["Unit bay width", "Use one bay per unit plus service clearances.", "A teaching range of 10-25 m/unit is reasonable before vendor drawings."],
    ["Erection bay", "Add a bay for runner/generator maintenance, assembly, and laydown.", "Often one to two unit-bay widths at concept level."],
    ["Hall height", "Depends on machine setting, generator, draft tube, crane lift, and cavern roof arch.", "The app shape ratios are only a first cut; poor rock usually increases support/pillar needs, not equipment height."],
    ["Pillar to transformer hall", "Rock bridge between caverns and galleries.", "Good rock may allow smaller pillars; fair/poor rock needs larger pillars, stress modelling, and more support."],
    ["Cover depth", "Vertical rock cover above the crown.", "Check cover against span, in-situ stress, hydraulic jacking, and topographic confinement."],
]


STEP8_INPUT_GUIDANCE = [
    ["Connected conduits", "Number of pipes/tunnels hydraulically connected to the screened event.", "Use 1 for a single shared headrace; use the number of parallel conduits only when all carry the stated event flow."],
    ["Connected conduit diameter", "Diameter of each conduit represented in the connected event area, not necessarily the unit branch.", "Keep diameter, conduit count and total/event discharge on the same hydraulic basis."],
    ["Rated discharge Q0", "Use total plant Q for a shared waterway, or unit Q for a unit branch check.", "Keep the basis consistent with the selected diameter and number of conduits."],
    ["Net head H", "Use civil net head after losses for operation; use gross/static head for conservative pressure checks.", "Document which one is used."],
    ["Headrace length L", "Length from reservoir/intake to surge tank or turbine depending on the check.", "Use the hydraulic length from Step 4/5, not only horizontal map distance."],
    ["Comparison pressure rise", "A user-selected screening line for sensitivity only.", "Do not present 10-30% of head as a universal acceptance criterion; final limits come from pressure ratings, minimum pressure, machine limits and transient studies."],
    ["Closure time", "Guide-vane or valve closing time.", "If T_c is shorter than 2L/a, the full Joukowsky rise is approached."],
]


HEAD_BASIS_GUIDANCE = [
    ["Use Step 4 selected head", "Gross head or a justified first-pass effective head from Step 4.", "Best for lecture continuity and early sizing before detailed losses are trusted."],
    ["Use civil net head", "Step 5 selected head minus major, minor and separately declared other hydraulic losses.", "Best for final internal consistency after diameter, roughness and loss allowances are selected."],
    ["Manual head", "User-entered sensitivity or vendor-style value.", "Use to test min/max operating head, guaranteed head, or a corrected design case."],
]


MARKET_ROLE_GUIDANCE = [
    ["Daily energy shifting", "Generation and pumping price spread, cycling frequency, cycle efficiency.", "Annual benefit proxy and cycling risk."],
    ["Renewable firming", "Renewable output profile, curtailment, firming contract, storage duration.", "Benefit proxy, dispatch schedule, grid-service evidence."],
    ["Peaking capacity", "Capacity value, peak demand events, unit availability.", "Benefit proxy and reliability risk."],
    ["Reserve/reliability", "Reserve duration, ramp rate, availability, system security need.", "Risk register and qualitative benefit case."],
    ["Black start/system restoration", "Start capability, grid-code requirements, auxiliary supply.", "Traceability and risk mitigation; value separately and avoid stacking it over incompatible dispatch."],
    ["Long-duration storage", "Multi-day energy need, drought/renewable lull evidence, water balance.", "Storage-duration justification, hydrology risk, and capital utilisation."],
]


SNOWY_PRESET_NAMES = {"Snowy 2.0 - Plateau", "Snowy 2.0 - Ravine"}

# Course-defined comparison for the Step 2 case-study tab. The app does not
# claim which concept matches the adopted public Snowy 2.0 layout.
SNOWY_CASE_STUDY_ROWS = [
    ["1", "Cavern geology / geotechnical risk", "Mapped geology, structures, cover, stress and investigation evidence required", "Mapped geology, structures, cover, stress and investigation evidence required", "Do not rank cavern options from labels such as plateau or ravine; use site evidence (Step 7)."],
    ["2", "Pump submergence / setting depth", "Machine setting and simultaneous minimum tailwater evidence required", "Machine setting and simultaneous minimum tailwater evidence required", "Check cavitation and pump-mode submergence from actual operating levels (Steps 4 and 9)."],
    ["3", "Waterway configuration and transients", "Connected-system layout and transient model required", "Connected-system layout and transient model required", "Route length alone cannot establish pressure, lining or surge-control performance (Steps 5, 7 and 8)."],
    ["4", "Environmental footprint and approvals", "Mapped footprint, receptors and approval pathway required", "Mapped footprint, receptors and approval pathway required", "Disturbed land does not automatically imply lower impact or easier approval (Step 10)."],
    ["5", "Construction access, adits and spoil", "Measured access, portal and spoil quantities required", "Measured access, portal and spoil quantities required", "Compare quantities and constraints on the same evidence basis."],
    ["6", "Cost and schedule", "Dated quantities, rates, scope and risk basis required", "Dated quantities, rates, scope and risk basis required", "Do not rank costs from unsupported qualitative claims (Step 10)."],
    ["7", "Representative head / energy difference", "Representative gross head about 682.3 m", "Representative gross head about 686.7 m", "A 4.4 m difference is a sensitivity result, not an option-selection basis by itself."],
]


# Qualitative risk scales (ISO 31000-style 5x5 teaching matrix).
LIKELIHOOD_LEVELS = ["Rare", "Unlikely", "Possible", "Likely", "Almost certain"]
CONSEQUENCE_LEVELS = ["Insignificant", "Minor", "Moderate", "Major", "Severe"]


def risk_rating(likelihood: object, consequence: object) -> str:
    """Rating from the 5x5 matrix: score = likelihood index x consequence index,
    banded 1-4 Low, 5-9 Medium, 10-16 High, 17-25 Extreme."""
    try:
        score = (LIKELIHOOD_LEVELS.index(likelihood) + 1) * (CONSEQUENCE_LEVELS.index(consequence) + 1)
    except (ValueError, TypeError):
        return "-"
    if score >= 17:
        return "Extreme"
    if score >= 10:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


WORKFLOW_STEPS = [
    ("1", "Project brief, energy target, constraints, and site screening"),
    ("2", "Reservoir opportunity mapping and dam concept selection"),
    ("3", "Reservoir levels, dam sizing, gross head, and storage energy"),
    ("4", "Waterway corridor, layout, and design discharge"),
    ("5", "Shared conduit losses and diameter sensitivity"),
    ("6", "Unit branches, penstocks, and intake/outlet sizing"),
    ("7", "Underground civil structures and geotechnical checks"),
    ("8", "Surge and transient check"),
    ("9", "Turbine, pumping, efficiency, and operating envelope"),
    ("10", "NEM integration, risk register, and final recommendation"),
]

OVERVIEW_PAGE = "Overview"
REPORT_REFERENCE_PAGE = "Report / Refs / Equations"

STEP_PAGE_OPTIONS = [OVERVIEW_PAGE] + [f"Step {number}. {title}" for number, title in WORKFLOW_STEPS] + [REPORT_REFERENCE_PAGE]

STEP_SHORT_TITLES = {
    "1": "Brief & site screening",
    "2": "Reservoirs & dam concept",
    "3": "Levels, head & storage",
    "4": "Waterway & discharge",
    "5": "Losses & diameter",
    "6": "Branches & intakes",
    "7": "Underground civil",
    "8": "Surge & transients",
    "9": "Turbine & operation",
    "10": "NEM & risk",
}

REPORT_LOGIC_ROWS = [
    (
        "Step 1",
        "Project brief, constraints and site screening",
        "Define the design problem and screen the site before calculating. State the storage role, target MW, target hours, study boundary, data sources, confidence, access, transmission and environmental exclusions.",
        "Energy target, gross-head screen, route-efficiency screen, and continue/reject decision.",
        "Project brief, map scale, DEM/source list, constraint/exclusion map, data-confidence notes and first-pass alternatives.",
    ),
    (
        "Step 2",
        "Reservoir opportunity mapping and dam concept",
        "Use QGIS/DEM/contours to map plausible upper and lower reservoirs, dam/powerhouse sites and waterway corridors, then choose the reservoir arrangement and preliminary dam concept from valley shape, foundation, materials and constructability.",
        "Storage-elevation, gross-head, route-length, multi-criteria and dam-type screening checks.",
        "QGIS maps, reservoir-pair table, route options, dam type rationale, foundation/material assumptions, alternative comparison and investigation gaps.",
    ),
    (
        "Step 3",
        "Levels, dam sizing, head and storage",
        "Convert the mapped reservoir concept into operating levels, active depth, crest level, dam height, gross head and storage energy.",
        "Reservoir level, tailwater, gross-head, dam crest/height and storage-energy checks.",
        "Level table, schematic, storage volume/area, crest and dam height, gross head, deliverable GWh and duration.",
    ),
    (
        "Step 4",
        "Waterway layout and design discharge",
        "Draw the hydraulic alignment rather than only the ground topography, then estimate total plant discharge from selected power, head and efficiency and compare it with storage duration.",
        "Waterway length, turbine-centreline, design-discharge, storage-volume and duration checks.",
        "Plan and long section, chainage/elevation table, route alternatives, total Q, unit Q preview, storage duration and sensitivity to head/efficiency.",
    ),
    (
        "Step 5",
        "Shared conduit losses and diameter",
        "Select a realistic shared tunnel/shaft diameter by checking velocity, Reynolds number, Darcy friction, local losses and net head sensitivity.",
        "Flow area, velocity, Reynolds number, roughness, friction/local-loss and net-head checks.",
        "Diameter sensitivity table, selected roughness, Reynolds/f value, local-loss components, loss budget and net head.",
    ),
    (
        "Step 6",
        "Unit branches and intake/outlet sizing",
        "Divide total discharge into units/penstocks and select branch velocity/diameter using hydraulic loss, transient risk, constructability and equipment interfaces.",
        "Unit-flow split, branch diameter, branch loss and instantaneous water-hammer checks.",
        "Velocity comparison, unit branch diameter, intake/outlet size, valve/bifurcation notes and selected design justification.",
    ),
    (
        "Step 7",
        "Underground civil and geotechnical checks",
        "Size the powerhouse system and check whether tunnels/caverns have enough cover, confinement, jacking resistance, access, support and space for electrical interfaces.",
        "Hydrostatic pressure, rock stress, confinement, hydraulic-jacking and lining-stress checks.",
        "Machine hall, transformer hall, IPB gallery, access, ventilation, drainage, rock class, lining class per waterway zone, support and geotechnical risks.",
    ),
    (
        "Step 8",
        "Surge and transient check",
        "Screen transient severity, identify whether surge control may be needed, and define the connected-system model required for design.",
        "Wave travel time, Joukowsky rapid-closure bound, rigid-column slower-closure screen, pressure envelope and next-study requirements.",
        "Surge concept, tank area/diameter, closure-time check, maximum/minimum transient head and need for detailed transient modelling.",
    ),
    (
        "Step 9",
        "Turbine, pumping and operating envelope",
        "Select a pump-turbine family and unit arrangement after head, discharge, unit flow, losses and operating role are known.",
        "Hydraulic power, generated power, pumping power, efficiency-chain and specific-speed checks.",
        "Turbine family, unit count, runner speed, generation and pumping power, part-load envelope and efficiency chain.",
    ),
    (
        "Step 10",
        "NEM integration, risk and recommendation",
        "Connect the design to grid need, market operation, construction-cost screening, environmental monitoring, risk and final go/no-go recommendation.",
        "Cycle efficiency, pumping energy, cost intensity, sourced economic screening and risk-rating checks.",
        "Dispatch logic, grid connection, cost screen, O&M, sourced discounted economics or a clearly labelled undiscounted proxy, monitoring plan, risk register and traceability matrix.",
    ),
]


def page_step_number(page: str) -> int | None:
    if not page.startswith("Step "):
        return None
    return int(page.split(".", 1)[0].replace("Step", "").strip())


def step_inputs_complete(step: int) -> bool:
    """Heuristic per-step completion flags used for the sidebar progress ticks."""
    checks = {
        1: step1_complete(),
        2: step1_complete(),
        3: levels_complete() and has_values("reservoir_volume_m3"),
        4: has_values("penstock_length_m") and head_defined(),
        5: has_values("penstock_diameter_m") and hydraulics_ready(),
        6: has_values("unit_penstock_diameter_m") and hydraulics_ready(),
    }
    return checks.get(step, False)


def page_nav_label(page: str) -> str:
    if page == OVERVIEW_PAGE:
        return "Overview"
    step = page_step_number(page)
    if step is not None:
        tick = "✓ " if step_inputs_complete(step) else ""
        return f"{tick}{step:02d} {STEP_SHORT_TITLES.get(str(step), page)}"
    if page == REPORT_REFERENCE_PAGE:
        return "Report / refs / equations"
    return page


def init_state() -> None:
    if st.session_state.get("_hydro_power_defaults_version", 0) < 6:
        for key, value in UI_STATE_DEFAULTS.items():
            st.session_state[key] = value
        for key in DESIGN_NONE_KEYS:
            st.session_state[key] = None
        st.session_state["page"] = STEP_PAGE_OPTIONS[0]
    for key, value in UI_STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    for key in DESIGN_NONE_KEYS:
        st.session_state.setdefault(key, None)
    st.session_state.setdefault("page", STEP_PAGE_OPTIONS[0])
    st.session_state.setdefault("market_role", "Daily energy shifting")
    st.session_state["_hydro_power_defaults_version"] = 6
    for item in LOSS_COMPONENT_DETAILS:
        component = item["component"]
        st.session_state.setdefault(f"loss_{component}", bool(item["default_use"]))
        st.session_state.setdefault(f"loss_count_{component}", int(item["default_count"]))
    # Re-pin persistent keys through the session-state API every run. Without
    # this, Streamlit deletes widget-owned keys as soon as their widget is not
    # rendered (e.g. navigating away from the step that owns the input), which
    # would silently reset design values to None.
    pinned_keys = list(UI_STATE_DEFAULTS) + list(DESIGN_NONE_KEYS) + ["market_role"]
    for item in LOSS_COMPONENT_DETAILS:
        component = item["component"]
        pinned_keys.extend([f"loss_{component}", f"loss_count_{component}"])
    for key in pinned_keys:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


EDITOR_STATE_KEYS = [
    "loss_component_editor",
    "concept_alternative_editor",
    "waterway_alignment_editor",
    "reach_loss_editor",
    "traceability_matrix_editor",
]


def reset_editable_tables() -> None:
    for key in EDITOR_STATE_KEYS:
        st.session_state.pop(key, None)


def reset_loss_component_defaults() -> None:
    for item in LOSS_COMPONENT_DETAILS:
        component = item["component"]
        st.session_state[f"loss_{component}"] = bool(item["default_use"])
        st.session_state[f"loss_count_{component}"] = int(item["default_count"])


def apply_loss_component_preset(name: str) -> None:
    preset = LOSS_COMPONENT_PRESETS.get(name, {})
    for item in LOSS_COMPONENT_DETAILS:
        component = item["component"]
        count = int(preset.get(component, 0))
        st.session_state[f"loss_{component}"] = count > 0
        st.session_state[f"loss_count_{component}"] = count
    st.session_state.pop("loss_component_editor", None)


def reset_lining_stress_defaults() -> None:
    penstock_diameter = num_or(st.session_state.get("penstock_diameter_m"), 5.0)
    upper_hwl = num_or(st.session_state.get("upper_hwl"), float("nan"))
    upper_lwl = num_or(st.session_state.get("upper_lwl"), float("nan"))
    lower_twl = num_or(st.session_state.get("lower_twl"), float("nan"))
    upper_representative = upper_hwl - max(upper_hwl - upper_lwl, 0.0) / 3.0 if np.isfinite(upper_hwl) and np.isfinite(upper_lwl) else float("nan")
    gross_head = upper_representative - lower_twl if np.isfinite(upper_representative) and np.isfinite(lower_twl) else float("nan")
    selected_head = fnum("teaching_effective_head_m")
    if np.isfinite(selected_head) and selected_head > 0:
        gross_head = selected_head
    st.session_state["lining_inner_radius_m"] = max(penstock_diameter / 2.0, 0.1)
    st.session_state["lining_thickness_m"] = 0.45
    st.session_state["lining_tensile_strength_mpa"] = 3.0
    st.session_state["lining_external_pressure_mpa"] = 1.0
    st.session_state["lining_static_head_m"] = max(num_or(gross_head, 100.0), 0.0)
    st.session_state["lining_transient_surcharge_m"] = 0.0


def apply_project_values(values: dict[str, object], active_preset: str) -> None:
    for key, value in UI_STATE_DEFAULTS.items():
        st.session_state[key] = value
    for key in DESIGN_NONE_KEYS:
        st.session_state[key] = None
    for key, value in values.items():
        st.session_state[key] = value
    st.session_state["active_preset"] = active_preset
    material = st.session_state.get("roughness_material", "Concrete (smooth)")
    if material != "Custom":
        st.session_state["roughness_m"] = ROUGHNESS[material]
    reset_loss_component_defaults()
    reset_lining_stress_defaults()
    reset_editable_tables()


def apply_preset(name: str) -> None:
    if name == NEW_PROJECT_PRESET:
        apply_project_values(NEW_PROJECT_DEFAULTS, NEW_PROJECT_PRESET)
        st.session_state["step4_head_basis"] = STEP4_HEAD_BASIS_GROSS
    else:
        apply_project_values(PRESETS[name], name)
        st.session_state["step4_head_basis"] = STEP4_HEAD_BASIS_MANUAL


def area_circle(diameter_m: float) -> float:
    if diameter_m <= 0:
        return float("nan")
    return math.pi * diameter_m**2 / 4.0


def safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0:
        return float("nan")
    return numerator / denominator


def undiscounted_benefit_cost_proxy(
    capital_cost_m: float,
    annual_om_m: float,
    annual_benefit_m: float,
    screening_life_y: float,
    cost_source: str,
    benefit_source: str,
) -> float:
    """Simple undiscounted comparison, shown only when both evidence bases exist."""
    if capital_cost_m <= 0 or screening_life_y <= 0 or not cost_source.strip() or not benefit_source.strip():
        return float("nan")
    simple_costs_m = capital_cost_m + annual_om_m * screening_life_y
    simple_benefits_m = annual_benefit_m * screening_life_y
    return safe_div(simple_benefits_m, simple_costs_m)


def fnum(key: str) -> float:
    """Read a numeric session value; None/invalid/missing become NaN."""
    value = st.session_state.get(key)
    if value is None:
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def num_or(value: object, fallback: float) -> float:
    """Coerce to a finite float, else return the fallback (for widget defaults)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return value if np.isfinite(value) else float(fallback)


def has_values(*keys: str) -> bool:
    """True when every named session input is a finite number."""
    return all(np.isfinite(fnum(key)) for key in keys)


def step1_complete() -> bool:
    return has_values("design_power_mw", "operation_hours")


def levels_complete() -> bool:
    return has_values("upper_hwl", "upper_lwl", "lower_hwl", "lower_twl")


def head_defined() -> bool:
    return has_values("teaching_effective_head_m")


def conduit_defined() -> bool:
    return has_values("penstock_length_m", "penstock_diameter_m")


def discharge_ready() -> bool:
    return step1_complete() and head_defined()


def hydraulics_ready() -> bool:
    return discharge_ready() and levels_complete() and conduit_defined()


def water_mu_dynamic_pa_s(temperature_c: float) -> float:
    temperature_k = temperature_c + 273.15
    return 2.414e-5 * 10 ** (247.8 / (temperature_k - 140.0))


def water_nu_kinematic_m2_s(temperature_c: float) -> float:
    return water_mu_dynamic_pa_s(temperature_c) / RHO


def q_from_power(power_mw: float, head_m: float, efficiency: float) -> float:
    if power_mw <= 0 or head_m <= 0 or efficiency <= 0:
        return float("nan")
    return power_mw * 1e6 / (RHO * G * head_m * efficiency)


def selected_design_discharge(head_m: float) -> float:
    """Return the declared benchmark flow or the power-target sizing flow."""
    basis = st.session_state.get("discharge_basis", DISCHARGE_BASIS_POWER)
    declared_q = fnum("operating_discharge_m3_s")
    if basis == DISCHARGE_BASIS_DECLARED and np.isfinite(declared_q) and declared_q > 0:
        return declared_q
    return q_from_power(fnum("design_power_mw"), head_m, float(st.session_state.sizing_efficiency))


def power_from_q_head(discharge_m3_s: float, head_m: float, efficiency: float) -> float:
    q = np.asarray(discharge_m3_s, dtype=float)
    if head_m <= 0 or efficiency <= 0:
        result = np.full_like(q, np.nan, dtype=float)
    else:
        result = RHO * G * q * head_m * efficiency / 1e6
        result = np.where(q < 0, np.nan, result)
    return float(result) if np.isscalar(discharge_m3_s) else result


def f_swamee_jain(reynolds: float, rel_roughness: float) -> float:
    if not np.isfinite(reynolds) or reynolds <= 0:
        return float("nan")
    if reynolds < 2000:
        return 64.0 / reynolds
    rr = max(rel_roughness, 0.0)
    turbulent = 0.25 / (math.log10(rr / 3.7 + 5.74 / reynolds**0.9)) ** 2
    if reynolds < 4000:
        laminar = 64.0 / reynolds
        weight = (reynolds - 2000.0) / 2000.0
        return (1.0 - weight) * laminar + weight * turbulent
    return turbulent


def head_loss(friction: float, length_m: float, diameter_m: float, velocity_m_s: float, k_sum: float) -> float:
    if diameter_m <= 0 or not np.isfinite(velocity_m_s):
        return float("nan")
    return (friction * length_m / diameter_m + k_sum) * velocity_m_s**2 / (2.0 * G)


def diameter_from_velocity(q_per_penstock: float, velocity_m_s: float) -> float:
    if q_per_penstock <= 0 or velocity_m_s <= 0:
        return float("nan")
    return math.sqrt(4.0 * q_per_penstock / (math.pi * velocity_m_s))


def diameter_from_headloss(
    q_per_penstock: float,
    length_m: float,
    target_headloss_m: float,
    roughness_m: float,
    temperature_c: float,
    k_sum: float,
) -> tuple[float, float, float, float, float]:
    if q_per_penstock <= 0 or length_m <= 0 or target_headloss_m <= 0:
        return (float("nan"),) * 5

    nu = water_nu_kinematic_m2_s(temperature_c)

    def evaluate(diameter_m: float) -> tuple[float, float, float, float]:
        area = area_circle(diameter_m)
        velocity = safe_div(q_per_penstock, area)
        reynolds = safe_div(velocity * diameter_m, nu)
        friction = f_swamee_jain(reynolds, safe_div(roughness_m, diameter_m))
        loss = head_loss(friction, length_m, diameter_m, velocity, k_sum)
        return loss, friction, reynolds, velocity

    low = 0.25
    high = 20.0
    low_loss = evaluate(low)[0]
    high_loss = evaluate(high)[0]
    while np.isfinite(high_loss) and high_loss > target_headloss_m and high < 80:
        high *= 1.5
        high_loss = evaluate(high)[0]
    if not np.isfinite(low_loss) or not np.isfinite(high_loss) or low_loss < target_headloss_m:
        return (float("nan"),) * 5

    for _ in range(80):
        mid = 0.5 * (low + high)
        mid_loss = evaluate(mid)[0]
        if not np.isfinite(mid_loss):
            return (float("nan"),) * 5
        if mid_loss > target_headloss_m:
            low = mid
        else:
            high = mid

    diameter = high
    loss, friction, reynolds, velocity = evaluate(diameter)
    return diameter, friction, reynolds, velocity, loss


def reservoir_levels() -> dict[str, float]:
    upper_hwl = float(fnum('upper_hwl'))
    upper_lwl = float(fnum('upper_lwl'))
    lower_hwl = float(fnum('lower_hwl'))
    lower_twl = float(fnum('lower_twl'))
    levels_are_finite = all(np.isfinite(value) for value in (upper_hwl, upper_lwl, lower_hwl, lower_twl))
    upper_range = max(upper_hwl - upper_lwl, 0.0) if levels_are_finite else float("nan")

    # Retain the course's one-third-drawdown value as an explicitly labelled
    # representative placeholder. It is not a universal definition of NWL.
    upper_representative = upper_hwl - upper_range / 3.0 if levels_are_finite else float("nan")
    lower_representative = lower_twl
    gross_head = upper_representative - lower_representative

    # Simultaneous operating extremes: upper-high/lower-low for maximum head,
    # and upper-low/lower-high for minimum head.
    gross_head_max = upper_hwl - lower_twl
    gross_head_min = upper_lwl - lower_hwl
    head_fluctuation_ratio = safe_div(gross_head_min, gross_head_max)
    return {
        "upper_hwl": upper_hwl,
        "upper_lwl": upper_lwl,
        "lower_hwl": lower_hwl,
        "lower_twl": lower_twl,
        "upper_nwl": upper_representative,
        "upper_representative": upper_representative,
        "lower_representative": lower_representative,
        "gross_head": gross_head,
        "gross_head_max": gross_head_max,
        "gross_head_min": gross_head_min,
        "head_fluctuation_ratio": head_fluctuation_ratio,
    }


def selected_k_sum() -> float:
    total = 0.0
    for component, value in LOSS_COMPONENTS.items():
        if st.session_state.get(f"loss_{component}", False):
            total += value * max(float(st.session_state.get(f"loss_count_{component}", 1)), 0.0)
    return total


def discharge_per_unit(total_discharge_m3_s: float) -> float:
    return safe_div(total_discharge_m3_s, max(int(st.session_state.get("units", 1)), 1))


def discharge_per_penstock(total_discharge_m3_s: float) -> float:
    return safe_div(total_discharge_m3_s, max(int(st.session_state.get("penstocks", 1)), 1))


def flow_for_reach_basis(basis: str, total_discharge_m3_s: float, custom_q_m3_s: float | None = None) -> float:
    if basis == "Per unit Q":
        return discharge_per_unit(total_discharge_m3_s)
    if basis == "Per penstock Q":
        return discharge_per_penstock(total_discharge_m3_s)
    if basis == "Custom Q" and custom_q_m3_s is not None and np.isfinite(custom_q_m3_s) and custom_q_m3_s > 0:
        return float(custom_q_m3_s)
    return total_discharge_m3_s


def reynolds_regime(reynolds: float) -> str:
    if not np.isfinite(reynolds):
        return "Not available"
    if reynolds < 2000:
        return "Laminar"
    if reynolds < 4000:
        return "Transitional"
    return "Turbulent"


def friction_method_label(reynolds: float) -> str:
    if not np.isfinite(reynolds):
        return "Check input data"
    if reynolds < 2000:
        return "Laminar: f = 64/Re"
    if reynolds < 4000:
        return "Transitional: blended laminar/turbulent estimate"
    return "Turbulent: Swamee-Jain explicit Colebrook approximation"


def render_loss_component_editor(key: str = "loss_component_editor") -> float:
    rows = []
    for item in LOSS_COMPONENT_DETAILS:
        component = item["component"]
        use = bool(st.session_state.get(f"loss_{component}", item["default_use"]))
        quantity = int(max(float(st.session_state.get(f"loss_count_{component}", item["default_count"])), 0.0))
        rows.append(
            {
                "Use": use,
                "Component": component,
                "Quantity": quantity,
                "Typical K": item["k"],
                "Guidance": item["guidance"],
            }
        )

    edited = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        key=key,
        disabled=["Component", "Typical K", "Guidance"],
        column_config={
            "Use": st.column_config.CheckboxColumn("Use"),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=0, max_value=20, step=1, format="%d"),
            "Typical K": st.column_config.NumberColumn("Typical K", format="%.2f"),
        },
    )

    for _, row in edited.iterrows():
        component = str(row["Component"])
        quantity_value = float(row["Quantity"]) if pd.notna(row["Quantity"]) else 0.0
        st.session_state[f"loss_{component}"] = bool(row["Use"])
        st.session_state[f"loss_count_{component}"] = int(max(quantity_value, 0.0))

    selected_rows = []
    for item in LOSS_COMPONENT_DETAILS:
        component = item["component"]
        if st.session_state.get(f"loss_{component}", False):
            quantity = max(float(st.session_state.get(f"loss_count_{component}", 1)), 0.0)
            contribution = item["k"] * quantity
            if contribution > 0:
                selected_rows.append([component, quantity, item["k"], contribution])

    if selected_rows:
        st.dataframe(
            pd.DataFrame(selected_rows, columns=["Selected component", "Quantity", "K each", "K contribution"]),
            hide_index=True,
            width="stretch",
            column_config={
                "Quantity": st.column_config.NumberColumn(format="%.0f"),
                "K each": st.column_config.NumberColumn(format="%.2f"),
                "K contribution": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        st.info("Select at least one component so local losses are included.")

    return selected_k_sum()


def render_reservoir_level_schematic(levels: dict[str, float], crest_level: float | None = None) -> None:
    upper_levels = [
        ("HWL", levels["upper_hwl"], MONASH_BLUE),
        ("Rep. upper", levels["upper_representative"], MONASH_ELECTRIC_BLUE),
        ("LWL", levels["upper_lwl"], MONASH_BLUEBERRY),
    ]
    lower_levels = [
        ("Lower HWL", levels["lower_hwl"], MONASH_BLUE),
        ("TWL", levels["lower_twl"], MONASH_BLUEBERRY),
    ]
    finite_values = [value for _, value, _ in upper_levels + lower_levels if np.isfinite(value)]
    if crest_level is not None and np.isfinite(crest_level):
        finite_values.append(crest_level)
    if not finite_values:
        return

    y_min = min(finite_values) - max(15.0, 0.04 * (max(finite_values) - min(finite_values)))
    y_max = max(finite_values) + max(15.0, 0.04 * (max(finite_values) - min(finite_values)))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-0.25, x1=0.25, y0=levels["upper_lwl"], y1=levels["upper_hwl"], fillcolor="rgba(171,245,249,0.42)", line=dict(width=0))
    fig.add_shape(type="rect", x0=1.15, x1=1.65, y0=levels["lower_twl"], y1=levels["lower_hwl"], fillcolor="rgba(171,245,249,0.42)", line=dict(width=0))

    upper_label_shifts = {"HWL": 24, "Rep. upper": 5, "LWL": -14}
    lower_label_shifts = {"Lower HWL": 12, "TWL": -12}
    for label, value, color in upper_levels:
        fig.add_shape(type="line", x0=-0.28, x1=0.28, y0=value, y1=value, line=dict(color=color, width=2))
        fig.add_annotation(
            x=0.32,
            y=value,
            yshift=upper_label_shifts[label],
            text=f"{label} {value:.1f} m",
            showarrow=False,
            xanchor="left",
            bgcolor=MONASH_WHITE,
            borderpad=1,
            font=dict(size=12, color=color),
        )
    for label, value, color in lower_levels:
        fig.add_shape(type="line", x0=1.12, x1=1.68, y0=value, y1=value, line=dict(color=color, width=2))
        fig.add_annotation(
            x=1.72,
            y=value,
            yshift=lower_label_shifts[label],
            text=f"{label} {value:.1f} m",
            showarrow=False,
            xanchor="left",
            bgcolor=MONASH_WHITE,
            borderpad=1,
            font=dict(size=12, color=color),
        )

    if crest_level is not None and np.isfinite(crest_level):
        fig.add_shape(type="line", x0=-0.35, x1=0.35, y0=crest_level, y1=crest_level, line=dict(color=MONASH_GREY_1, width=2, dash="dash"))
        fig.add_annotation(x=-0.38, y=crest_level, text=f"Crest {crest_level:.1f} m", showarrow=False, xanchor="right", font=dict(size=12, color=MONASH_GREY_1))

    fig.add_shape(type="line", x0=0.7, x1=0.7, y0=levels["lower_representative"], y1=levels["upper_representative"], line=dict(color=MONASH_GREY_1, width=2, dash="dot"))
    fig.add_annotation(x=0.74, y=0.5 * (levels["upper_representative"] + levels["lower_representative"]), text=f"Hg,rep = {levels['gross_head']:.1f} m", showarrow=False, xanchor="left", font=dict(size=13, color=MONASH_GREY_1))
    fig.add_annotation(x=0.0, y=y_min, text="Upper reservoir", showarrow=False, yanchor="bottom", font=dict(size=13))
    fig.add_annotation(x=1.4, y=y_min, text="Lower reservoir", showarrow=False, yanchor="bottom", font=dict(size=13))
    fig.update_layout(template="plotly_white", height=390, margin=dict(l=10, r=10, t=20, b=10), showlegend=False, yaxis_title="Elevation (m AHD)")
    fig.update_xaxes(visible=False, range=[-0.55, 2.15])
    fig.update_yaxes(range=[y_min, y_max])
    st.plotly_chart(fig, width="stretch")


def render_waterway_alignment_schematic(profile: pd.DataFrame) -> dict[str, float]:
    clean = profile.copy()
    for column in ["Chainage_m", "Ground_RL_m", "Waterway_RL_m"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["Chainage_m", "Waterway_RL_m"]).sort_values("Chainage_m")
    if len(clean) < 2:
        st.warning("At least two alignment points are required for a waterway drawing.")
        return {"profile_length_m": float("nan"), "plan_length_m": float("nan"), "drop_m": float("nan"), "average_gradient": float("nan")}

    dx = np.diff(clean["Chainage_m"].to_numpy(dtype=float))
    dz = np.diff(clean["Waterway_RL_m"].to_numpy(dtype=float))
    profile_length = float(np.sum(np.sqrt(dx**2 + dz**2)))
    plan_length = float(clean["Chainage_m"].iloc[-1] - clean["Chainage_m"].iloc[0])
    drop = float(clean["Waterway_RL_m"].iloc[0] - clean["Waterway_RL_m"].iloc[-1])
    average_gradient = safe_div(drop, profile_length)

    fig = go.Figure()
    if "Ground_RL_m" in clean and clean["Ground_RL_m"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=clean["Chainage_m"],
                y=clean["Ground_RL_m"],
                mode="lines+markers",
                name="Ground topography",
                line=dict(color=MONASH_GREY_2, width=2, dash="dash"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=clean["Chainage_m"],
            y=clean["Waterway_RL_m"],
            mode="lines+markers+text",
            text=clean.get("Element", pd.Series([""] * len(clean))),
            textposition="top center",
            name="Selected waterway alignment",
            line=dict(color=MONASH_BLUE, width=4),
            marker=dict(size=9, color=MONASH_BLUE),
        )
    )
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=35, b=10), legend=dict(orientation="h"), xaxis_title="Chainage (m)", yaxis_title="Elevation RL (m)")
    st.plotly_chart(fig, width="stretch")
    return {"profile_length_m": profile_length, "plan_length_m": plan_length, "drop_m": drop, "average_gradient": average_gradient}


def hydraulic_snapshot(power_mw: float, gross_head_m: float) -> dict[str, float]:
    diameter = float(fnum('penstock_diameter_m'))
    length = float(fnum('penstock_length_m'))
    roughness = float(st.session_state.roughness_m)
    temperature = float(st.session_state.temperature_c)
    n_pen = int(st.session_state.penstocks)
    k_sum = selected_k_sum()
    nu = water_nu_kinematic_m2_s(temperature)
    draft_head = float(fnum('draft_head_m'))
    if not np.isfinite(draft_head):
        draft_head = 0.0
    selected_head = float(fnum('teaching_effective_head_m'))
    other_loss = float(fnum('other_head_loss_m'))
    if not np.isfinite(other_loss):
        other_loss = 0.0
    flow_area_mode = st.session_state.get("flow_area_mode", "Per penstock")
    q_total = selected_design_discharge(selected_head)
    result = {
        "power_mw": power_mw,
        "gross_head_m": gross_head_m,
        "selected_head_m": selected_head,
        "draft_head_m": draft_head,
        "other_loss_m": other_loss,
        "net_head_m": float("nan"),
        "total_discharge_m3_s": float("nan"),
        "per_penstock_m3_s": float("nan"),
        "velocity_m_s": float("nan"),
        "flow_for_velocity_m3_s": float("nan"),
        "flow_area_mode": flow_area_mode,
        "reynolds": float("nan"),
        "friction": float("nan"),
        "major_loss_m": float("nan"),
        "minor_loss_m": float("nan"),
        "conduit_loss_m": float("nan"),
        "total_loss_m": float("nan"),
        "achieved_generation_mw": float("nan"),
        "k_sum": k_sum,
    }

    if (
        not np.isfinite(selected_head)
        or selected_head <= 0
        or not np.isfinite(q_total)
        or q_total <= 0
        or not np.isfinite(diameter)
        or diameter <= 0
        or not np.isfinite(length)
        or not np.isfinite(power_mw)
        or n_pen <= 0
    ):
        return result

    q_per = safe_div(q_total, n_pen)
    flow_for_velocity = q_total if flow_area_mode == "Shared conduit" else q_per
    velocity = safe_div(flow_for_velocity, area_circle(diameter))
    reynolds = safe_div(velocity * diameter, nu)
    friction = f_swamee_jain(reynolds, safe_div(roughness, diameter))
    major = head_loss(friction, length, diameter, velocity, 0.0)
    minor = head_loss(0.0, length, diameter, velocity, k_sum)
    conduit_loss = major + minor
    total_loss = other_loss + conduit_loss
    head_net = selected_head - total_loss
    eta_gen = st.session_state.eta_turbine * st.session_state.eta_generator * st.session_state.eta_transformer
    achieved_generation = power_from_q_head(q_total, head_net, eta_gen) if head_net > 0 else float("nan")

    result.update(
        {
            "net_head_m": head_net,
            "total_discharge_m3_s": q_total,
            "per_penstock_m3_s": q_per,
            "velocity_m_s": velocity,
            "flow_for_velocity_m3_s": flow_for_velocity,
            "reynolds": reynolds,
            "friction": friction,
            "major_loss_m": major,
            "minor_loss_m": minor,
            "conduit_loss_m": conduit_loss,
            "total_loss_m": total_loss,
            "achieved_generation_mw": achieved_generation,
            "k_sum": k_sum,
        }
    )
    return result


def select_turbine(head_m: float, discharge_m3_s: float) -> str | None:
    try:
        head_m = float(head_m)
        discharge_m3_s = float(discharge_m3_s)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(head_m) and np.isfinite(discharge_m3_s)):
        return None
    if head_m > 300 and discharge_m3_s < 50:
        return "Pelton"
    if 50 <= head_m <= 700 and discharge_m3_s >= 50:
        return "Francis"
    if head_m < 20 and discharge_m3_s >= 20:
        return "Bulb"
    if head_m < 50 and discharge_m3_s >= 10:
        return "Kaplan"
    return "Francis"


def turbine_zone_candidates(head_m: float, discharge_m3_s: float) -> list[str]:
    if not (np.isfinite(head_m) and np.isfinite(discharge_m3_s) and head_m > 0 and discharge_m3_s > 0):
        return []
    candidates = []
    if 50.0 <= head_m <= 2000.0 and 0.1 <= discharge_m3_s <= 50.0:
        candidates.append("Pelton")
    if 20.0 <= head_m <= 700.0 and 0.5 <= discharge_m3_s <= 200.0:
        candidates.append("Francis")
    if 5.0 <= head_m <= 100.0 and 10.0 <= discharge_m3_s <= 1000.0:
        candidates.append("Kaplan")
    if 5.0 <= head_m <= 20.0 and 50.0 <= discharge_m3_s <= 1000.0:
        candidates.append("Bulb")
    return candidates


def turbine_application_message(head_m: float, plant_q: float, unit_q: float) -> tuple[str, str]:
    plant_candidates = turbine_zone_candidates(head_m, plant_q)
    unit_candidates = turbine_zone_candidates(head_m, unit_q)
    if unit_candidates:
        if not plant_candidates:
            return (
                "info",
                "The plant-total point is outside the teaching chart mainly because total Q is large. "
                f"Use the unit-level point for selection: {', '.join(unit_candidates)} is plausible at the current unit flow.",
            )
        return ("success", f"The current point sits inside the teaching envelope: {', '.join(unit_candidates)} is plausible at unit level.")
    if head_m > 700.0 and unit_q > 50.0:
        return (
            "warning",
            "High head and high unit flow sit outside the simplified teaching zones. Increase unit count to reduce Q_u, "
            "split the flow path, revise head/loss assumptions, or treat the case as vendor-specific high-head Francis/Pelton screening.",
        )
    if head_m < 5.0:
        return ("warning", "Head is below the teaching turbine chart. Recheck levels, tailwater, and whether PHES is the right technology for this site.")
    if unit_q > 1000.0:
        return ("warning", "Unit discharge is very high. Increase the number of units or split waterways before selecting a turbine family.")
    if unit_q < 0.1:
        return ("warning", "Unit discharge is very low for the plotted hydropower families. Recheck power, head, and unit count.")
    return (
        "warning",
        "The operating point is near or outside the simplified teaching chart. Use unit-level Q, min/max head, specific speed, "
        "and manufacturer data before choosing a turbine family.",
    )


def ensure_option_state(key: str, options: list[str]) -> None:
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def recommend_dam_type(
    arrangement: str,
    valley_geometry: str,
    foundation_quality: str,
    construction_material: str,
    dam_height_m: float,
    active_volume_m3: float,
) -> str:
    if arrangement in {"Existing reservoir reuse", "Mine pit or quarry reuse"}:
        return "Existing reservoir/dam reuse"
    if foundation_quality == "Unknown":
        return "Further site investigation required"
    if arrangement == "Ring-dike / turkey-nest reservoir" or valley_geometry == "Plateau or ridge top":
        return "Ring-dike / turkey-nest reservoir"
    if valley_geometry == "Narrow gorge" and foundation_quality == "Very strong rock":
        return "Arch dam" if dam_height_m >= 50.0 else "RCC / concrete gravity dam"
    if valley_geometry == "Narrow valley" and foundation_quality in {"Very strong rock", "Good rock"}:
        if construction_material == "RCC / concrete aggregate available" or active_volume_m3 < 30e6:
            return "RCC / concrete gravity dam"
        return "Concrete faced rockfill dam (CFRD)"
    if construction_material == "Rockfill available" or dam_height_m >= 60.0 or active_volume_m3 >= 30e6:
        return "Rockfill embankment"
    if construction_material == "Clay / earthfill available" and dam_height_m <= 50.0:
        return "Earthfill embankment"
    if foundation_quality in {"Soil / weathered foundation", "Unknown"}:
        return "Further site investigation required"
    return "Concrete faced rockfill dam (CFRD)"


def dam_selection_trace_rows(
    arrangement: str,
    valley_geometry: str,
    foundation_quality: str,
    construction_material: str,
    dam_height_m: float,
    active_volume_m3: float,
) -> list[list[str]]:
    height_label = metric_value(dam_height_m, " m", 1) if np.isfinite(dam_height_m) else "Not calculated"
    volume_label = metric_value(active_volume_m3 / 1e6, " GL", 1) if np.isfinite(active_volume_m3) else "Not calculated"
    rows: list[list[str]] = []
    governed = False

    def add(priority: int, criterion: str, current_value: str, passes: bool, outcome: str, note: str) -> None:
        nonlocal governed
        if passes and not governed:
            status = "Governing rule"
            governed = True
        elif passes:
            status = "Would support, but later in priority"
        else:
            status = "Not triggered"
        rows.append([str(priority), criterion, current_value, status, outcome if passes else "-", note])

    add(
        1,
        "Existing storage or mine/quarry reuse",
        arrangement,
        arrangement in {"Existing reservoir reuse", "Mine pit or quarry reuse"},
        "Existing reservoir/dam reuse",
        "Reuse cases need operating constraints, dam-safety limits, intake/outlet works and approvals evidence.",
    )
    add(
        2,
        "Foundation known well enough for new-dam selection",
        foundation_quality,
        foundation_quality == "Unknown",
        "Further site investigation required",
        "For a new dam, unknown foundation quality should stop a confident dam-family recommendation.",
    )
    add(
        3,
        "Plateau/ridge or ring-dike arrangement",
        f"{arrangement}; {valley_geometry}",
        arrangement == "Ring-dike / turkey-nest reservoir" or valley_geometry == "Plateau or ridge top",
        "Ring-dike / turkey-nest reservoir",
        "The storage is formed mainly by a perimeter embankment rather than a valley closure.",
    )
    narrow_gorge = valley_geometry == "Narrow gorge" and foundation_quality == "Very strong rock"
    add(
        4,
        "Narrow gorge with very strong rock",
        f"{valley_geometry}; {foundation_quality}; height {height_label}",
        narrow_gorge,
        "Arch dam" if np.isfinite(dam_height_m) and dam_height_m >= 50.0 else "RCC / concrete gravity dam",
        "Arch dams require very strong abutments; lower dams in a competent gorge may screen as RCC/gravity.",
    )
    narrow_valley = valley_geometry == "Narrow valley" and foundation_quality in {"Very strong rock", "Good rock"}
    add(
        5,
        "Narrow valley with competent rock",
        f"{valley_geometry}; {foundation_quality}; {construction_material}; volume {volume_label}",
        narrow_valley,
        "RCC / concrete gravity dam" if construction_material == "RCC / concrete aggregate available" or (np.isfinite(active_volume_m3) and active_volume_m3 < 30e6) else "Concrete faced rockfill dam (CFRD)",
        "Competent narrow valleys can suit RCC/gravity or CFRD depending on material supply and storage scale.",
    )
    large_or_rockfill = construction_material == "Rockfill available" or (np.isfinite(dam_height_m) and dam_height_m >= 60.0) or (np.isfinite(active_volume_m3) and active_volume_m3 >= 30e6)
    add(
        6,
        "Rockfill supply, high dam or large storage",
        f"{construction_material}; height {height_label}; volume {volume_label}",
        large_or_rockfill,
        "Rockfill embankment",
        "Large PHES reservoirs often need robust embankment concepts and rockfill may be available from excavation.",
    )
    add(
        7,
        "Clay/earthfill available and low-to-moderate height",
        f"{construction_material}; height {height_label}",
        construction_material == "Clay / earthfill available" and np.isfinite(dam_height_m) and dam_height_m <= 50.0,
        "Earthfill embankment",
        "Earthfill is a plausible screening choice only where materials and modest height support it.",
    )
    add(
        8,
        "Weak or weathered foundation",
        foundation_quality,
        foundation_quality == "Soil / weathered foundation",
        "Further site investigation required",
        "Weathered or soil foundations need seepage, settlement and stability evidence before a dam family is defensible.",
    )
    add(
        9,
        "Fallback when no stronger rule governs",
        "No earlier rule selected",
        not governed,
        "Concrete faced rockfill dam (CFRD)",
        "CFRD is used as a cautious concept placeholder where rockfill-style construction is plausible but seepage control is important.",
    )
    return rows


def dam_selection_governing_text(
    arrangement: str,
    valley_geometry: str,
    foundation_quality: str,
    construction_material: str,
    dam_height_m: float,
    active_volume_m3: float,
) -> str:
    for _, criterion, _, status, outcome, _ in dam_selection_trace_rows(
        arrangement,
        valley_geometry,
        foundation_quality,
        construction_material,
        dam_height_m,
        active_volume_m3,
    ):
        if status == "Governing rule":
            return f"{criterion} -> {outcome}"
    return "No governing rule recorded"


def velocity_status(velocity_m_s: float) -> tuple[str, str]:
    if not np.isfinite(velocity_m_s):
        return "info", "Provide valid head, discharge, and diameter inputs."
    if velocity_m_s > 7.0:
        return "error", "Velocity exceeds about 7 m/s. Revisit diameter, layout, or number of penstocks."
    if velocity_m_s > 5.5:
        return "warning", "Velocity is above the usual 3.5-5.5 m/s pressure-waterway teaching range."
    if velocity_m_s >= 3.5:
        return "success", "Velocity is within the usual 3.5-5.5 m/s pressure-waterway teaching range."
    return "info", "Velocity is below 3.5 m/s. This is hydraulically gentle but may be uneconomic."


def metric_value(value: float, unit: str = "", digits: int = 1) -> str:
    if not np.isfinite(value):
        return "-"
    if abs(value) >= 1000 and digits > 0:
        return f"{value:,.{digits}f}{unit}"
    return f"{value:.{digits}f}{unit}"


def render_screening_limitation() -> None:
    st.warning(
        "Desktop GIS screening identifies candidates, not confirmed feasible projects. "
        "Students must still check geology, hydrology, environmental approvals, heritage, land tenure, "
        "grid connection, water rights, constructability, and commercial feasibility before recommending a site."
    )


def render_reference_links(links: list[list[str]]) -> None:
    st.markdown("\n".join(f"- [{title}]({url}) - {note}" for title, url, note in links))


def qgis_starter_screen(
    head_m: float,
    route_km: float,
    storage_gwh: float,
    target_gwh: float,
    constraint_flag: str,
    ready_count: int,
) -> tuple[str, str, float, list[str]]:
    head_to_distance = safe_div(head_m, route_km)
    notes = []
    score = 0

    if head_m >= 150.0:
        score += 2
        notes.append("Head is strong enough for a first-pass PHES screen.")
    elif head_m >= 80.0:
        score += 1
        notes.append("Head is marginal; compare with a higher-head alternative.")
    else:
        notes.append("Head is low for a first PHES teaching case.")

    if head_to_distance >= 60.0:
        score += 2
        notes.append("Head-to-distance is attractive for desktop screening.")
    elif head_to_distance >= 30.0:
        score += 1
        notes.append("Head-to-distance needs review; route length may drive cost.")
    else:
        notes.append("Head-to-distance is weak; find a shorter or steeper pair.")

    if target_gwh > 0 and storage_gwh >= target_gwh:
        score += 2
        notes.append("Storage meets the project energy target.")
    elif target_gwh > 0 and storage_gwh >= 0.6 * target_gwh:
        score += 1
        notes.append("Storage is close; test active depth or plant duration sensitivity.")
    else:
        notes.append("Storage is below target; resize the reservoir or reduce duration.")

    if constraint_flag == "No obvious fatal flag":
        score += 2
        notes.append("No fatal constraint recorded yet.")
    elif constraint_flag == "Review required":
        score += 1
        notes.append("Constraint layer needs a review decision before Step 2.")
    else:
        notes.append("A possible fatal constraint should stop or reroute the candidate.")

    if ready_count >= 5:
        score += 1
        notes.append("Enough GIS evidence exists for a tutorial-level Step 2 entry.")
    else:
        notes.append("Collect more GIS evidence before treating this as a candidate.")

    if constraint_flag == "Possible fatal constraint":
        return "Reject / reroute", "error", head_to_distance, notes
    if score >= 7:
        return "Proceed to Step 2", "success", head_to_distance, notes
    if score >= 4:
        return "Review before Step 2", "warning", head_to_distance, notes
    return "Hold: map another option", "error", head_to_distance, notes


def render_qgis_starter_lab() -> None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="lab-kicker">Tutorial activity</div>
            <div class="lab-title">Screen one mapped candidate</div>
            <div class="lab-caption">Pick a QGIS demo, tick the starter layers you have, then enter rough mapped values to decide whether the candidate is worth carrying into Step 2.</div>
            """,
            unsafe_allow_html=True,
        )
        demo_names = [row[0] for row in QGIS_TUTORIAL_DEMOS]
        selected_demo = st.selectbox("Choose today's tutorial demo", demo_names, key="step1_qgis_demo")
        demo = next(row for row in QGIS_TUTORIAL_DEMOS if row[0] == selected_demo)
        st.dataframe(
            pd.DataFrame(
                [
                    ["Starting data", demo[1]],
                    ["QGIS tool sequence", demo[2]],
                    ["Student output", demo[3]],
                    ["App connection", demo[4]],
                ],
                columns=["Lab card item", "What to do"],
            ),
            hide_index=True,
            width="stretch",
        )

        st.markdown("**Data readiness**")
        ready_cols = st.columns(3)
        ready_count = 0
        for i, (label, help_text) in enumerate(QGIS_STARTER_DATA_ITEMS):
            with ready_cols[i % 3]:
                if st.checkbox(label, key=f"step1_qgis_ready_{i}", help=help_text):
                    ready_count += 1
        st.progress(ready_count / len(QGIS_STARTER_DATA_ITEMS))
        st.caption(f"{ready_count} of {len(QGIS_STARTER_DATA_ITEMS)} starter layers/evidence items are ready.")

        st.markdown("**Quick candidate test**")
        target_gwh = fnum('design_power_mw') * fnum('operation_hours') / 1000.0
        target_gwh = num_or(target_gwh, 2.0)
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            starter_upper = st.number_input("Upper RL from map (m)", min_value=0.0, max_value=3000.0, value=num_or(fnum('upper_hwl'), 500.0), step=10.0, key="step1_starter_upper_rl")
        with q2:
            starter_lower = st.number_input("Lower RL from map (m)", min_value=0.0, max_value=3000.0, value=num_or(fnum('lower_twl'), 200.0), step=10.0, key="step1_starter_lower_rl")
        with q3:
            starter_route_km = st.number_input("Feasible route length (km)", min_value=0.1, max_value=100.0, value=max(num_or(safe_div(fnum('penstock_length_m'), 1000.0), 5.0), 0.1), step=0.5, key="step1_starter_route_km")
        with q4:
            starter_storage_gwh = st.number_input("Estimated storage (GWh)", min_value=0.0, max_value=5000.0, value=max(target_gwh, 0.1), step=0.5, key="step1_starter_storage_gwh")
        s1, s2 = st.columns(2)
        with s1:
            constraint_flag = st.selectbox(
                "Constraint result",
                ["No obvious fatal flag", "Review required", "Possible fatal constraint"],
                key="step1_starter_constraint",
                help="Classify this from a QGIS constraint overlay, not from head or storage alone. Open the guide below for the evidence rule.",
            )
        with s2:
            st.metric("Target storage", metric_value(target_gwh, " GWh", 1))

        with st.expander("How to choose the constraint result"):
            st.dataframe(
                pd.DataFrame(
                    CONSTRAINT_RESULT_GUIDE,
                    columns=["Option", "When to select it", "Typical evidence", "What to do next"],
                ),
                hide_index=True,
                width="stretch",
            )
            st.markdown("**New-project checking workflow**")
            st.dataframe(
                pd.DataFrame(
                    NEW_PROJECT_CONSTRAINT_CHECKS,
                    columns=["Order", "Check", "Student action"],
                ),
                hide_index=True,
                width="stretch",
            )
            st.info(
                "The Snowy teaching presets use an existing public reservoir pair, but their detailed geometry remains a course model. "
                "For a new project, do not tick 'No obvious fatal flag' "
                "until your own QGIS overlay shows that the reservoir, waterway, powerhouse, access and grid path "
                "avoid hard exclusions at desktop scale."
            )

        starter_head = starter_upper - starter_lower
        status, status_kind, head_to_distance, notes = qgis_starter_screen(
            starter_head,
            starter_route_km,
            starter_storage_gwh,
            target_gwh,
            constraint_flag,
            ready_count,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mapped head", metric_value(starter_head, " m", 1))
        m2.metric("Head / route", metric_value(head_to_distance, " m/km", 1))
        m3.metric("Storage / target", metric_value(safe_div(starter_storage_gwh, target_gwh), " x", 2))
        m4.metric("Decision", status)
        st.markdown(f'<span class="decision-pill decision-{status_kind}">{status}</span>', unsafe_allow_html=True)
        getattr(st, status_kind)("; ".join(notes[:3]))
        st.caption("Use this as a tutorial triage only. A real candidate still needs the Step 2 evidence table, Step 3 storage/head check, and Step 10 risk review.")


def build_step2_evidence_summary(ready_flags: list[bool] | None = None) -> dict[str, object]:
    if ready_flags is None:
        ready_flags = [bool(st.session_state.get(f"step2_evidence_ready_{i}", False)) for i in range(len(STEP2_EVIDENCE_ITEMS))]
    ready_rows = [row for row, ready in zip(STEP2_EVIDENCE_ITEMS, ready_flags) if ready]
    missing_rows = [row for row, ready in zip(STEP2_EVIDENCE_ITEMS, ready_flags) if not ready]
    ready_count = len(ready_rows)
    total = len(STEP2_EVIDENCE_ITEMS)
    if ready_count >= 6:
        level = "Strong"
        kind = "success"
        message = "Evidence is strong enough for a concept-level Step 2 submission."
    elif ready_count >= 4:
        level = "Partial"
        kind = "warning"
        message = "Evidence is partly ready. Keep the option, but clearly mark missing survey, geology or storage data."
    else:
        level = "Thin"
        kind = "info"
        message = "Evidence is thin. Treat the site as a candidate idea, not a selected concept."
    return {
        "ready_count": ready_count,
        "total": total,
        "fraction": safe_div(ready_count, total),
        "level": level,
        "kind": kind,
        "message": message,
        "ready_items": [row[0] for row in ready_rows],
        "missing_items": [row[0] for row in missing_rows],
        "ready_next_uses": [row[2] for row in ready_rows],
    }


def store_step2_evidence_summary(summary: dict[str, object]) -> None:
    st.session_state["step2_evidence_ready_count"] = int(summary["ready_count"])
    st.session_state["step2_evidence_total"] = int(summary["total"])
    st.session_state["step2_evidence_level"] = str(summary["level"])
    st.session_state["step2_evidence_ready_items"] = list(summary["ready_items"])
    st.session_state["step2_evidence_missing_items"] = list(summary["missing_items"])
    st.session_state["step2_evidence_ready_next_uses"] = list(summary["ready_next_uses"])


def current_step2_evidence_summary() -> dict[str, object]:
    summary = build_step2_evidence_summary()
    store_step2_evidence_summary(summary)
    return summary


def compact_list_text(items: list[str], limit: int = 4, empty: str = "None yet") -> str:
    if not items:
        return empty
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:limit])}, plus {len(items) - limit} more"


def render_step2_evidence_snapshot(summary: dict[str, object] | None = None) -> None:
    summary = summary or current_step2_evidence_summary()
    next_uses = list(summary["ready_next_uses"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Evidence ready", f"{int(summary['ready_count'])}/{int(summary['total'])}")
    c2.metric("Readiness status", str(summary["level"]))
    c3.metric("Downstream links", str(len(next_uses)))
    st.caption(f"Ready evidence feeds: {compact_list_text(next_uses, limit=3, empty='No downstream use yet')}.")
    st.caption(
        "Evidence ticks feed this Step 2 status, the Step 3 provisional-warning check, and the report-export traceability text."
    )


def render_step2_evidence_checklist() -> None:
    st.markdown("**Evidence readiness checklist**")
    ready_flags = []
    cols = st.columns(2)
    for i, (item, evidence, next_use) in enumerate(STEP2_EVIDENCE_ITEMS):
        with cols[i % 2]:
            ready_flags.append(
                st.checkbox(item, key=f"step2_evidence_ready_{i}", help=f"{evidence} Used in: {next_use}.")
            )
    summary = build_step2_evidence_summary(ready_flags)
    store_step2_evidence_summary(summary)
    st.progress(float(summary["fraction"]))
    render_step2_evidence_snapshot(summary)
    getattr(st, str(summary["kind"]))(str(summary["message"]))

    status_df = pd.DataFrame(
        [
            [
                item,
                "Yes" if ready_flags[i] else "No",
                "Mapped / calculated" if ready_flags[i] else "Not started",
                evidence,
                next_use,
            ]
            for i, (item, evidence, next_use) in enumerate(STEP2_EVIDENCE_ITEMS)
        ],
        columns=["Evidence item", "Ticked", "Detailed status", "What students should attach", "Feeds later step"],
    )
    with st.expander("Detailed evidence status table"):
        st.data_editor(
            status_df,
            num_rows="fixed",
            width="stretch",
            key="step2_evidence_status_editor",
            disabled=["Evidence item", "Ticked", "What students should attach", "Feeds later step"],
            column_config={
                "Detailed status": st.column_config.SelectboxColumn(
                    "Detailed status",
                    options=["Not started", "Mapped / calculated", "Screenshot attached", "Needs review", "Not applicable"],
                    required=True,
                )
            },
        )
        st.caption(
            "The ticked column is the quick readiness score used by the app. The detailed status column is for report notes."
        )


def dam_concept_schematic_figure(selected_type: str | None = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0))
    axes = axes.flatten()
    concepts = [
        ("Earth/rockfill", "Embankment\nwith core/filters"),
        ("CFRD", "Rockfill with\nconcrete face"),
        ("Gravity/RCC", "Concrete mass\non rock"),
        ("Arch", "Plan view:\nload to abutments"),
        ("Ring-dike", "Perimeter\nembankment"),
        ("Existing reuse", "Modify intake,\noutlet, operations"),
    ]
    highlight_map = {
        "Earthfill embankment": "Earth/rockfill",
        "Rockfill embankment": "Earth/rockfill",
        "Concrete faced rockfill dam (CFRD)": "CFRD",
        "RCC / concrete gravity dam": "Gravity/RCC",
        "Arch dam": "Arch",
        "Ring-dike / turkey-nest reservoir": "Ring-dike",
        "Existing reservoir/dam reuse": "Existing reuse",
    }
    highlighted = highlight_map.get(selected_type or "")

    for ax, (name, subtitle) in zip(axes, concepts):
        ax.set_aspect("equal")
        ax.axis("off")
        edge = MONASH_ELECTRIC_BLUE if name == highlighted else MONASH_GREY_1
        lw = 2.6 if name == highlighted else 1.4
        ax.set_title(name, fontsize=11, color=edge)

        if name == "Earth/rockfill":
            ax.fill([0.15, 0.48, 0.85], [0.15, 0.72, 0.15], color=MONASH_GREY_3, ec=edge, lw=lw)
            ax.fill([0.42, 0.48, 0.55], [0.15, 0.68, 0.15], color=MONASH_GREY_2, alpha=0.75)
            ax.fill_between([0.02, 0.25], 0.15, 0.46, color=MONASH_HERITAGE_BLUE, alpha=0.85)
            ax.text(0.49, 0.08, "foundation", ha="center", fontsize=8)
            ax.text(0.49, 0.42, "core", ha="center", fontsize=8, color="white")
        elif name == "CFRD":
            ax.fill([0.15, 0.48, 0.85], [0.15, 0.72, 0.15], color=MONASH_GREY_3, ec=edge, lw=lw)
            ax.plot([0.18, 0.48], [0.16, 0.70], color=MONASH_BLUEBERRY, lw=3)
            ax.fill_between([0.02, 0.22], 0.15, 0.47, color=MONASH_HERITAGE_BLUE, alpha=0.85)
            ax.text(0.27, 0.48, "concrete face", fontsize=8, rotation=60)
        elif name == "Gravity/RCC":
            ax.fill([0.25, 0.45, 0.78, 0.25], [0.15, 0.72, 0.15, 0.15], color=MONASH_GREY_2, ec=edge, lw=lw)
            ax.fill_between([0.02, 0.28], 0.15, 0.55, color=MONASH_HERITAGE_BLUE, alpha=0.85)
            ax.text(0.52, 0.37, "weight\nresists load", ha="center", fontsize=8)
        elif name == "Arch":
            theta = np.linspace(-1.2, 1.2, 80)
            ax.plot(0.5 + 0.28 * np.sin(theta), 0.45 + 0.28 * np.cos(theta), color=edge, lw=4)
            ax.fill([0.08, 0.25, 0.25, 0.08], [0.18, 0.18, 0.78, 0.78], color=MONASH_GREY_2)
            ax.fill([0.75, 0.92, 0.92, 0.75], [0.18, 0.18, 0.78, 0.78], color=MONASH_GREY_2)
            ax.fill([0.25, 0.75, 0.66, 0.34], [0.62, 0.62, 0.78, 0.78], color=MONASH_HERITAGE_BLUE, alpha=0.85)
            ax.text(0.5, 0.12, "strong abutments", ha="center", fontsize=8)
        elif name == "Ring-dike":
            outer = plt.Circle((0.5, 0.48), 0.34, fill=False, ec=edge, lw=lw + 1)
            inner = plt.Circle((0.5, 0.48), 0.24, fill=True, color=MONASH_HERITAGE_BLUE, alpha=0.85)
            ax.add_patch(inner)
            ax.add_patch(outer)
            ax.text(0.5, 0.10, "long perimeter crest", ha="center", fontsize=8)
        else:
            ax.fill([0.15, 0.48, 0.85], [0.15, 0.66, 0.15], color=MONASH_GREY_2, ec=edge, lw=lw)
            ax.fill_between([0.02, 0.25], 0.15, 0.48, color=MONASH_HERITAGE_BLUE, alpha=0.85)
            ax.add_patch(plt.Rectangle((0.58, 0.18), 0.12, 0.18, facecolor=MONASH_GREY_1))
            ax.text(0.62, 0.40, "new intake /\noutlet works", ha="center", fontsize=8)

        ax.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=8.4, color=MONASH_GREY_1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Dam concept families used for PHES screening", fontsize=14, color=MONASH_GREY_1)
    fig.tight_layout()
    return fig


def dam_dimension_schematic_figure(
    levels: dict[str, float],
    crest_level: float,
    dam_height: float,
    dam_volume: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.set_facecolor("white")
    h = max(num_or(dam_height, 30.0), 1.0)
    crest_width = max(num_or(st.session_state.get("dam_crest_width_m"), 8.0), 1.0)
    us = max(num_or(st.session_state.get("upstream_slope_hv"), 2.5), 0.1)
    ds = max(num_or(st.session_state.get("downstream_slope_hv"), 2.0), 0.1)
    base_left = -us * h
    base_right = crest_width + ds * h
    ax.fill([base_left, 0.0, crest_width, base_right], [0.0, h, h, 0.0], color=MONASH_GREY_3, ec=MONASH_GREY_1, lw=1.8)
    ax.fill_between([base_left - 0.38 * h, base_left + 0.05 * h], 0.0, 0.72 * h, color=MONASH_HERITAGE_BLUE, alpha=0.85)
    ax.plot([0.0, crest_width], [h, h], color=MONASH_GREY_1, lw=2)
    ax.hlines(max(levels.get("upper_hwl", h * 0.82), 0.0) - num_or(st.session_state.get("dam_foundation_rl"), 0.0), base_left - 0.35 * h, base_left + 0.08 * h, color=MONASH_BLUE, lw=1.6)
    ax.annotate("", xy=(base_right + 0.12 * h, h), xytext=(base_right + 0.12 * h, 0.0), arrowprops=dict(arrowstyle="<->", lw=1.5, color=MONASH_GREY_1))
    ax.text(base_right + 0.16 * h, 0.5 * h, r"$H_{dam}$", va="center", fontsize=11)
    ax.annotate("", xy=(0.0, h + 0.08 * h), xytext=(crest_width, h + 0.08 * h), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(0.5 * crest_width, h + 0.10 * h, "crest width", ha="center", fontsize=9)
    ax.text(0.35 * base_left, 0.45 * h, f"upstream slope\n{us:.1f}H:1V", ha="center", fontsize=9)
    ax.text(crest_width + 0.42 * (base_right - crest_width), 0.42 * h, f"downstream slope\n{ds:.1f}H:1V", ha="center", fontsize=9)
    ax.text(0.5 * (base_left + base_right), -0.12 * h, f"screening fill volume = {metric_value(dam_volume / 1e6, ' million m3', 2)}", ha="center", fontsize=9.5)
    ax.text(base_left - 0.30 * h, 0.75 * h, "reservoir", ha="center", fontsize=9, color=MONASH_BLUE)
    ax.text(0.5 * (base_left + base_right), 0.04 * h, "foundation RL", ha="center", fontsize=9)
    ax.set_xlim(base_left - 0.45 * h, base_right + 0.38 * h)
    ax.set_ylim(-0.18 * h, h * 1.25)
    ax.axis("off")
    ax.set_title("Dam dimension definition used by the Step 3 screening calculation", loc="left", fontsize=13, color=MONASH_GREY_1)
    fig.tight_layout()
    return fig


def confinement_jacking_schematic_figure(
    hs: float,
    c_rv: float,
    c_rm: float,
    sigma3_mpa: float,
    water_pressure_mpa: float,
    f_jack: float,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    ax = axes[0]
    ax.set_title("Confinement cover criteria", fontsize=12)
    terrain_x = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
    terrain_y = np.array([3.0, 3.7, 3.2, 2.2, 1.8])
    ax.fill_between(terrain_x, terrain_y, 0.0, color=MONASH_GREY_3)
    ax.plot(terrain_x, terrain_y, color=MONASH_GREY_1, lw=2)
    tunnel = plt.Circle((3.0, 0.95), 0.28, fill=False, ec=MONASH_BLUEBERRY, lw=3)
    ax.add_patch(tunnel)
    ax.annotate("", xy=(3.0, 3.15), xytext=(3.0, 1.23), arrowprops=dict(arrowstyle="<->", lw=1.5, color=MONASH_GREY_1))
    ax.text(3.12, 2.2, f"C_RV = {metric_value(c_rv, ' m', 0)}", fontsize=9, va="center")
    ax.annotate("", xy=(4.55, 2.2), xytext=(3.22, 1.07), arrowprops=dict(arrowstyle="<->", lw=1.5, color=MONASH_BLUE))
    ax.text(4.05, 1.45, f"C_RM = {metric_value(c_rm, ' m', 0)}", fontsize=9, rotation=-33)
    ax.text(0.15, 0.22, f"Hydrostatic head h_s = {metric_value(hs, ' m', 0)}", fontsize=9)
    ax.set_xlim(-0.1, 6.1)
    ax.set_ylim(0.0, 4.1)
    ax.axis("off")

    ax = axes[1]
    ax.set_title("Hydraulic jacking screen", fontsize=12)
    ax.add_patch(plt.Circle((0.42, 0.50), 0.22, fill=False, ec=MONASH_BLUEBERRY, lw=3))
    ax.annotate("", xy=(0.42, 0.72), xytext=(0.42, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color=MONASH_BLUE))
    ax.annotate("", xy=(0.64, 0.50), xytext=(0.42, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color=MONASH_BLUE))
    ax.annotate("", xy=(0.20, 0.50), xytext=(0.42, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color=MONASH_BLUE))
    ax.text(0.42, 0.77, f"p_i = {metric_value(water_pressure_mpa, ' MPa', 2)}", ha="center", fontsize=9, color=MONASH_BLUE)
    ax.annotate("", xy=(0.42, 0.18), xytext=(0.42, 0.34), arrowprops=dict(arrowstyle="->", lw=2, color=MONASH_BLUEBERRY))
    ax.text(0.42, 0.10, f"sigma_3 = {metric_value(sigma3_mpa, ' MPa', 2)}", ha="center", fontsize=9, color=MONASH_BLUEBERRY)
    ax.text(0.42, 0.93, f"FoS = sigma_3 / p_i = {metric_value(f_jack, '', 2)}", ha="center", fontsize=10, color=MONASH_GREY_1)
    ax.text(0.42, 0.02, "If internal water pressure exceeds minimum in-situ stress,\njoints can open and the tunnel may need steel lining or realignment.", ha="center", fontsize=8.5)
    ax.set_xlim(0.0, 0.84)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.tight_layout()
    return fig


STEP_GUIDANCE = {
    1: {
        "equations": [
            ("Energy target", r"E_{target}=P_{rated}t_{target}"),
            ("Storage duration", r"t_{target}=\frac{E_{target}}{P_{rated}}"),
            ("Reconnaissance score", r"S_{recon}=\sum w_i r_i"),
            ("Gross head screen", r"H_g=z_{upper}-z_{lower}"),
            ("Route efficiency screen", r"S_{route}=\frac{H_g}{L_{route}}"),
        ],
        "symbols": [
            ("P_{rated}", "Installed generating power selected for the project brief", "MW"),
            ("t_{target}", "Target full-load generation duration", "h"),
            ("E_{target}", "Required deliverable storage energy", "MWh or GWh"),
            ("w_i, r_i", "Weight and rating for reconnaissance criteria such as data quality, grid access, environmental constraint, and constructability", "-"),
            ("z_{upper}, z_{lower}", "Candidate upper and lower reservoir operating levels used for screening", "m AHD"),
            ("L_{route}", "Approximate waterway route length from GIS", "m"),
            ("S_{route}", "Head-to-length screening ratio", "-"),
        ],
        "refs": [
            ("Reconnaissance scope", "Define reservoir-pair search area, data sources, map scale, vertical datum, data confidence, grid access, environmental constraints, and initial exclusion areas."),
            ("QGIS screening layers", "DEM, contours, map scale, watercourses, reservoirs, roads, transmission, protected areas, land use, heritage, satellite imagery, and data source confidence."),
            ("Evidence-based GIS screen", "Use source category, storage class, head, separation, constraint flags, data confidence, and relative cost rank as a desktop shortlist only."),
            ("Design basis", "State grid role, operating duration, water source, environmental limits, access, transmission, and intended level of accuracy."),
            ("Viability screen", "Before detailed design, identify whether engineering, economic, environmental, and approvals constraints make the scheme worth continuing."),
        ],
    },
    2: {
        "equations": [
            ("Storage opportunity", r"V_{active}\approx A_{res}\Delta z"),
            ("Representative gross head", r"H_{g,rep}=z_{u,rep}-z_{l,rep}"),
            ("Weighted concept score", r"Score=\sum w_i r_i"),
            ("Qualitative value-cost screen", r"\text{Compare distinct, compatible benefits with civil, electro-mechanical, grid, environmental and operating costs}"),
        ],
        "symbols": [
            ("A_{res}", "Reservoir footprint area from contours, DEM, or published storage curve", "m^2"),
            ("w_i", "Weight assigned to each concept criterion", "-"),
            ("r_i", "Rating for reservoir/dam concept against criterion i", "-"),
            ("Dam type", "Preliminary dam family selected from valley, foundation, material, and storage evidence", "-"),
            ("Economic evidence", "Qualitative comparison at reconnaissance level; a numerical ratio requires dated, consistent cost and benefit bases", "-"),
        ],
        "refs": [
            ("Reservoir-pair mapping", "Compare at least two candidate reservoir pairs using head, storage opportunity, route length, grid distance, access, environmental constraints, and confidence rating."),
            ("Candidate fields", "Record source category, coordinates, storage class, head, separation, dam geometry, water-to-rock ratio, cost rank, constraint flag, and data confidence."),
            ("Reconnaissance outputs", "Shortlist dam site, powerhouse site, waterway corridor, construction access, and initial maximum plant discharge range for later steps."),
            ("Dam type screening", "Earthfill, rockfill, CFRD, RCC/gravity, arch, ring-dike, or existing dam reuse."),
            ("Alternative optimisation", "Compare dam site, powerhouse site, waterway route, maximum plant discharge, storage duration, construction cost, cost per kW/kWh, and sourced economic evidence."),
            ("Evidence", "Valley geometry, foundation quality, local materials, constructability, storage curve, grid connection, environmental constraints, and data confidence."),
        ],
    },
    3: {
        "equations": [
            ("Available drawdown", r"H_a=HWL-LWL"),
            ("Teaching representative upper level", r"z_{u,rep}=HWL_u-\frac{HWL_u-LWL_u}{3}\quad\text{(placeholder only)}"),
            ("Representative gross head", r"H_{g,rep}=z_{u,rep}-z_{l,rep}"),
            ("Simultaneous head envelope", r"H_{g,max}=HWL_u-LWL_l,\qquad H_{g,min}=LWL_u-HWL_l"),
            ("Head-range ratio", r"HFR=\frac{H_{g,min}}{H_{g,max}}"),
            ("Crest level", r"RL_{crest}=HWL+F_b+H_w+S_a"),
            ("Dam height", r"H_{dam}=RL_{crest}-RL_{foundation}"),
            ("Gross-head storage energy", r"E_g=\rho g V H_g\eta/3.6\times10^{12}"),
            ("Deliverable storage energy", r"E_{del}=\rho g V H_e\eta/3.6\times10^{12}"),
        ],
        "symbols": [
            ("HWL", "High water level", "m"),
            ("H_a", "Available operating drawdown between HWL and LWL", "m"),
            ("z_{u,rep}", "Representative upper level; the one-third-drawdown value is a course placeholder unless supported by an operating rule or storage curve", "m"),
            ("LWL", "Low water level", "m"),
            ("H_{g,rep}", "Representative gross head from explicitly stated representative upper and lower levels", "m"),
            ("H_{g,min}, H_{g,max}", "Simultaneous minimum and maximum gross operating heads", "m"),
            ("HFR", "Minimum-to-maximum simultaneous gross-head ratio", "-"),
            ("F_b, H_w, S_a", "Freeboard, wave allowance, and settlement allowance", "m"),
            ("V", "Active storage volume", "m^3"),
            ("H_e", "Effective head after first-pass allowance for hydraulic losses and operating head effects", "m"),
        ],
        "refs": [
            ("Reservoir storage", "Use QGIS contours or storage-area curves to justify active area and depth."),
            ("Desktop civil screen", "Report storage GWh, usable GL, reservoir area, dam volume, and water-to-rock ratio for comparison with mapped candidates."),
            ("Dam sizing", "Treat app outputs as concept screening; detailed dam design needs hydrology, geology, and dam-safety design."),
        ],
    },
    4: {
        "equations": [
            ("Waterway alignment length", r"L_w=\sum \sqrt{(\Delta x)^2+(\Delta z_w)^2}"),
            ("Turbine centreline", r"CL=TWL_{lower}-h_{set}"),
            ("Average gradient", r"i=\frac{\Delta z}{L}"),
            ("Power-target sizing discharge", r"Q_{target}=\frac{P\times10^6}{\rho g H\eta},\quad H=H_g\ \mathrm{or\ justified}\ H_e"),
            ("Declared operating discharge", r"Q=Q_{declared}\quad\text{with source and operating condition}"),
            ("Average discharge from annual energy", r"Q_{ave}=\frac{E_{annual}}{\rho g H\eta T}"),
            ("Maximum discharge from capacity factor", r"Q_{max}=\frac{Q_{ave}}{CF}"),
            ("Required active storage", r"V_{req}=Q t 3600"),
            ("Duration from storage", r"t=\frac{V}{Q3600}"),
        ],
        "symbols": [
            ("L_w", "Waterway alignment/profile length used in hydraulics", "m"),
            (r"\Delta x", "Horizontal chainage increment", "m"),
            (r"\Delta z_w", "Elevation change along selected waterway alignment, not ground surface", "m"),
            ("CL", "Turbine centreline elevation", "m"),
            ("h_{set}", "Turbine centreline setting below the stated tailwater condition; geometry/cavitation input, not a hydraulic loss", "m"),
            ("i", "Average waterway gradient", "-"),
            ("Q", "Selected total plant operating discharge: target-derived or explicitly declared", "m^3/s"),
            ("H", "Head used in Step 4 discharge: gross head or a justified first-pass effective head", "m"),
            (r"\eta", "Efficiency assumption used for first-pass sizing", "-"),
            ("V_{req}", "Required active storage volume for selected duration", "m^3"),
            ("CF", "Capacity factor used to convert average discharge to maximum/design discharge", "-"),
        ],
        "refs": [
            ("Layout evidence", "Plan, long section, intake, headrace, pressure shaft, powerhouse, tailrace, access adits, spoil areas."),
            ("QGIS output", "Show both ground topography and selected waterway alignment. The tunnel can be below ground and should not simply follow the terrain."),
            ("Teaching check", "Use total plant discharge first; divide by units only after selecting unit branches."),
            ("Snowy-style scale", "Large PHES cases can have total discharge in the hundreds of m^3/s."),
        ],
    },
    5: {
        "equations": [
            ("Flow area", r"A=\frac{\pi D^2}{4}"),
            ("Velocity", r"v=\frac{Q_v}{A}"),
            ("Reynolds number", r"Re=\frac{vD}{\nu}"),
            ("Relative roughness", r"\epsilon_r=\frac{\epsilon}{D}"),
            ("Darcy friction", r"f=\frac{0.25}{\left[\log_{10}\left(\epsilon_r/3.7+5.74/Re^{0.9}\right)\right]^2}"),
            ("Major and minor losses", r"h_{major}=f\frac{L}{D}\frac{v^2}{2g},\qquad h_{minor}=\Sigma K\frac{v^2}{2g}"),
            ("Net head", r"H_e=H_{sel}-h_{major}-h_{minor}-h_{other}"),
        ],
        "symbols": [
            ("D", "Shared conduit diameter", "m"),
            ("Q_v", "Flow used for the velocity calculation: total plant, per-unit, per-penstock, or custom flow", "m^3/s"),
            ("v", "Flow velocity in selected conduit", "m/s"),
            ("Re", "Reynolds number used to identify laminar/transitional/turbulent flow", "-"),
            (r"\epsilon", "Absolute roughness for selected lining or pipe material", "m"),
            ("f", "Darcy friction factor", "-"),
            (r"\Sigma K", "Sum of local loss coefficients", "-"),
            ("h_{other}", "Separately declared hydraulic loss not represented by Darcy friction or local-loss coefficients", "m"),
        ],
        "refs": [
            ("Velocity guidance", "Use about 4-6 m/s as a teaching range, then test sensitivity."),
            ("Darcy f selection", "Use f = 64/Re for laminar flow, review transitional flow, and use Swamee-Jain/Colebrook for turbulent PHES flow."),
            ("Local losses", "Select actual components and quantities rather than adding an unknown manual K."),
        ],
    },
    6: {
        "equations": [
            ("Unit branch diameter", r"D_u=\sqrt{\frac{4Q_u}{\pi v_u}}"),
            ("Empirical diameter check", r"D\approx0.802Q_u^{0.437}"),
            ("Head-loss budget", r"h_{f,allow}\approx(0.05\text{--}0.10)H_g"),
            ("Unit branch loss", r"h_{u}=\left(f\frac{L_u}{D_u}+K_u\right)\frac{v_u^2}{2g}"),
            ("Water-hammer screen", r"\Delta H\approx\frac{a v_u}{g}"),
        ],
        "symbols": [
            ("Q_u", "Discharge per unit branch or penstock", "m^3/s"),
            ("v_u", "Candidate unit branch velocity", "m/s"),
            ("D_u", "Candidate unit branch diameter", "m"),
            ("L_u", "Unit branch centreline length from bifurcation/manifold to turbine inlet", "m"),
            ("K_u", "Sum of local loss coefficients in the unit branch only", "-"),
            ("a", "Pressure-wave speed", "m/s"),
            ("h_{f,allow}", "Screening head-loss budget for diameter verification", "m"),
        ],
        "refs": [
            ("Velocity selection", "Compare 3, 4, 5, 6, and 7 m/s against diameter, loss, transient risk, and constructability."),
            ("Branch length", "Use the Step 4 long-section chainage for the individual unit branch; if missing, use a sensitivity range and state that the branch is not yet drawn."),
            ("Local loss coefficient", "Build K_u from the actual branch, valve, bends, reducer/transition and turbine-inlet components; avoid double counting shared-conduit losses from Step 5."),
            ("Wave speed", "Choose a from pipe/tunnel material and lining stiffness; use low/base/high sensitivity for concept water-hammer checks."),
            ("Design decision", "Select velocity with hydraulic, civil, valve, bifurcation, and cost justification."),
        ],
    },
    7: {
        "equations": [
            ("Vertical cover criterion", r"F_{RV}=\frac{C_{RV}\gamma_r\cos\alpha}{h_s\gamma_w}"),
            ("Minimum cover criterion", r"F_{RM}=\frac{C_{RM}\gamma_r\cos\beta}{h_s\gamma_w}"),
            ("Hydraulic jacking screen", r"F_{jack}=\frac{\sigma_3}{\gamma_w h_s}\ge 1.3,\quad \sigma_3\approx k_{min}\gamma_r C_{RM}"),
            ("Lining stress", r"\sigma_\theta=\frac{p_i r_i^2-p_e r_o^2}{r_o^2-r_i^2}+\frac{(p_i-p_e)r_i^2r_o^2}{(r_o^2-r_i^2)r^2}"),
            ("Cavern crown", r"z_{crown}=CL+H_{MH}"),
            ("In-situ stress", r"\sigma_v=\gamma h_c,\quad \sigma_h=k\sigma_v"),
        ],
        "symbols": [
            ("C_{RV}, C_{RM}", "Given vertical and minimum rock cover", "m"),
            (r"\gamma_r, \gamma_w", "Rock and water unit weight", "kN/m^3"),
            ("h_s", "Hydrostatic head at tunnel/cavern", "m"),
            (r"\sigma_3, k_{min}", "Minimum in-situ principal stress and its ratio to vertical stress, used for the hydraulic jacking screen", "MPa, -"),
            ("F_{jack}", "Jacking factor of safety: minimum stress divided by internal water pressure", "-"),
            (r"\sigma_\theta", "Hoop stress in lining", "MPa"),
            ("CL", "Turbine centreline elevation used to locate powerhouse cavern", "m"),
            ("H_{MH}", "Machine hall height", "m"),
            ("IPB", "Isolated phase bus gallery connecting machines and transformers", "-"),
        ],
        "refs": [
            ("Geotechnical evidence", "Cover, rock mass quality, in-situ stress, faults, excavation method, support class."),
            ("Lining selection", "Combine the cover and jacking screens to classify each waterway zone as unlined/shotcrete, reinforced concrete with grouting, or steel-lined."),
            ("Cavern sizing", "Report machine hall, transformer hall, IPB gallery, access, ventilation, drainage, and egress."),
        ],
    },
    8: {
        "equations": [
            ("Connected flow area", r"A_{conn}=n\frac{\pi D^2}{4}"),
            ("Event velocity", r"v_0=\frac{Q_0}{A_{conn}}"),
            ("Round-trip wave time", r"T_w=\frac{2L}{a}"),
            ("Rapid-closure severity bound", r"\Delta H_J=\frac{a v_0}{g}\quad(T_c\le T_w)"),
            ("Slower-closure rigid-column screen", r"\Delta H_{RC}=\frac{2Lv_0}{gT_c}\quad(T_c>T_w)"),
        ],
        "symbols": [
            ("A_{conn}", "Total conduit area hydraulically connected to the screened event", "m^2"),
            ("T_c", "Valve or guide-vane closure time", "s"),
            ("T_w", "Round-trip wave time separating rapid and slower closure screens", "s"),
            (r"\Delta H_J,\Delta H_{RC}", "Severity screens, not predicted plant pressures", "m"),
        ],
        "refs": [
            ("Transient study", "The app provides severity bounds only; final design needs a connected-system method-of-characteristics or equivalent transient model."),
            ("Surge control", "Do not size a surge structure from a universal area ratio. Select its type and dimensions from cited stability, pressure, minimum-level and operational criteria."),
            ("Controls", "Model valve/guide-vane laws, rotating inertia, load rejection, pump trip/reversal, runaway, reservoir boundaries, maximum pressure and minimum pressure."),
        ],
    },
    9: {
        "equations": [
            ("Hydraulic power", r"P_h=\rho g QH/10^6"),
            ("Generated power", r"P_g=\rho g QH\eta_{total}/10^6"),
            ("Pumping power", r"P_p=\frac{\rho gQH}{\eta_p10^6}"),
            ("Unit flow and power", r"Q_u=\frac{Q_{total}}{N},\quad P_u=\frac{P_{design}}{N}"),
            ("Flow-based specific speed", r"n_q=\frac{N\sqrt{Q_u}}{H^{3/4}}"),
            ("Power-based convention", r"N_s=\frac{N\sqrt{P_u}}{H^{5/4}}"),
        ],
        "symbols": [
            ("Q", "Operating discharge used for turbine/pump selection", "m^3/s"),
            ("H", "Operating head used for turbine/pump selection", "m"),
            ("N", "Runner rotational speed", "rpm"),
            (r"\eta_{total}", "Turbine, generator, and transformer efficiency chain", "-"),
            ("Q/Q_{max}", "Part-load operating ratio", "-"),
            ("Q_u, P_u", "Unit-level discharge and power used for manufacturability and specific-speed checks", "m^3/s, MW"),
        ],
        "refs": [
            ("Turbine application table", "Pelton: high head/low flow; Francis: medium-high head and flow; Kaplan/bulb: low head/high flow."),
            ("Operating envelope", "Check generation, pumping, part-load efficiency, unit count, and reversible pump-turbine suitability."),
        ],
    },
    10: {
        "equations": [
            ("Approximate cycle efficiency", r"\eta_{cycle}=\eta_{\mathrm{gen\,chain}}\eta_p"),
            ("Pumping energy", r"E_{pump}=\frac{E_{delivered}}{\eta_{cycle}}"),
            ("Daily generation", r"E_{gen,day}=P_{gen}t_{gen,day}"),
            ("Daily pumping", r"E_{pump,day}=P_{pump}t_{pump,day}"),
            ("Cost per kW", r"C_{kW}=\frac{C_{capital}}{P_{rated}}"),
            ("Cost per kWh", r"C_{kWh}=\frac{C_{capital}}{E_{delivered}}"),
            ("Discounted benefit-cost ratio", r"BCR=\frac{\sum_t B_t/(1+r)^t}{\sum_t C_t/(1+r)^t}"),
            ("App undiscounted proxy", r"R_{undisc}=\frac{B_{annual}N}{C_{capital}+O\&M_{annual}N}"),
            ("Risk rating", r"R_{risk}=Likelihood\times Consequence"),
        ],
        "symbols": [
            (r"\eta_{cycle}", "Approximate round-trip cycle efficiency", "-"),
            ("E_{delivered}", "Deliverable generation energy", "GWh"),
            ("E_{pump}", "Energy required for pumping cycle", "GWh"),
            ("NEM", "National Electricity Market or equivalent grid-market context", "-"),
            ("C_{capital}", "Screening capital cost including civil, electro-mechanical, grid connection, owner, environmental and contingency costs", "$"),
            ("O&M", "Annual operation and maintenance cost assumption", "$/year or % of capital cost"),
        ],
        "refs": [
            ("Market role", "Discuss renewable firming, peak capacity, reserve, FCAS, black start, transmission/grid connection and network constraints."),
            ("Relative cost rank", "Use AAA to E or a similar class as a desktop benchmark only; final economics need project-specific quantities and market costs."),
            ("Economic screen", "Document source date, scope and price basis for cost and benefit inputs; avoid double counting energy, capacity and services. Use a discounted model for a decision claim."),
            ("Final recommendation", "State proceed, proceed with conditions, or reject, with key risks, environmental monitoring commitments and next investigations."),
        ],
    },
}


def render_step_header(step: int, page: str) -> None:
    full_title = page.split(". ", 1)[1] if ". " in page else page
    short_title = STEP_SHORT_TITLES.get(str(step), full_title)
    st.markdown(
        f"""
        <section class="step-heading">
            <div class="step-eyebrow">PHES design workflow / Step {step:02d}</div>
            <h1>{short_title}</h1>
            <div class="step-subtitle">{full_title}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


TEXT_ANNOTATION_TOKENS = {"Dam type", "NEM", "O&M", "IPB"}


def latex_text_cell(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def latex_annotation_expr(value: object) -> str:
    text = str(value).strip()
    if not text or text == "-":
        return r"\text{-}"
    if text in TEXT_ANNOTATION_TOKENS or (" " in text and not any(marker in text for marker in ("\\", "_", "{", "}", "^", "/", ","))):
        return rf"\text{{{latex_text_cell(text)}}}"
    return text


def latex_unit_part(part: str) -> str:
    text = part.strip()
    if not text or text == "-":
        return r"\text{-}"
    if any(token in text for token in (" ", "$", "%")):
        return rf"\text{{{latex_text_cell(text)}}}"
    return rf"\mathrm{{{text}}}"


def latex_unit_expr(value: object) -> str:
    text = str(value).strip()
    if "," in text:
        return r",\ ".join(latex_unit_part(part) for part in text.split(","))
    return latex_unit_part(text)


def markdown_table_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def render_annotation_table(symbol_rows: list[tuple[str, str, str]]) -> None:
    rows = [
        "| Annotation | Definition | Unit |",
        "|---|---|---|",
    ]
    for symbol, definition, unit in symbol_rows:
        rows.append(
            f"| ${latex_annotation_expr(symbol)}$ | {markdown_table_cell(definition)} | ${latex_unit_expr(unit)}$ |"
        )
    st.markdown("\n".join(rows))


def render_symbol_value_card(label_html: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="symbol-value-card">
            <div class="symbol-value-label">{label_html}</div>
            <div class="symbol-value-number">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


SECTION_LABELS = {
    "input": "INPUT",
    "output": "OUTPUT",
    "reference": "REFERENCE",
    "check": "CHECK",
}


@contextmanager
def section_panel(kind: str, title: str, note: str | None = None):
    label = SECTION_LABELS.get(kind, kind.upper())
    marker_class = f"section-{kind}-marker"
    with st.container(border=True):
        st.markdown(
            f"""
            <span class="section-marker {marker_class}"></span>
            <div class="section-title-row">
                <span class="section-badge section-badge-{html.escape(kind)}">{html.escape(label)}</span>
                <h2>{html.escape(title)}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if note:
            st.caption(note)
        yield


def render_step_guidance(step: int) -> None:
    guide = STEP_GUIDANCE.get(step)
    if not guide:
        return

    with st.expander("Equations, annotations, and reference table"):
        tab_eq, tab_symbols, tab_refs = st.tabs(["Equations", "Annotations", "Reference table"])
        with tab_eq:
            for label, equation in guide["equations"]:
                st.markdown(f"**{label}**")
                st.latex(equation)
        with tab_symbols:
            render_annotation_table(guide["symbols"])
        with tab_refs:
            st.dataframe(
                pd.DataFrame(guide["refs"], columns=["Reference item", "Guideline use"]),
                hide_index=True,
                width="stretch",
            )


LATEX_INLINE_TOKEN_REPLACEMENTS = (
    ("σ₃/pᵢ", r"$\sigma_3/p_i$"),
    ("σ₃/σᵥ", r"$\sigma_3/\sigma_v$"),
    ("ΣK", r"$\Sigma K$"),
    ("ΔH", r"$\Delta H$"),
    ("Δh", r"$\Delta h$"),
    ("η", r"$\eta$"),
    ("ρ", r"$\rho$"),
    ("ν", r"$\nu$"),
    ("ε", r"$\varepsilon$"),
    ("α", r"$\alpha$"),
    ("β", r"$\beta$"),
    ("γ", r"$\gamma$"),
    ("σ", r"$\sigma$"),
    ("π", r"$\pi$"),
    ("₀", r"$_0$"),
    ("₁", r"$_1$"),
    ("₂", r"$_2$"),
    ("₃", r"$_3$"),
    ("₄", r"$_4$"),
    ("₅", r"$_5$"),
    ("₆", r"$_6$"),
    ("₇", r"$_7$"),
    ("₈", r"$_8$"),
    ("₉", r"$_9$"),
    ("ₐ", r"$_a$"),
    ("ₑ", r"$_e$"),
    ("ₕ", r"$_h$"),
    ("ᵢ", r"$_i$"),
    ("ₚ", r"$_p$"),
    ("ᵣ", r"$_r$"),
    ("ₛ", r"$_s$"),
    ("ₜ", r"$_t$"),
    ("²", r"$^2$"),
    ("³", r"$^3$"),
    ("≤", r"$\le$"),
    ("≥", r"$\ge$"),
    ("≈", r"$\approx$"),
    ("×", r"$\times$"),
    ("°", r"$^\circ$"),
    ("→", r"$\rightarrow$"),
    ("✓", r"$\checkmark$"),
    ("–", "--"),
    ("—", "---"),
    ("−", "-"),
)

LATEX_MOJIBAKE_HINTS = ("Ã", "Â", "Î", "Ï", "â")


def repair_mojibake(text: str) -> str:
    if not any(hint in text for hint in LATEX_MOJIBAKE_HINTS):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired if repaired.count("\ufffd") <= text.count("\ufffd") else text


def latex_escape(value: object) -> str:
    text = repair_mojibake(str(value))
    placeholders: dict[str, str] = {}
    for index, (token, replacement) in enumerate(LATEX_INLINE_TOKEN_REPLACEMENTS):
        if token in text:
            placeholder = f"@@LATEX_TOKEN_{index}@@"
            text = text.replace(token, placeholder)
            placeholders[placeholder] = replacement
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(replacements.get(char, char) for char in text)
    for placeholder, replacement in placeholders.items():
        escaped = escaped.replace(placeholder, replacement)
    return escaped


def safe_file_stem(value: object, fallback: str = "phes_design") -> str:
    text = str(value).strip().lower()
    chars = []
    last_was_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_was_sep = False
        elif not last_was_sep:
            chars.append("_")
            last_was_sep = True
    stem = "".join(chars).strip("_")
    return stem or fallback


def latex_metric(value: float, unit: str = "", digits: int = 1) -> str:
    return latex_escape(metric_value(value, f" {unit}" if unit else "", digits))


def report_logic_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        REPORT_LOGIC_ROWS,
        columns=["Step", "Report section", "Design logic", "Key equations / checks", "Evidence expected"],
    )


# Typeset equivalents of the plain-text "Key equations / checks" column in
# REPORT_LOGIC_ROWS. The plain text stays for the in-app dataframe (which
# cannot render LaTeX); the report export uses these display equations.
# Each entry: (list of display equations, optional trailing prose note).
REPORT_LOGIC_EQUATIONS = {
    "Step 1": (
        [
            r"E_{target}=P_{design}\,t_{target}",
            r"H_g=z_{upper}-z_{lower},\qquad S_{route}=\frac{H_g}{L_{route}}",
        ],
        "Continue or reject based on head, storage, waterway length, grid access and approvals risk.",
    ),
    "Step 2": (
        [
            r"H_{g,rep}=z_{u,rep}-z_{l,rep},\qquad V_{active}\approx\sum_i A_i\,\Delta z_i",
        ],
        "Route length from chainage/profile; qualitative multi-criteria comparison; dam type follows valley geometry, foundation quality, materials, dam height and storage volume.",
    ),
    "Step 3": (
        [
            r"H_a=HWL-LWL,\qquad z_{u,rep}=HWL_u-\frac{HWL_u-LWL_u}{3}\quad\text{(teaching placeholder)}",
            r"H_{g,rep}=z_{u,rep}-z_{l,rep}",
            r"H_{g,max}=HWL_u-LWL_l,\qquad H_{g,min}=LWL_u-HWL_l,\qquad HFR=\frac{H_{g,min}}{H_{g,max}}",
            r"RL_{crest}=HWL+F_b+H_w+S_a,\qquad H_{dam}=RL_{crest}-RL_{foundation}",
            r"E=\frac{\rho g V H_e \eta}{3.6\times10^{12}}",
        ],
        "",
    ),
    "Step 4": (
        [
            r"L_{profile}=\sum_i\sqrt{(\Delta x_i)^2+(\Delta z_i)^2}",
            r"CL=TWL_{lower}-h_{set}",
            r"Q_{target}=\frac{P\times10^6}{\rho g H \eta},\qquad Q=Q_{target}\ \text{or sourced }Q_{declared}",
            r"V_{req}=Q\,t\times3600,\qquad t=\frac{V}{3600\,Q}",
        ],
        "",
    ),
    "Step 5": (
        [
            r"A=\frac{\pi D^2}{4},\qquad v=\frac{Q_v}{A},\qquad Re=\frac{vD}{\nu}",
            r"h_{major}=f\frac{L}{D}\frac{v^2}{2g},\quad h_{minor}=\Sigma K\frac{v^2}{2g},\quad H_e=H_{sel}-h_{major}-h_{minor}-h_{other}",
        ],
        "",
    ),
    "Step 6": (
        [
            r"Q_u=\frac{Q_{total}}{N_{units}},\qquad D_u=\sqrt{\frac{4Q_u}{\pi v_u}}",
            r"h_u=\left(f\frac{L_u}{D_u}+K_u\right)\frac{v_u^2}{2g},\qquad \Delta H_{inst}=\frac{a\,v_u}{g}",
        ],
        "",
    ),
    "Step 7": (
        [
            r"p=\frac{\rho g H}{10^6},\qquad \sigma_v=\gamma h_c,\qquad \sigma_h=k\,\sigma_v",
            r"F_{RV}=\frac{C_{RV}\gamma_r\cos\alpha}{h_s\gamma_w},\qquad F_{RM}=\frac{C_{RM}\gamma_r\cos\beta}{h_s\gamma_w}",
            r"F_{jack}=\frac{\sigma_3}{\gamma_w h_s}\ge 1.3",
        ],
        "Lining class follows from the combined cover and jacking screens; apply the hoop-stress check where a lining is required.",
    ),
    "Step 8": (
        [
            r"A_{conn}=n\frac{\pi D^2}{4},\qquad v_0=\frac{Q_0}{A_{conn}},\qquad T_w=\frac{2L}{a}",
            r"\Delta H_J=\frac{a v_0}{g}\ (T_c\le T_w),\qquad \Delta H_{RC}=\frac{2Lv_0}{gT_c}\ (T_c>T_w)",
        ],
        "Severity bounds only; surge-control sizing requires cited criteria and a connected-system transient model.",
    ),
    "Step 9": (
        [
            r"P_h=\frac{\rho g Q H}{10^6},\qquad P_{gen}=\frac{\rho g Q H\,\eta_{total}}{10^6}",
            r"P_{pump}=\frac{\rho g Q H}{\eta_p 10^6},\qquad n_q=\frac{N\sqrt{Q_u}}{H^{3/4}}",
        ],
        "",
    ),
    "Step 10": (
        [
            r"\eta_{cycle}=\eta_{gen}\,\eta_{pump},\qquad E_{pump}=\frac{E_{delivered}}{\eta_{cycle}}",
            r"C_{kW}=\frac{C_{capital}}{P_{rated}},\qquad C_{kWh}=\frac{C_{capital}}{E_{delivered}}",
            r"BCR=\frac{\sum_t B_t/(1+r)^t}{\sum_t C_t/(1+r)^t},\qquad R_{undisc}=\frac{B_{annual}N}{C_{capital}+C_{O\&M,annual}N}",
        ],
        "",
    ),
}


def report_logic_latex_blocks() -> str:
    blocks = []
    for step, section, logic, equations, evidence in REPORT_LOGIC_ROWS:
        latex_equations, note = REPORT_LOGIC_EQUATIONS.get(step, ([], ""))
        if latex_equations:
            note_line = f"\n{latex_escape(note)}" if note else ""
            equation_item = latex_equation_blocks(latex_equations) + note_line
        else:
            equation_item = latex_escape(equations)
        blocks.append(
            f"""\\subsection*{{{latex_escape(step)}. {latex_escape(section)}}}
\\begin{{description}}
\\item[Design logic:] {latex_escape(logic)}
\\item[Key equations/checks:]
{equation_item}
\\item[Evidence expected:] {latex_escape(evidence)}
\\end{{description}}"""
        )
    return "\n\n".join(blocks)


def latex_itemize(items: list[str]) -> str:
    lines = ["\\begin{itemize}"]
    lines.extend(f"\\item {latex_escape(item)}" for item in items)
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def latex_description(items: list[tuple[str, str]]) -> str:
    lines = ["\\begin{description}"]
    lines.extend(f"\\item[{latex_escape(label)}:] {latex_escape(text)}" for label, text in items)
    lines.append("\\end{description}")
    return "\n".join(lines)


def latex_equation_blocks(equations: list[str]) -> str:
    return "\n".join(f"\\[\n{equation}\n\\]" for equation in equations)


# Shared preamble matching the RSE3241 study-note / handout LaTeX format.
LATEX_STUDY_NOTE_PREAMBLE = r"""\documentclass[a4paper,12pt]{article}

% ======================================================================
% PACKAGES
% ======================================================================
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{array}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{parskip}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{hyperref}
\usepackage{tabularx}
\usepackage[table]{xcolor}
\usepackage{titlesec}
\usepackage[most]{tcolorbox}
\usepackage{float}
\usepackage{graphicx}
\usepackage{siunitx}
% Rollback: longtable v4.24 (2025-10-13) on this MiKTeX errors with
% "Infinite glue shrinkage" whenever a longtable splits across pages.
% Remove the [=v4.13] once MiKTeX packages are updated past v4.24.
\usepackage{longtable}[=v4.13]
\usepackage{pdflscape}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{makecell}
\usepackage{microtype}

\geometry{margin=2.5cm}
\definecolor{MonashBlue}{cmyk}{1.00,0.50,0.05,0.05}
\definecolor{MonashBlack}{cmyk}{0.00,0.00,0.00,1.00}
\definecolor{MonashGreyOne}{cmyk}{0.00,0.00,0.00,0.80}
\definecolor{MonashGreyTwo}{cmyk}{0.00,0.00,0.00,0.50}
\definecolor{MonashGreyThree}{cmyk}{0.00,0.00,0.00,0.10}
\definecolor{MonashOrange}{cmyk}{0.00,0.64,0.94,0.00}
\definecolor{MonashGreen}{cmyk}{0.56,0.16,1.00,0.00}
\hypersetup{
    colorlinks=true,
    linkcolor=MonashBlue,
    urlcolor=MonashBlue,
    citecolor=MonashBlue
}

\setlist[itemize]{topsep=3pt,itemsep=2pt,parsep=0pt}
\setlist[enumerate]{topsep=3pt,itemsep=2pt,parsep=0pt}
\captionsetup{font=small,labelfont={bf,color=MonashBlue}}
\renewcommand{\arraystretch}{1.16}
\setlength{\tabcolsep}{5pt}
\setlength{\headheight}{15pt}
\rowcolors{2}{white}{MonashGreyThree}
\arrayrulecolor{MonashBlue}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcommand{\MonashTableHeader}{\rowcolor{MonashBlue}}
\newcommand{\MonashHeaderCell}[1]{{\color{white}\bfseries #1}}
\newcommand{\Hnet}{H_{\mathrm{net}}}
\newcommand{\Hgross}{H_{\mathrm{g}}}
\newcommand{\Heff}{H_{\mathrm{e}}}
\newcommand{\etal}{\eta_{\mathrm{overall}}}
\newcommand{\etap}{\eta_{\mathrm{pump}}}
\newcommand{\etart}{\eta_{\mathrm{round\mbox{-}trip}}}

\titleformat{\section}{\Large\bfseries\color{MonashBlue}}{\thesection}{0.75em}{}
\titleformat{\subsection}{\large\bfseries\color{MonashBlue}}{\thesubsection}{0.75em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{MonashBlue}}{\thesubsubsection}{0.75em}{}

% ======================================================================
% BOX STYLES
% ======================================================================
\newtcolorbox{notebox}[1][]{
    enhanced,
    breakable,
    colback=white,
    colframe=MonashBlue,
    colbacktitle=MonashBlue,
    coltitle=white,
    fonttitle=\bfseries,
    title=#1,
    boxrule=0.6pt,
    arc=2mm,
    left=2mm,right=2mm,top=1mm,bottom=1mm
}

\newtcolorbox{warnbox}[1][]{
    enhanced,
    breakable,
    colback=white,
    colframe=MonashOrange,
    colbacktitle=MonashOrange,
    coltitle=white,
    fonttitle=\bfseries,
    title=#1,
    boxrule=0.6pt,
    arc=2mm,
    left=2mm,right=2mm,top=1mm,bottom=1mm
}

\newtcolorbox{keybox}[1][]{
    enhanced,
    breakable,
    colback=white,
    colframe=MonashBlue,
    colbacktitle=MonashBlue,
    coltitle=white,
    fonttitle=\bfseries,
    title=#1,
    boxrule=0.6pt,
    arc=2mm,
    left=2mm,right=2mm,top=1mm,bottom=1mm
}

\newtcolorbox{assignmentbox}[1][]{
    enhanced,
    breakable,
    colback=white,
    colframe=MonashBlue,
    colbacktitle=MonashBlue,
    coltitle=white,
    fonttitle=\bfseries,
    title=#1,
    boxrule=0.6pt,
    arc=2mm,
    left=2mm,right=2mm,top=1mm,bottom=1mm
}

\newtcolorbox{rubricbox}[1][]{
    enhanced,
    breakable,
    colback=white,
    colframe=MonashGreyOne,
    colbacktitle=MonashGreyOne,
    coltitle=white,
    fonttitle=\bfseries,
    title=#1,
    boxrule=0.6pt,
    arc=2mm,
    left=2mm,right=2mm,top=1mm,bottom=1mm
}
"""


def latex_study_note_document(
    doc_subtitle: str,
    rhead: str,
    date_text: str,
    core_box_title: str,
    core_box_text: str,
    body: str,
    include_toc: bool = True,
) -> str:
    """Wrap a body in the RSE3241 study-note document format (header/footer,
    title block, core-message keybox, and optional table of contents)."""
    toc_block = "\n\\newpage\n\\tableofcontents\n\\newpage\n\n" if include_toc else "\n\\vspace{0.4cm}\n\n"
    return (
        LATEX_STUDY_NOTE_PREAMBLE
        + "\n% ======================================================================\n"
        + "% HEADER AND FOOTER\n"
        + "% ======================================================================\n"
        + "\\pagestyle{fancy}\n"
        + "\\fancyhf{}\n"
        + "\\lhead{RSE3241 Hydropower}\n"
        + "\\rhead{" + latex_escape(rhead) + "}\n"
        + "\\cfoot{\\thepage\\ of \\pageref{LastPage}}\n"
        + "\\renewcommand{\\headrulewidth}{0.4pt}\n"
        + "\\renewcommand{\\footrulewidth}{0.2pt}\n\n"
        + "% ======================================================================\n"
        + "% TITLE\n"
        + "% ======================================================================\n"
        + "\\title{\\color{MonashBlue}\\textbf{RSE3241: Hydropower}\\\\[0.45em]\n"
        + doc_subtitle + "}\n"
        + "\\author{Monash University, Semester 2, 2026}\n"
        + "\\date{" + latex_escape(date_text) + "}\n\n"
        + "\\begin{document}\n\n"
        + "\\maketitle\n"
        + "\\thispagestyle{empty}\n"
        + "\\vspace{0.3cm}\n\n"
        + "\\begin{keybox}[" + latex_escape(core_box_title) + "]\n"
        + latex_escape(core_box_text) + "\n"
        + "\\end{keybox}\n"
        + toc_block
        + body
        + "\n\\end{document}\n"
    )


def report_step_section(
    number: int,
    title: str,
    purpose: str,
    method_items: list[str],
    equations: list[str],
    outputs: list[tuple[str, str]],
    evidence_items: list[str],
    discussion_items: list[str],
    figure: tuple[str, str] | None = None,
) -> str:
    figure_block = ""
    if figure is not None:
        filename, caption = figure
        figure_block = f"""
\\subsection*{{Report figure}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\linewidth]{{{filename}}}
\\caption{{{latex_escape(caption)}}}
\\end{{figure}}
"""
    return f"""\\section{{Step {number}. {latex_escape(title)}}}

\\begin{{notebox}}[Purpose and design decision]
{latex_escape(purpose)}
\\end{{notebox}}

\\subsection*{{Method and calculation logic}}
{latex_itemize(method_items)}

\\subsection*{{Equations and checks}}
{latex_equation_blocks(equations)}

\\subsection*{{Current Streamlit outputs to report}}
{latex_description(outputs)}
{figure_block}
\\begin{{assignmentbox}}[Evidence to insert]
{latex_itemize(evidence_items)}
\\end{{assignmentbox}}

\\begin{{rubricbox}}[Interpretation and discussion required]
{latex_itemize(discussion_items)}
\\end{{rubricbox}}
"""


def build_extended_report_sections(
    levels: dict[str, float],
    design_case: dict[str, float],
    market_role: str,
    figure_files: frozenset[str] | set[str] = frozenset(),
) -> str:
    eta_gen = st.session_state.eta_turbine * st.session_state.eta_generator * st.session_state.eta_transformer
    eta_cycle = eta_gen * st.session_state.eta_pump
    q_design = selected_design_discharge(fnum('teaching_effective_head_m'))
    q_unit = safe_div(q_design, st.session_state.units)
    active_depth = max(levels["upper_hwl"] - levels["upper_lwl"], 0.0)
    crest_level = levels["upper_hwl"] + st.session_state.freeboard_m + st.session_state.wave_allowance_m + st.session_state.settlement_allowance_m
    dam_height = max(crest_level - fnum('dam_foundation_rl'), 0.0)
    reservoir_area = safe_div(fnum('reservoir_volume_m3'), active_depth)
    section_area = dam_height * (
        st.session_state.dam_crest_width_m
        + 0.5 * (st.session_state.upstream_slope_hv + st.session_state.downstream_slope_hv) * dam_height
    )
    embankment_volume = section_area * st.session_state.dam_crest_length_m
    energy_head = design_case["net_head_m"] if np.isfinite(design_case["net_head_m"]) else fnum('teaching_effective_head_m')
    e_gwh = RHO * G * fnum('reservoir_volume_m3') * max(energy_head, 0.0) * eta_gen / 3.6e12
    duration_h = safe_div(e_gwh * 1000.0, fnum('design_power_mw'))
    q_report = design_case["total_discharge_m3_s"] if np.isfinite(design_case["total_discharge_m3_s"]) else q_design
    q_per_penstock = safe_div(q_report, st.session_state.penstocks)
    q_per_unit = safe_div(q_report, st.session_state.units)
    branch_diameter = float(fnum('unit_penstock_diameter_m'))
    branch_velocity = safe_div(q_per_unit, area_circle(branch_diameter))
    operating_head = energy_head
    turbine = select_turbine(operating_head, q_report)
    p_hyd_mw = RHO * G * q_report * operating_head / 1e6
    p_gen_mw = p_hyd_mw * eta_gen
    p_pump_mw = safe_div(RHO * G * q_report * operating_head, st.session_state.eta_pump * 1e6)
    n_q = st.session_state.runner_speed_rpm * math.sqrt(max(q_per_unit, 0.001)) / max(operating_head, 0.001) ** 0.75
    capital_cost_m = float(st.session_state.get("summary_capital_cost_m", 0.0))
    annual_om_percent = float(st.session_state.get("summary_annual_om_percent", 1.5))
    annual_benefit_m = float(st.session_state.get("summary_annual_benefit_m", 0.0))
    screening_life_y = float(st.session_state.get("summary_screening_life_y", 40))
    annual_om_m = capital_cost_m * annual_om_percent / 100.0
    cost_per_kw = safe_div(capital_cost_m * 1e6, fnum('design_power_mw') * 1000.0)
    cost_per_kwh = safe_div(capital_cost_m * 1e6, e_gwh * 1e6)
    simple_bc = undiscounted_benefit_cost_proxy(
        capital_cost_m,
        annual_om_m,
        annual_benefit_m,
        screening_life_y,
        str(st.session_state.get("summary_cost_source", "")),
        str(st.session_state.get("summary_benefit_source", "")),
    )
    dam_recommendation = recommend_dam_type(
        st.session_state.reservoir_arrangement,
        st.session_state.valley_geometry,
        st.session_state.foundation_quality,
        st.session_state.construction_material,
        dam_height,
        fnum('reservoir_volume_m3'),
    )
    dam_selection_basis = dam_selection_governing_text(
        st.session_state.reservoir_arrangement,
        st.session_state.valley_geometry,
        st.session_state.foundation_quality,
        st.session_state.construction_material,
        dam_height,
        fnum('reservoir_volume_m3'),
    )
    step2_evidence = current_step2_evidence_summary()
    step2_evidence_text = f"{int(step2_evidence['ready_count'])}/{int(step2_evidence['total'])} ({step2_evidence['level']})"
    step2_ready_text = compact_list_text(list(step2_evidence["ready_items"]), limit=6)
    step2_missing_text = compact_list_text(list(step2_evidence["missing_items"]), limit=6, empty="No missing items currently flagged")

    sections = [
        report_step_section(
            1,
            "Project Brief, Energy Target, Site Constraints, and Site Screening",
            "This step defines the design problem, screens the site, and prevents the project from becoming a calculation exercise without a clear grid, storage, environmental and site context.",
            [
                "State the intended energy-storage service and why the project is needed.",
                "Define the study boundary, vertical datum, map scale, topographic data source and confidence level.",
                "Set the target installed power and target generation duration before sizing reservoirs and waterways.",
                "Screen major exclusions: protected areas, cultural heritage, settlements, roads, water licences, land ownership and grid access.",
                "Use QGIS DEM/contour and constraint layers to screen the search area, first-pass head opportunity and route-length ratio before mapping reservoir pairs in Step 2.",
            ],
            [
                r"E_{target}=P_{design}t_{target}",
                r"V_{first\ pass}\approx\frac{E_{target}\,3.6\times10^{12}}{\rho g H_e\eta_{gen}}",
                r"H_g=z_{upper}-z_{lower},\qquad S_{route}=\frac{H_g}{L_{route}}",
            ],
            [
                ("Project name", st.session_state.get("project_name", "PHES project")),
                ("Design power", metric_value(fnum('design_power_mw'), " MW", 0)),
                ("Target duration", metric_value(fnum('operation_hours'), " h", 1)),
                ("Market/storage role", market_role),
                ("First-pass target energy", metric_value(fnum('design_power_mw') * fnum('operation_hours') / 1000.0, " GWh", 2)),
            ],
            [
                "Location map and study boundary.",
                "DEM or contour data source, grid resolution, date and vertical datum.",
                "Constraint map showing environment, access, transmission and social constraints.",
                "Brief note on data confidence and missing investigations.",
            ],
            [
                "Explain why the selected storage duration is useful for the target grid role.",
                "Explain whether the site is a Snowy-style existing-reservoir case, a closed-loop off-river case, or another PHES type.",
                "Identify the assumptions that most affect the design before detailed calculations begin.",
            ],
        ),
        report_step_section(
            2,
            "Reservoir Opportunity Mapping and Dam Concept Selection",
            "This step turns the reconnaissance problem into mapped alternatives, then chooses the reservoir arrangement and preliminary dam type before levels and dimensions are locked in.",
            [
                "Use QGIS to map candidate upper and lower reservoirs, contour bands, possible dam alignments, powerhouse areas, waterway corridors and grid connection routes.",
                "Compare at least two reservoir pairs using gross head, active storage, waterway length, environmental constraints, constructability and grid distance.",
                "Record whether the selected reservoirs are existing assets, new off-river reservoirs, mine-pit/quarry reuse, or open-loop reservoirs.",
                "Assess valley geometry, foundation quality and construction material availability.",
                "Select a preliminary dam type and state why other types are less suitable.",
                "Identify required next investigations for foundation, seepage, slope stability and dam safety.",
            ],
            [
                r"H_{g,rep}=z_{u,rep}-z_{l,rep}",
                r"V_{active}\approx\sum_i A_i\Delta z_i",
                r"L_{profile}=\sum_i\sqrt{(\Delta x_i)^2+(\Delta z_i)^2}",
                r"Dam\ type=f(geometry,\ foundation,\ materials,\ H_{dam},\ V_{active})",
            ],
            [
                ("Representative upper level", metric_value(levels["upper_representative"], " m", 1)),
                ("Lower TWL", metric_value(levels["lower_twl"], " m", 1)),
                ("Representative gross head", metric_value(levels["gross_head"], " m", 1)),
                ("Active storage input", metric_value(fnum('reservoir_volume_m3') / 1e6, " GL", 1)),
                ("Reservoir arrangement", st.session_state.reservoir_arrangement),
                ("Valley geometry", st.session_state.valley_geometry),
                ("Foundation quality", st.session_state.foundation_quality),
                ("Construction material", st.session_state.construction_material),
                ("Recommended screening dam type", dam_recommendation),
                ("Dam selection basis", dam_selection_basis),
                ("Evidence readiness", step2_evidence_text),
                ("Ready evidence items", step2_ready_text),
                ("Missing evidence items", step2_missing_text),
            ],
            [
                "QGIS plan map with reservoirs, candidate routes, access and environmental constraints.",
                "Reservoir-pair comparison table with data source and confidence.",
                "Dam site sketch or GIS screenshot with foundation and abutment assumptions.",
                "Material-source assumption, including excavation reuse if applicable.",
                "Comparison of at least two dam/reservoir concepts and selected option justification.",
            ],
            [
                "Discuss why the selected reservoir pair and dam concept are better than the rejected alternatives.",
                "Explain whether the dam concept controls the reservoir levels or whether existing reservoir levels control the dam concept.",
                "Identify which GIS-derived quantities were measured directly and which were assumed.",
                "State what field investigations are needed before the concept could become a preliminary design.",
            ],
        ),
        report_step_section(
            3,
            "Reservoir Levels, Dam Sizing, Gross Head, and Storage Energy",
            "This step converts the reservoir concept into water levels, active depth, crest level, dam height, gross head, deliverable energy and full-load duration.",
            [
                "Define upper and lower HWL/LWL values using a consistent vertical datum.",
                "State the representative operating levels explicitly; the app's one-third upper-drawdown value is only a teaching placeholder.",
                "Calculate simultaneous minimum and maximum gross head before selecting a representative design head.",
                "Check whether active depth and reservoir area are physically plausible from contours.",
                "Size the preliminary dam crest and height using freeboard, wave and settlement allowances.",
            ],
            [
                r"z_{u,rep}=HWL_u-\frac{HWL_u-LWL_u}{3}\quad\text{(teaching placeholder)}",
                r"H_{g,rep}=z_{u,rep}-z_{l,rep}",
                r"H_{g,max}=HWL_u-LWL_l,\qquad H_{g,min}=LWL_u-HWL_l",
                r"E_{delivered}=\frac{\rho g V_{active}H_e\eta_{gen}}{3.6\times10^{12}}",
                r"t_{full}=\frac{1000E_{delivered}}{P_{design}}",
            ],
            [
                ("Upper HWL / representative / LWL", f"{levels['upper_hwl']:.1f} / {levels['upper_representative']:.1f} / {levels['upper_lwl']:.1f} m"),
                ("Lower HWL / TWL", f"{levels['lower_hwl']:.1f} / {levels['lower_twl']:.1f} m"),
                ("Simultaneous gross-head envelope", f"{levels['gross_head_min']:.1f} to {levels['gross_head_max']:.1f} m"),
                ("Active depth", metric_value(active_depth, " m", 1)),
                ("Crest level", metric_value(crest_level, " m", 1)),
                ("Dam height", metric_value(dam_height, " m", 1)),
                ("Approximate active reservoir area", metric_value(reservoir_area / 1e6, " km^2", 2)),
                ("Approximate embankment volume", metric_value(embankment_volume / 1e6, " million m^3", 2)),
                ("Deliverable energy", metric_value(e_gwh, " GWh", 2)),
                ("Full-load duration", metric_value(duration_h, " h", 1)),
            ],
            [
                "Reservoir level schematic with both operating bands and the stated representative levels.",
                "Storage-elevation or contour-area evidence.",
                "Dam crest and freeboard assumptions.",
                "Sensitivity of energy and duration to active volume and head.",
            ],
            [
                "Discuss whether the selected active storage is realistic for the mapped basin.",
                "Explain the difference between gross head, selected first-pass head and final civil net head.",
                "State whether storage volume or hydraulic power is the limiting feature.",
            ],
            figure=(
                ("fig_reservoir_levels.png", "Reservoir operating levels with crest level and gross head (screening schematic generated by the teaching app).")
                if "fig_reservoir_levels.png" in figure_files
                else None
            ),
        ),
        report_step_section(
            4,
            "Waterway Corridor, Layout, and Design Discharge",
            "This step fixes the hydraulic alignment, then selects either a power-target sizing discharge or a sourced declared operating discharge and carries that flow consistently into later checks.",
            [
                "Draw the selected waterway alignment using chainage and waterway RL, not only the ground topography.",
                "Locate intake, headrace, pressure shaft, powerhouse, tailrace, surge tank, access adits and construction portals.",
                "Compare route alternatives by length, cover, constructability, geology, environmental footprint and access.",
                "Use total plant discharge first; only divide into units after choosing the unit arrangement.",
                "Start with the Step 3 gross head for first-pass discharge, or use a lower effective head only when a preliminary allowance or benchmark net-head basis is stated.",
                "Check that the selected discharge drains the active storage over a plausible operating duration, and compare it with large PHES benchmarks.",
            ],
            [
                r"L_{profile}=\sum_i\sqrt{(\Delta chainage_i)^2+(\Delta RL_i)^2}",
                r"Gradient=\frac{\Delta RL}{L_{profile}}",
                r"Cover=RL_{ground}-RL_{waterway}",
                r"Q_{target}=\frac{P_{design}\times10^6}{\rho g H\eta},\qquad Q=Q_{target}\ \text{or sourced }Q_{declared}",
                r"Q_u=\frac{Q_{total}}{N_{units}}",
                r"t=\frac{V_{active}}{Q_{total}3600}",
            ],
            [
                ("Hydraulic length used in loss check", metric_value(fnum('penstock_length_m'), " m", 0)),
                ("Lower tailwater", metric_value(levels["lower_twl"], " m", 1)),
                ("Representative upper level", metric_value(levels["upper_representative"], " m", 1)),
                ("Selected first-pass head", metric_value(fnum('teaching_effective_head_m'), " m", 1)),
                ("Sizing efficiency", metric_value(st.session_state.sizing_efficiency * 100.0, "%", 1)),
                ("Discharge basis", str(st.session_state.get("discharge_basis", DISCHARGE_BASIS_POWER))),
                ("Selected operating discharge", metric_value(q_design, " m^3/s", 1)),
                ("Discharge per unit", metric_value(q_unit, " m^3/s", 1)),
                ("Storage duration from Q", metric_value(safe_div(fnum('reservoir_volume_m3'), q_design * 3600.0), " h", 1)),
            ],
            [
                "Waterway plan and longitudinal section exported from QGIS or the Streamlit alignment table.",
                "Chainage, ground RL and waterway RL table with route alternatives and selected-route justification.",
                "Calculation table showing P, H, efficiency and Q with a storage-duration check.",
                "Sensitivity case for head or efficiency.",
                "Benchmark comparison to a PHES or hydropower project of similar scale.",
            ],
            [
                "Explain why the chosen waterway route is hydraulically and constructably reasonable.",
                "Discuss how route length affects both head loss and surge/transient design.",
                "Explain why the total discharge is large for high-power PHES and whether discharge or storage controls the duration.",
                "Identify how later loss calculations will update net head and therefore Q.",
            ],
            figure=(
                ("fig_long_section.png", "PHES scheme long section with waterway alignment, powerhouse location and gross/effective head (screening schematic generated by the teaching app).")
                if "fig_long_section.png" in figure_files
                else None
            ),
        ),
        report_step_section(
            5,
            "Shared Conduit Losses and Diameter Sensitivity",
            "This step tests whether the shared waterway diameter gives acceptable velocity, Reynolds number, friction factor, local losses and net head.",
            [
                "Select roughness from lining/material evidence rather than arbitrary manual loss factors.",
                "Calculate Reynolds number and Darcy friction factor for each diameter case.",
                "Include local losses from actual components such as entrance, trash rack, bends, transitions, branch and outlet.",
                "Compare a diameter sensitivity table and select a diameter based on velocity, loss budget, constructability and cost.",
            ],
            [
                r"A=\frac{\pi D^2}{4}",
                r"v=\frac{Q_v}{A}",
                r"Re=\frac{vD}{\nu}",
                r"h_{major}=f\frac{L}{D}\frac{v^2}{2g},\qquad h_{minor}=\Sigma K\frac{v^2}{2g}",
                r"H_e=H_{sel}-h_{major}-h_{minor}-h_{other}",
            ],
            [
                ("Shared conduit diameter", metric_value(fnum('penstock_diameter_m'), " m", 2)),
                ("Flow area basis", str(design_case.get("flow_area_mode", "Per penstock"))),
                ("Velocity", metric_value(design_case["velocity_m_s"], " m/s", 2)),
                ("Reynolds number", metric_value(design_case["reynolds"], "", 0)),
                ("Darcy friction factor", metric_value(design_case["friction"], "", 4)),
                ("Local loss sum ΣK", metric_value(design_case["k_sum"], "", 2)),
                ("Major / minor loss", f"{design_case['major_loss_m']:.2f} / {design_case['minor_loss_m']:.2f} m"),
                ("Other hydraulic loss", metric_value(design_case["other_loss_m"], " m", 2)),
                ("Updated net head", metric_value(design_case["net_head_m"], " m", 1)),
            ],
            [
                "Diameter sensitivity table from the Streamlit app.",
                "Selected roughness material and roughness value.",
                "Local-loss component table and selected component counts.",
                "Comment on whether velocity is inside or outside the teaching range.",
            ],
            [
                "Explain why lower velocity reduces loss but increases diameter and civil cost.",
                "Explain whether the loss budget is acceptable relative to gross head.",
                "State if the selected diameter should be revised before turbine selection.",
            ],
            figure=(
                ("fig_system_curves.png", "System net head and generated power against total discharge, with the design point marked (generated by the teaching app).")
                if "fig_system_curves.png" in figure_files
                else None
            ),
        ),
        report_step_section(
            6,
            "Unit Branches, Penstocks, and Intake/Outlet Sizing",
            "This step splits the total plant flow into units and checks branch velocity, branch diameter, intake/outlet sizing and transient implications.",
            [
                "Divide total flow among units and penstocks.",
                "Compare branch velocities, typically from about 3 to 7 m/s, using diameter, head loss, valve size and transient risk.",
                "Check that branch and intake/outlet dimensions are constructable and compatible with turbine unit flow.",
                "Carry the selected unit branch diameter forward into surge and powerhouse checks.",
            ],
            [
                r"Q_{unit}=\frac{Q_{total}}{N_{units}}",
                r"D_u=\sqrt{\frac{4Q_u}{\pi v_u}}",
                r"h_u=\left(f\frac{L_u}{D_u}+K_u\right)\frac{v_u^2}{2g}",
                r"\Delta H_{inst}=\frac{av_u}{g}",
            ],
            [
                ("Number of units", str(int(st.session_state.units))),
                ("Number of penstocks", str(int(st.session_state.penstocks))),
                ("Total operating discharge", metric_value(q_report, " m^3/s", 1)),
                ("Per-unit discharge", metric_value(q_per_unit, " m^3/s", 1)),
                ("Per-penstock discharge", metric_value(q_per_penstock, " m^3/s", 1)),
                ("Selected unit branch diameter", metric_value(branch_diameter, " m", 2)),
                ("Implied unit branch velocity", metric_value(branch_velocity, " m/s", 2)),
            ],
            [
                "Velocity comparison table for unit branches.",
                "Selected intake/outlet and valve arrangement.",
                "Sketch of manifold or bifurcation logic.",
                "Comment on constructability, access and transient implications.",
            ],
            [
                "Justify the selected branch velocity; do not only state the 4-7 m/s guidance.",
                "Explain whether the unit arrangement is driven by hydraulic, equipment or civil constraints.",
                "Identify whether branch losses require revising the shared conduit or unit diameter.",
            ],
        ),
        report_step_section(
            7,
            "Underground Civil Structures and Geotechnical Checks",
            "This step checks whether the pressure waterways and powerhouse cavern system are spatially and geotechnically plausible.",
            [
                "Check pressure tunnel cover and confinement using hydrostatic head and rock cover.",
                "Apply the hydraulic jacking screen (minimum in-situ stress versus internal water pressure), then classify each waterway zone as unlined/shotcrete, reinforced concrete with grouting, or steel-lined.",
                "Size the machine hall, transformer hall and isolated phase bus gallery using unit count, bay spacing, access and maintenance allowance.",
                "Check access adits, ventilation, drainage, egress and construction sequence at a concept level.",
                "State geotechnical investigation needs: rock mass quality, faults, groundwater, stress measurement and support class.",
            ],
            [
                r"p=\frac{\rho gH}{10^6}",
                r"\sigma_v=\gamma_r h_c,\quad \sigma_h=k\sigma_v",
                r"F_{RV}=\frac{C_{RV}\gamma_r\cos\alpha}{h_s\gamma_w}",
                r"F_{RM}=\frac{C_{RM}\gamma_r\cos\beta}{h_s\gamma_w}",
                r"F_{jack}=\frac{\sigma_3}{\gamma_w h_s}\ge 1.3,\quad \sigma_3\approx k_{min}\gamma_r C_{RM}",
            ],
            [
                ("Cover depth input", metric_value(st.session_state.cover_depth_m, " m", 1)),
                ("Rock unit weight", metric_value(st.session_state.rock_unit_weight, " kN/m^3", 1)),
                ("Effective head for pressure check", metric_value(fnum('teaching_effective_head_m'), " m", 1)),
                ("Units driving cavern length", str(int(st.session_state.units))),
                ("Selected unit branch diameter", metric_value(branch_diameter, " m", 2)),
            ],
            [
                "Powerhouse cavern concept sketch showing machine hall, transformer hall and IPB gallery.",
                "Cover and geological long section.",
                "Rock mass and fault assumptions.",
                "Lining-class map along the waterway from the cover and jacking screens.",
                "Preliminary support, lining, drainage, ventilation and access notes.",
            ],
            [
                "Explain which underground component is likely to control project risk.",
                "Discuss whether the selected powerhouse location is compatible with waterway alignment and access.",
                "State which geotechnical uncertainties must be resolved before final design.",
            ],
            figure=(
                ("fig_lining_stress.png", "Pressure tunnel lining radial and hoop stress across the concrete lining, with the allowable tensile-strength screen (generated by the teaching app).")
                if "fig_lining_stress.png" in figure_files
                else None
            ),
        ),
        report_step_section(
            8,
            "Surge and Transient Check",
            "This step screens transient severity and identifies the connected-system analysis required before pressure limits or surge-control dimensions can be accepted.",
            [
                "Use the selected waterway length, diameter and discharge to calculate critical closure time.",
                "Use the Joukowsky bound for closure at or below the round-trip wave time and a rigid-column screen for slower closure.",
                "Do not size a surge tank from a universal area ratio; define the cited stability/transient criteria and detailed model required.",
                "Check both generating load rejection and pumping trip/reversal qualitatively.",
            ],
            [
                r"A_{conn}=n\frac{\pi D^2}{4},\qquad v_0=\frac{Q_0}{A_{conn}}",
                r"T_w=\frac{2L}{a}",
                r"\Delta H_J=\frac{av_0}{g}\quad(T_c\le T_w)",
                r"\Delta H_{RC}=\frac{2Lv_0}{gT_c}\quad(T_c>T_w)",
            ],
            [
                ("Shared conduit length", metric_value(fnum('penstock_length_m'), " m", 0)),
                ("Shared conduit diameter", metric_value(fnum('penstock_diameter_m'), " m", 2)),
                ("Velocity carried from losses", metric_value(design_case["velocity_m_s"], " m/s", 2)),
                ("Net head carried into transient check", metric_value(design_case["net_head_m"], " m", 1)),
            ],
            [
                "Defined transient event and connected-system boundaries.",
                "Rapid, selected and slow closure severity table with the equation used for each case.",
                "Maximum and minimum head screens, including any negative-pressure flag.",
                "Selected surge-control concept and statement that no structure is sized by this screen.",
                "Named method-of-characteristics or equivalent modelling plan and acceptance criteria.",
            ],
            [
                "Explain whether the surge tank is a headrace, tailrace or combined protection issue.",
                "Identify operating events that may control transient design.",
                "Discuss how increasing diameter, changing closure time or adding surge control would affect the result.",
            ],
        ),
        report_step_section(
            9,
            "Turbine, Pumping, Efficiency, and Operating Envelope",
            "This step selects the pump-turbine family and checks generation, pumping, unit flow and operating envelope after head and discharge are known.",
            [
                "Select turbine family from head and discharge, then check whether reversible pump-turbine operation is plausible.",
                "Use unit flow and unit power to check manufacturability and specific-speed range.",
                "Apply turbine, generator and transformer efficiencies for generation; apply pump efficiency for charging power.",
                "Check part-load operation using Q/Q_{max} and explain the operating envelope.",
            ],
            [
                r"P_h=\frac{\rho gQH}{10^6}",
                r"P_{gen}=\frac{\rho gQH\eta_{total}}{10^6}",
                r"P_{pump}=\frac{\rho gQH}{\eta_p10^6}",
                r"n_q=\frac{N\sqrt{Q_u}}{H^{3/4}}",
            ],
            [
                ("Recommended turbine family", turbine),
                ("Runner speed", metric_value(st.session_state.runner_speed_rpm, " rpm", 0)),
                ("Unit discharge", metric_value(q_per_unit, " m^3/s", 1)),
                ("Hydraulic power", metric_value(p_hyd_mw, " MW", 1)),
                ("Generated power estimate", metric_value(p_gen_mw, " MW", 1)),
                ("Pumping power estimate", metric_value(p_pump_mw, " MW", 1)),
                ("Flow specific speed n_q", metric_value(n_q, "", 2)),
                ("Generation efficiency chain", metric_value(eta_gen * 100.0, "%", 1)),
            ],
            [
                "Turbine application chart or table.",
                "Efficiency assumptions and source.",
                "Unit flow/power table.",
                "Operating envelope discussion for generation and pumping.",
            ],
            [
                "Explain why the selected turbine family suits the head-flow combination.",
                "Discuss whether unit count is reasonable for operation, maintenance and manufacturability.",
                "State limitations of the simplified efficiency and specific-speed checks.",
            ],
            figure=(
                ("fig_turbine_chart.png", "Turbine application zones with the current design operating point (generated by the teaching app).")
                if "fig_turbine_chart.png" in figure_files
                else None
            ),
        ),
        report_step_section(
            10,
            "NEM Integration, Risk Register, and Final Recommendation",
            "This step connects the technical design to grid value, economic screening, environmental monitoring, residual risks and the final recommendation.",
            [
                "State how the project charges and generates in the National Electricity Market or equivalent grid context.",
                "Estimate construction-cost screening metrics and compare alternatives using cost per kW, cost per kWh and sourced, scope-consistent economic evidence.",
                "Prepare a risk register covering hydrology, geotechnical uncertainty, environment, transmission, market value and construction cost.",
                "Provide a final recommendation: proceed, proceed with conditions, or reject.",
            ],
            [
                r"\eta_{cycle}=\eta_{gen}\eta_{pump}",
                r"E_{pump}=\frac{E_{delivered}}{\eta_{cycle}}",
                r"C_{kW}=\frac{C_{capital}}{P_{rated}}",
                r"C_{kWh}=\frac{C_{capital}}{E_{delivered}}",
                r"BCR=\frac{\sum_t B_t/(1+r)^t}{\sum_t C_t/(1+r)^t}",
                r"R_{undisc}=\frac{B_{annual}N}{C_{capital}+C_{O\&M,annual}N}",
            ],
            [
                ("Grid/storage role", market_role),
                ("Deliverable energy", metric_value(e_gwh, " GWh", 2)),
                ("Cycle efficiency", metric_value(eta_cycle * 100.0, "%", 1)),
                ("Pumping energy for full cycle", metric_value(safe_div(e_gwh, eta_cycle), " GWh", 2)),
                ("Screening capital cost", metric_value(capital_cost_m, " $M", 1)),
                ("Annual O&M", metric_value(annual_om_m, " $M/y", 1)),
                ("Cost per kW", metric_value(cost_per_kw, " $/kW", 0)),
                ("Cost per kWh", metric_value(cost_per_kwh, " $/kWh", 0)),
                ("Undiscounted economic proxy", metric_value(simple_bc, "", 2)),
            ],
            [
                "Dispatch story: pumping window, generation window, storage duration and grid services.",
                "Transmission connection assumption and network-constraint discussion.",
                "Cost build-up and benchmark source.",
                "Benefit source and a check that arbitrage, capacity and service value are not double counted.",
                "Risk register with likelihood, consequence, rating and mitigation.",
                "Environmental monitoring and traceability matrix.",
            ],
            [
                "Explain the national electricity market role and whether the storage duration supports it.",
                "Discuss whether economics are strong, uncertain or not yet assessable.",
                "State the most important next investigations and the final recommendation.",
            ],
        ),
    ]
    return "\n\n".join(sections)


def find_pdflatex() -> str | None:
    """Find pdflatex from PATH or common Windows TeX distribution locations."""
    for command in ("pdflatex", "pdflatex.exe"):
        found = shutil.which(command)
        if found:
            return found

    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    program_files = os.environ.get("ProgramFiles")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")

    if local_app_data:
        candidates.extend(
            [
                Path(local_app_data) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "pdflatex.exe",
                Path(local_app_data) / "Programs" / "MiKTeX" / "miktex" / "bin" / "pdflatex.exe",
            ]
        )
    if program_files:
        candidates.extend(
            [
                Path(program_files) / "MiKTeX" / "miktex" / "bin" / "x64" / "pdflatex.exe",
                Path(program_files) / "MiKTeX" / "miktex" / "bin" / "pdflatex.exe",
            ]
        )
    if program_files_x86:
        candidates.append(Path(program_files_x86) / "MiKTeX" / "miktex" / "bin" / "pdflatex.exe")

    system_drive = os.environ.get("SystemDrive", "C:")
    texlive_root = Path(system_drive) / "texlive"
    if texlive_root.exists():
        candidates.extend(sorted(texlive_root.glob(r"*\bin\windows\pdflatex.exe"), reverse=True))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def decode_process_output(output: bytes) -> str:
    if not output:
        return ""
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return output.decode(encoding)
        except UnicodeDecodeError:
            continue
    return output.decode("utf-8", errors="replace")


def build_latex_summary_source(levels: dict[str, float], design_case: dict[str, float], market_role: str = "Daily energy shifting") -> str:
    project_name_raw = st.session_state.get("project_name", "PHES project")
    project_name = latex_escape(project_name_raw)
    eta_gen = st.session_state.eta_turbine * st.session_state.eta_generator * st.session_state.eta_transformer
    eta_cycle = eta_gen * st.session_state.eta_pump
    q_design = selected_design_discharge(fnum('teaching_effective_head_m'))
    q_unit = discharge_per_unit(q_design)
    operating_head = design_case["net_head_m"] if np.isfinite(design_case["net_head_m"]) else fnum('teaching_effective_head_m')
    turbine = select_turbine(operating_head, q_design)
    active_depth = max(levels["upper_hwl"] - levels["upper_lwl"], 0.0)
    energy_head = design_case["net_head_m"] if np.isfinite(design_case["net_head_m"]) else fnum('teaching_effective_head_m')
    e_deliverable_gwh = RHO * G * fnum('reservoir_volume_m3') * max(energy_head, 0.0) * eta_gen / 3.6e12
    duration_h = safe_div(e_deliverable_gwh * 1000.0, fnum('design_power_mw'))
    pump_energy_gwh = safe_div(e_deliverable_gwh, eta_cycle)
    crest_level = levels["upper_hwl"] + st.session_state.freeboard_m + st.session_state.wave_allowance_m + st.session_state.settlement_allowance_m
    dam_height = max(crest_level - fnum('dam_foundation_rl'), 0.0)
    dam_recommendation = recommend_dam_type(
        st.session_state.reservoir_arrangement,
        st.session_state.valley_geometry,
        st.session_state.foundation_quality,
        st.session_state.construction_material,
        dam_height,
        fnum('reservoir_volume_m3'),
    )
    dam_selection_basis = dam_selection_governing_text(
        st.session_state.reservoir_arrangement,
        st.session_state.valley_geometry,
        st.session_state.foundation_quality,
        st.session_state.construction_material,
        dam_height,
        fnum('reservoir_volume_m3'),
    )
    capital_cost_m = float(st.session_state.get("summary_capital_cost_m", 0.0))
    annual_om_percent = float(st.session_state.get("summary_annual_om_percent", 1.5))
    annual_benefit_m = float(st.session_state.get("summary_annual_benefit_m", 0.0))
    screening_life_y = float(st.session_state.get("summary_screening_life_y", 40))
    annual_om_m = capital_cost_m * annual_om_percent / 100.0
    cost_per_kw = safe_div(capital_cost_m * 1e6, fnum('design_power_mw') * 1000.0)
    cost_per_kwh = safe_div(capital_cost_m * 1e6, e_deliverable_gwh * 1e6)
    simple_bc = undiscounted_benefit_cost_proxy(
        capital_cost_m,
        annual_om_m,
        annual_benefit_m,
        screening_life_y,
        str(st.session_state.get("summary_cost_source", "")),
        str(st.session_state.get("summary_benefit_source", "")),
    )
    step2_evidence = current_step2_evidence_summary()
    step2_evidence_text = f"{int(step2_evidence['ready_count'])}/{int(step2_evidence['total'])} ({step2_evidence['level']})"
    step2_missing_text = compact_list_text(list(step2_evidence["missing_items"]), limit=6, empty="none currently flagged")

    body = f"""\\section*{{Project Snapshot}}
\\small
\\begin{{tabular}}{{@{{}}p{{0.46\\linewidth}}p{{0.46\\linewidth}}@{{}}}}
\\toprule
\\MonashTableHeader
\\MonashHeaderCell{{Item}} & \\MonashHeaderCell{{Current design value}} \\\\
\\midrule
Project name & {project_name} \\\\
Primary storage/grid role & {latex_escape(market_role)} \\\\
Installed design power & {latex_metric(fnum('design_power_mw'), "MW", 0)} \\\\
Maximum power & {latex_metric(fnum('max_power_mw'), "MW", 0)} \\\\
Units / penstocks & {int(st.session_state.units)} / {int(st.session_state.penstocks)} \\\\
Reservoir arrangement & {latex_escape(st.session_state.reservoir_arrangement)} \\\\
Preliminary dam type & {latex_escape(dam_recommendation)} \\\\
Dam selection basis & {latex_escape(dam_selection_basis)} \\\\
Step 2 evidence readiness & {latex_escape(step2_evidence_text)} \\\\
Turbine family & {latex_escape(turbine)} \\\\
\\bottomrule
\\end{{tabular}}
\\normalsize

\\section*{{Key Hydraulic and Storage Results}}
\\small
\\begin{{tabular}}{{@{{}}p{{0.40\\linewidth}}p{{0.22\\linewidth}}p{{0.30\\linewidth}}@{{}}}}
\\toprule
\\MonashTableHeader
\\MonashHeaderCell{{Quantity}} & \\MonashHeaderCell{{Value}} & \\MonashHeaderCell{{Note}} \\\\
\\midrule
Upper HWL / LWL & {latex_metric(levels["upper_hwl"], "m", 1)} / {latex_metric(levels["upper_lwl"], "m", 1)} & Upper storage operating band \\\\
Lower HWL / TWL & {latex_metric(levels["lower_hwl"], "m", 1)} / {latex_metric(levels["lower_twl"], "m", 1)} & Lower reservoir/tailwater \\\\
Representative upper level & {latex_metric(levels["upper_representative"], "m", 1)} & One-third-drawdown teaching placeholder \\\\
Active depth & {latex_metric(active_depth, "m", 1)} & HWL minus LWL \\\\
Representative gross head & {latex_metric(levels["gross_head"], "m", 1)} & Stated representative levels \\\\
Simultaneous head envelope & {latex_metric(levels["gross_head_min"], "m", 1)}--{latex_metric(levels["gross_head_max"], "m", 1)} & Upper-low/lower-high to upper-high/lower-low \\\\
Effective head & {latex_metric(fnum('teaching_effective_head_m'), "m", 1)} & Current design input \\\\
Active storage & {latex_metric(fnum('reservoir_volume_m3') / 1e6, "GL", 1)} & User input \\\\
Deliverable storage & {latex_metric(e_deliverable_gwh, "GWh", 2)} & Using generating efficiency chain \\\\
Full-load duration & {latex_metric(duration_h, "h", 1)} & Deliverable energy divided by power \\\\
Operating discharge & {latex_metric(q_design, "m^3/s", 1)} & {latex_escape(str(st.session_state.get("discharge_basis", DISCHARGE_BASIS_POWER)))} \\\\
Unit discharge & {latex_metric(q_unit, "m^3/s", 1)} & Total Q divided by units \\\\
Shared conduit diameter & {latex_metric(fnum('penstock_diameter_m'), "m", 2)} & Current design input \\\\
Shared conduit velocity & {latex_metric(design_case["velocity_m_s"], "m/s", 2)} & Current loss calculation \\\\
Total head deduction & {latex_metric(design_case["total_loss_m"], "m", 2)} & Major plus minor plus declared other hydraulic loss \\\\
Updated net head & {latex_metric(design_case["net_head_m"], "m", 1)} & Current civil loss result \\\\
\\bottomrule
\\end{{tabular}}
\\normalsize

\\section*{{Energy, Operation and Economics}}
\\small
\\begin{{tabular}}{{@{{}}p{{0.40\\linewidth}}p{{0.22\\linewidth}}p{{0.30\\linewidth}}@{{}}}}
\\toprule
\\MonashTableHeader
\\MonashHeaderCell{{Quantity}} & \\MonashHeaderCell{{Value}} & \\MonashHeaderCell{{Note}} \\\\
\\midrule
Generating efficiency chain & {latex_metric(eta_gen * 100.0, "%", 1)} & Turbine, generator, transformer \\\\
Approximate cycle efficiency & {latex_metric(eta_cycle * 100.0, "%", 1)} & Generation chain times pump efficiency \\\\
Pumping energy for full cycle & {latex_metric(pump_energy_gwh, "GWh", 2)} & Deliverable energy divided by cycle efficiency \\\\
Screening capital cost & {latex_metric(capital_cost_m, "$M", 1)} & Entered in Step 10 \\\\
Annual O\\&M & {latex_metric(annual_om_m, "$M/y", 1)} & {latex_metric(annual_om_percent, "%", 1)} of capital cost \\\\
Cost per kW & {latex_metric(cost_per_kw, "$/kW", 0)} & Capital cost divided by installed kW \\\\
Cost per kWh & {latex_metric(cost_per_kwh, "$/kWh", 0)} & Capital cost divided by delivered kWh \\\\
Undiscounted economic proxy & {latex_metric(simple_bc, "", 2)} & Shown only with sourced cost and benefit bases; not NPV or bankable BCR \\\\
\\bottomrule
\\end{{tabular}}
\\normalsize

\\section*{{Final Checks to Complete in the Full Report}}
\\begin{{warnbox}}[Before submission]
\\begin{{itemize}}
\\item Confirm reservoir-pair search, map scale, data source, datum and confidence.
\\item Attach QGIS maps for reservoirs, dam site, powerhouse site, access, transmission and environmental constraints.
\\item Complete missing Step 2 evidence items: {latex_escape(step2_missing_text)}.
\\item Compare at least two alternatives using dam site, powerhouse site, waterway route, maximum plant discharge, cost per kW/kWh and sourced, scope-consistent economic evidence.
\\item Complete risk register, environmental monitoring plan and design traceability matrix.
\\item Cross-check Streamlit outputs against the student spreadsheet before final submission.
\\end{{itemize}}
\\end{{warnbox}}
"""

    return latex_study_note_document(
        doc_subtitle=f"PHES Design Summary --- {project_name}",
        rhead="Design Summary, Semester 2, 2026",
        date_text="Streamlit Design Summary",
        core_box_title="Snapshot purpose",
        core_box_text=(
            "This compact summary captures the current design state from the teaching app. Use it as a "
            "design snapshot for meetings and progress checks, not as the final project report."
        ),
        body=body,
        include_toc=False,
    )


def compile_latex_to_pdf(
    latex_source: str,
    file_stem: str,
    assets: dict[str, bytes] | None = None,
) -> tuple[bytes | None, str | None]:
    pdflatex = find_pdflatex()
    if pdflatex is None:
        return None, (
            "pdflatex was not found. TeXworks is an editor; the app needs the MiKTeX or TeX Live "
            "compiler executable. Install MiKTeX/TeX Live, or add the TeX bin folder to PATH, then "
            "restart Streamlit. You can still download the LaTeX source and compile it manually."
        )

    safe_stem = safe_file_stem(file_stem, "phes_design_summary")
    with tempfile.TemporaryDirectory(prefix="phes_summary_") as tmpdir:
        tmp_path = Path(tmpdir)
        tex_path = tmp_path / f"{safe_stem}.tex"
        tex_path.write_text(latex_source, encoding="utf-8")
        for asset_name, asset_bytes in (assets or {}).items():
            (tmp_path / Path(asset_name).name).write_bytes(asset_bytes)
        command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
        try:
            # Two passes so the table of contents and \pageref{LastPage} resolve.
            for _ in range(2):
                completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=False, timeout=120)
                if completed.returncode != 0:
                    break
        except subprocess.TimeoutExpired:
            return None, "pdflatex timed out after 120 seconds."
        except OSError as exc:
            return None, f"pdflatex could not be started: {exc}"
        pdf_path = tmp_path / f"{safe_stem}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return pdf_path.read_bytes(), None
        if completed.returncode != 0:
            output = (decode_process_output(completed.stdout) + "\n" + decode_process_output(completed.stderr)).strip()
            return None, output[-4000:] if output else "pdflatex failed without diagnostic output."
        if not pdf_path.exists():
            return None, "pdflatex finished but no PDF was produced."
        return None, "pdflatex produced an empty PDF file."


def build_latex_report_source(
    levels: dict[str, float],
    design_case: dict[str, float],
    market_role: str = "Daily energy shifting",
    figure_files: frozenset[str] | set[str] | None = None,
) -> str:
    figure_files = figure_files or frozenset()
    project_name = latex_escape(st.session_state.get("project_name", "PHES project"))
    eta_gen = st.session_state.eta_turbine * st.session_state.eta_generator * st.session_state.eta_transformer
    eta_cycle = eta_gen * st.session_state.eta_pump
    q_design = selected_design_discharge(fnum('teaching_effective_head_m'))
    q_unit = safe_div(q_design, st.session_state.units)
    active_depth = max(levels["upper_hwl"] - levels["upper_lwl"], 0.0)
    crest_level = levels["upper_hwl"] + st.session_state.freeboard_m + st.session_state.wave_allowance_m + st.session_state.settlement_allowance_m
    dam_height = max(crest_level - fnum('dam_foundation_rl'), 0.0)
    reservoir_area = safe_div(fnum('reservoir_volume_m3'), active_depth)
    section_area = dam_height * (
        st.session_state.dam_crest_width_m
        + 0.5 * (st.session_state.upstream_slope_hv + st.session_state.downstream_slope_hv) * dam_height
    )
    embankment_volume = section_area * st.session_state.dam_crest_length_m
    energy_head = design_case["net_head_m"] if np.isfinite(design_case["net_head_m"]) else fnum('teaching_effective_head_m')
    e_gwh = RHO * G * fnum('reservoir_volume_m3') * max(energy_head, 0.0) * eta_gen / 3.6e12
    duration_h = safe_div(e_gwh * 1000.0, fnum('design_power_mw'))
    operating_head = design_case["net_head_m"] if np.isfinite(design_case["net_head_m"]) else fnum('teaching_effective_head_m')
    turbine = select_turbine(operating_head, q_design)
    dam_recommendation = recommend_dam_type(
        st.session_state.reservoir_arrangement,
        st.session_state.valley_geometry,
        st.session_state.foundation_quality,
        st.session_state.construction_material,
        dam_height,
        fnum('reservoir_volume_m3'),
    )
    logic_blocks = report_logic_latex_blocks()
    extended_sections = build_extended_report_sections(levels, design_case, market_role, figure_files)

    body = f"""\\section*{{Executive Summary}}
This report presents a preliminary pumped hydro energy storage (PHES) design developed using the 2026 10-step workflow.
The current Streamlit case, {project_name}, uses an installed power of {fnum('design_power_mw'):.0f} MW, a selected first-pass head of {fnum('teaching_effective_head_m'):.1f} m, and an estimated deliverable storage of {e_gwh:.2f} GWh.
The report should explain the design logic, assumptions, evidence and checks behind each value rather than only listing Streamlit outputs.

\\section*{{How to Read the Design Logic}}
\\begin{{notebox}}[Five questions for every section]
Each section should answer five questions: what decision is being made, what data were used, what equation or comparison was applied, what result was obtained, and what evidence supports the result.
The following map should remain in the submitted report so the calculation chain is visible to the reader.
\\end{{notebox}}

{logic_blocks}

{extended_sections}

\\appendix
\\section{{Spreadsheet and Streamlit Cross-Checks}}
Attach spreadsheet screenshots/tables and compare key results against Streamlit outputs.

\\section{{Traceability Matrix}}
Attach a matrix linking final design claims to workflow step, equation, data source, assumption, sensitivity case and responsible discipline.
"""

    return latex_study_note_document(
        doc_subtitle=f"PHES Final Design Report --- {project_name}",
        rhead="Final Design Report, Semester 2, 2026",
        date_text="Student Final Design Report",
        core_box_title="Core message",
        core_box_text=(
            "This report follows the RSE3241 10-step PHES design workflow. Every number must trace to a "
            "workflow step, an equation, a data source and an assumption. Replace all placeholder prompts "
            "with project evidence, maps, spreadsheet cross-checks and sensitivity cases before submission."
        ),
        body=body,
    )


def add_status(level: str, message: str) -> None:
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "success":
        st.success(message)
    else:
        st.info(message)


def app_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --hp-bg: #ffffff;
            --hp-surface: #ffffff;
            --hp-surface-soft: #ffffff;
            --hp-ink: #5A5A5A;
            --hp-muted: #5A5A5A;
            --hp-line: #e6e6e6;
            --hp-blue: #006dae;
            --hp-grey-1: #5A5A5A;
            --hp-grey-2: #969696;
            --hp-grey-3: #e6e6e6;
            --hp-orange: #f86700;
            --hp-green: #83a00a;
            --hp-heritage-blue: #abf5f9;
            --hp-electric-blue: #285aff;
            --hp-blueberry: #121256;
            --hp-input-bg: #ffffff;
            --hp-input-line: #006dae;
            --hp-output-bg: #ffffff;
            --hp-output-line: #006dae;
            --hp-ref-bg: #ffffff;
            --hp-ref-line: #5A5A5A;
            --hp-check-bg: #ffffff;
            --hp-check-line: #006dae;
            --hp-shadow: 0 3px 12px rgba(51, 51, 51, 0.08);
            --hp-radius: 8px;
        }
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: "Segoe UI", "Inter", "Roboto", "Helvetica Neue", Arial, sans-serif;
            color: var(--hp-ink);
        }
        .stApp {
            background: var(--hp-bg);
        }
        .block-container {
            padding-top: 0.85rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }
        .block-container h1 {
            color: var(--hp-blue);
            font-size: 2.0rem !important;
            line-height: 1.2 !important;
            letter-spacing: 0 !important;
            margin: 0.2rem 0 0.75rem !important;
            font-weight: 750 !important;
        }
        .block-container h2 {
            color: var(--hp-blue);
            font-size: 1.28rem !important;
            line-height: 1.25 !important;
            letter-spacing: 0 !important;
            margin: 1.05rem 0 0.5rem !important;
            font-weight: 720 !important;
        }
        .block-container h3 {
            color: var(--hp-blue);
            font-size: 1.0rem !important;
            line-height: 1.25 !important;
            letter-spacing: 0 !important;
            margin: 0.8rem 0 0.35rem !important;
            font-weight: 700 !important;
        }
        .block-container p,
        .block-container li,
        .block-container label,
        .block-container [data-testid="stMarkdownContainer"] {
            font-size: 0.94rem;
            line-height: 1.45;
        }
        .step-heading {
            border-left: 5px solid var(--hp-blue);
            padding: 0.2rem 0 0.25rem 0.95rem;
            margin: 0.15rem 0 1.05rem;
        }
        .step-eyebrow {
            color: var(--hp-blue);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0.02rem;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .step-heading h1 {
            font-size: 1.9rem !important;
            line-height: 1.18 !important;
            margin: 0 !important;
        }
        .step-subtitle {
            color: var(--hp-muted);
            font-size: 0.88rem;
            line-height: 1.25;
            margin-top: 0.25rem;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--hp-line);
        }
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 1.15rem;
        }
        section[data-testid="stSidebar"] h1 {
            color: var(--hp-blue) !important;
            font-size: 1.36rem !important;
            line-height: 1.25 !important;
            margin-bottom: 0.35rem !important;
            font-weight: 780 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--hp-muted) !important;
            font-size: 0.9rem !important;
            line-height: 1.35 !important;
            margin-bottom: 0.75rem !important;
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 0.95rem !important;
            line-height: 1.25 !important;
            margin: 0.55rem 0 0.25rem !important;
        }
        section[data-testid="stSidebar"] label p {
            font-size: 0.96rem !important;
            line-height: 1.25 !important;
            font-weight: 720 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.45rem;
        }
        section[data-testid="stSidebar"] [data-testid="stTabs"] button {
            padding: 0.35rem 0.25rem !important;
            font-size: 0.78rem !important;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.75rem 0 !important;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: var(--hp-radius);
            padding: 0.32rem 0.45rem;
            margin: 0.08rem 0;
            transition: background 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(0, 109, 174, 0.10);
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #ffffff;
            box-shadow: inset 4px 0 0 var(--hp-blue);
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
            color: var(--hp-blue) !important;
        }
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: var(--hp-radius) !important;
            border: 1px solid var(--hp-blue) !important;
            color: #ffffff !important;
            background: var(--hp-blue) !important;
            font-weight: 680 !important;
            min-height: 2.35rem;
        }
        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            border-color: var(--hp-blueberry) !important;
            background: var(--hp-blueberry) !important;
        }
        div[data-testid="stButton"] button:focus,
        div[data-testid="stDownloadButton"] button:focus,
        input:focus,
        textarea:focus,
        [role="combobox"]:focus {
            outline: 3px solid rgba(0, 109, 174, 0.22) !important;
            outline-offset: 1px !important;
        }
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stNumberInput"] label p,
        div[data-testid="stTextInput"] label p,
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stRadio"] label p {
            color: var(--hp-ink) !important;
            font-size: 0.84rem !important;
            font-weight: 650 !important;
            line-height: 1.25 !important;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {
            border-radius: var(--hp-radius) !important;
            border-color: var(--hp-line) !important;
            min-height: 2.35rem;
        }
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="select"] > div,
        div[data-testid="stDateInput"] input {
            background: var(--hp-input-bg) !important;
            border-color: var(--hp-input-line) !important;
        }
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stDateInput"] input,
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] > div * {
            color: var(--hp-ink) !important;
            -webkit-text-fill-color: var(--hp-ink) !important;
        }
        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder {
            color: var(--hp-muted) !important;
            -webkit-text-fill-color: var(--hp-muted) !important;
            opacity: 1;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background: var(--hp-input-line) !important;
            border-color: var(--hp-input-line) !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--hp-output-line);
            border-radius: var(--hp-radius);
            padding: 0.55rem 0.65rem;
            background: var(--hp-output-bg);
            min-height: 4.55rem;
            overflow: visible;
            box-shadow: 0 1px 2px rgba(29, 39, 51, 0.04);
        }
        div[data-testid="stMetric"]:hover {
            border-color: var(--hp-blueberry);
            box-shadow: 0 4px 14px rgba(29, 39, 51, 0.07);
            transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] * {
            color: var(--hp-muted) !important;
            -webkit-text-fill-color: var(--hp-muted) !important;
            font-size: 0.74rem !important;
            line-height: 1.2 !important;
            white-space: normal !important;
            font-weight: 650 !important;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: normal !important;
            color: var(--hp-ink) !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-size: 1.18rem !important;
            line-height: 1.15 !important;
            letter-spacing: 0 !important;
            font-weight: 760 !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stDataEditor"] {
            border-radius: var(--hp-radius);
            overflow: hidden;
            border: 1px solid var(--hp-blue);
            background: var(--hp-surface);
        }
        div[data-testid="stTable"] table thead th,
        [data-testid="stMarkdownContainer"] table thead th {
            background: var(--hp-blue) !important;
            color: #ffffff !important;
            border-color: #ffffff !important;
        }
        div[data-testid="stTable"] table tbody tr:nth-child(even),
        [data-testid="stMarkdownContainer"] table tbody tr:nth-child(even) {
            background: var(--hp-grey-3);
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--hp-grey-1);
            border-radius: var(--hp-radius);
            background: var(--hp-ref-bg);
        }
        div[data-testid="stExpander"] details > summary {
            background: var(--hp-grey-3);
            border-radius: var(--hp-radius) var(--hp-radius) 0 0;
        }
        div[data-testid="stTabs"] button {
            font-size: 0.88rem !important;
            font-weight: 650 !important;
            padding: 0.55rem 0.75rem !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--hp-blue) !important;
            border-bottom-color: var(--hp-blue) !important;
        }
        div[data-testid="stAlert"] {
            border-radius: var(--hp-radius);
            border-width: 1px;
            background: #ffffff;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
            border-left: 5px solid var(--hp-blue);
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]),
        div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
            border-left: 5px solid var(--hp-orange);
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
            border-left: 5px solid var(--hp-green);
        }
        [data-testid="stAlertContentWarning"],
        [data-testid="stAlertContentWarning"] *,
        [data-testid="stAlertContentError"],
        [data-testid="stAlertContentError"] * {
            color: var(--hp-orange) !important;
            -webkit-text-fill-color: var(--hp-orange) !important;
        }
        [data-testid="stAlertContentSuccess"],
        [data-testid="stAlertContentSuccess"] * {
            color: var(--hp-green) !important;
            -webkit-text-fill-color: var(--hp-green) !important;
        }
        .stProgress > div > div > div > div {
            background: var(--hp-blue);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--hp-line);
            border-radius: var(--hp-radius);
            background: var(--hp-surface);
            box-shadow: var(--hp-shadow);
            padding: 0.25rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.section-input-marker) {
            background: var(--hp-input-bg);
            border-color: var(--hp-input-line);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.section-output-marker) {
            background: var(--hp-output-bg);
            border-color: var(--hp-output-line);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.section-reference-marker) {
            background: var(--hp-ref-bg);
            border-color: var(--hp-ref-line);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.section-check-marker) {
            background: var(--hp-check-bg);
            border-color: var(--hp-check-line);
        }
        .section-marker {
            display: none;
        }
        .section-title-row {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0.05rem 0 0.45rem;
        }
        .section-title-row h2 {
            margin: 0 !important;
            font-size: 1.05rem !important;
            line-height: 1.25 !important;
        }
        .section-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            padding: 0.18rem 0.55rem;
            font-size: 0.68rem;
            line-height: 1;
            font-weight: 780;
            letter-spacing: 0.02rem;
            border: 1px solid currentColor;
            background: rgba(255, 255, 255, 0.72);
            white-space: nowrap;
        }
        .section-badge-input {color: var(--hp-blue);}
        .section-badge-output {color: var(--hp-blue);}
        .section-badge-reference {color: var(--hp-grey-1);}
        .section-badge-check {color: var(--hp-blue);}
        div[data-testid="stMetric"] {
            scroll-margin-top: 1rem;
        }
        .symbol-value-card {
            border: 1px solid var(--hp-line);
            border-radius: var(--hp-radius);
            padding: 0.55rem 0.65rem;
            background: var(--hp-output-bg);
            min-height: 4.55rem;
            box-shadow: 0 1px 2px rgba(29, 39, 51, 0.04);
        }
        .symbol-value-card:hover {
            border-color: var(--hp-blueberry);
            box-shadow: 0 4px 14px rgba(29, 39, 51, 0.07);
            transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
        }
        .symbol-value-label {
            color: var(--hp-muted);
            font-size: 0.74rem;
            line-height: 1.2;
            font-weight: 650;
            margin-bottom: 0.22rem;
        }
        .symbol-value-label sub {
            font-size: 0.72em;
            line-height: 0;
            vertical-align: -0.25em;
        }
        .symbol-value-number {
            color: var(--hp-ink);
            font-size: 1.18rem;
            line-height: 1.15;
            letter-spacing: 0;
            font-weight: 760;
            overflow-wrap: anywhere;
        }
        .equation-label {
            color: var(--hp-ink);
            font-size: 0.84rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 0.25rem;
        }
        .equation-label sub {
            font-size: 0.72em;
            line-height: 0;
            vertical-align: -0.25em;
        }
        .small-note {color: var(--hp-muted); font-size: 0.92rem;}
        .hero-card {
            background: #ffffff;
            border: 1px solid var(--hp-line);
            border-top: 6px solid var(--hp-blue);
            border-radius: var(--hp-radius);
            padding: 1.35rem 1.55rem 1.2rem;
            margin: 0.2rem 0 1.1rem;
            color: var(--hp-ink);
            box-shadow: none;
        }
        .hero-card h1 {
            color: var(--hp-blue) !important;
            font-size: 2.0rem !important;
            line-height: 1.2 !important;
            margin: 0 0 0.55rem !important;
        }
        .hero-card p {
            color: var(--hp-grey-1);
            font-size: 0.97rem;
            line-height: 1.5;
            margin: 0;
            max-width: 62rem;
        }
        .lab-kicker {
            color: var(--hp-blue);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0.04rem;
            text-transform: uppercase;
            margin: 0.2rem 0 0.15rem;
        }
        .lab-title {
            color: var(--hp-blue);
            font-size: 1.18rem;
            font-weight: 760;
            line-height: 1.25;
            margin: 0 0 0.15rem;
        }
        .lab-caption {
            color: var(--hp-muted);
            font-size: 0.9rem;
            line-height: 1.4;
            margin: 0 0 0.25rem;
        }
        .decision-pill {
            display: inline-block;
            border-radius: 4px;
            padding: 0.22rem 0.7rem;
            font-size: 0.82rem;
            font-weight: 760;
            border: 1px solid currentColor;
            background: #ffffff;
        }
        .decision-success {color: var(--hp-green);}
        .decision-warning {color: var(--hp-orange);}
        .decision-error {color: var(--hp-orange);}
        @media (max-width: 760px) {
            .block-container h1 {font-size: 1.65rem !important;}
            .hero-card {padding: 1.05rem 1.1rem;}
            .hero-card h1 {font-size: 1.55rem !important;}
            div[data-testid="stMetricValue"] > div {font-size: 1.02rem !important;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> str:
    with st.sidebar:
        st.title("Hydropower")
        project_name = str(st.session_state.get("project_name") or "").strip()
        st.caption(f"Project: {project_name}" if project_name else "No project yet - start at Step 1")
        if st.session_state.get("page") not in STEP_PAGE_OPTIONS:
            st.session_state["page"] = STEP_PAGE_OPTIONS[0]
        page = st.radio(
            "Design workflow",
            STEP_PAGE_OPTIONS,
            format_func=page_nav_label,
            key="page",
            label_visibility="collapsed",
        )

    return page


def phes_overview_schematic_figure() -> plt.Figure:
    """Teaching schematic for a closed-loop pumped hydro energy storage layout."""
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_facecolor("white")

    terrain_x = np.array([0.0, 0.5, 1.4, 2.3, 3.4, 4.6, 5.6, 6.6, 7.8, 9.0, 10.2])
    terrain_y = np.array([5.6, 5.85, 5.75, 5.25, 5.35, 5.2, 3.9, 2.3, 1.55, 1.15, 1.35])
    ax.fill_between(terrain_x, terrain_y, 0.0, color=MONASH_GREY_3, alpha=0.95, zorder=0)
    ax.plot(terrain_x, terrain_y, color=MONASH_GREY_1, linewidth=2.0, zorder=2)

    # Upper and lower storages.
    ax.fill([0.05, 1.8, 2.15, 0.05], [5.45, 5.45, 5.15, 5.15], color=MONASH_HERITAGE_BLUE, alpha=0.9, zorder=1)
    ax.plot([0.05, 2.15], [5.45, 5.45], color=MONASH_BLUE, linewidth=2.0)
    ax.fill([8.35, 10.25, 10.25, 8.65], [1.08, 1.08, 0.62, 0.62], color=MONASH_HERITAGE_BLUE, alpha=0.9, zorder=1)
    ax.plot([8.35, 10.25], [1.08, 1.08], color=MONASH_BLUE, linewidth=2.0)

    # Concept dam/embankment and powerhouse.
    ax.fill([1.8, 2.08, 2.35], [5.15, 4.35, 5.15], color=MONASH_GREY_2, alpha=0.95, zorder=3)
    ax.add_patch(plt.Rectangle((6.65, 0.88), 0.56, 0.54, facecolor=MONASH_GREY_1, edgecolor=MONASH_BLACK, linewidth=1.2, zorder=5))
    ax.add_patch(plt.Rectangle((7.85, 1.05), 0.16, 1.55, facecolor=MONASH_HERITAGE_BLUE, edgecolor=MONASH_BLUE, linewidth=1.3, zorder=4))
    ax.add_patch(plt.Rectangle((4.55, 5.15), 0.18, 1.15, facecolor=MONASH_HERITAGE_BLUE, edgecolor=MONASH_BLUE, linewidth=1.3, zorder=4))

    # Waterway.
    pipe_color = MONASH_BLUEBERRY
    ax.plot([2.15, 4.65], [5.30, 5.25], color=pipe_color, linewidth=5.0, solid_capstyle="round", zorder=4)
    ax.plot([4.65, 5.75, 6.65], [5.25, 2.75, 1.42], color=pipe_color, linewidth=5.0, solid_capstyle="round", zorder=4)
    ax.plot([7.20, 8.90], [1.18, 1.05], color=pipe_color, linewidth=5.0, solid_capstyle="round", zorder=4)
    ax.annotate("", xy=(6.4, 1.82), xytext=(5.7, 2.85), arrowprops=dict(arrowstyle="->", color=pipe_color, lw=1.7))
    ax.annotate("", xy=(8.5, 1.07), xytext=(7.65, 1.12), arrowprops=dict(arrowstyle="->", color=pipe_color, lw=1.7))

    # Operating levels and gross head.
    for y, label in [(5.72, "HWL"), (5.45, "Representative"), (5.18, "LWL")]:
        ax.hlines(y, -0.08, 0.35, color=MONASH_GREY_2, linewidth=1.1)
        ax.text(-0.16, y, label, ha="right", va="center", fontsize=8.8, color=MONASH_GREY_1)
    ax.hlines(1.08, 9.86, 10.32, color=MONASH_GREY_2, linewidth=1.1)
    ax.text(10.38, 1.08, "TWL", ha="left", va="center", fontsize=8.8, color=MONASH_GREY_1)
    ax.annotate("", xy=(9.55, 5.45), xytext=(9.55, 1.08), arrowprops=dict(arrowstyle="<->", lw=1.5, color=MONASH_GREY_1))
    ax.text(9.68, 3.35, r"$H_{g,rep}=z_{u,rep}-z_{l,rep}$", fontsize=11, color=MONASH_GREY_1, va="center")

    labels = [
        ("a", "upper reservoir", (0.72, 6.12)),
        ("b", "intake", (2.10, 5.86)),
        ("c", "low-pressure\nheadrace tunnel", (3.38, 5.82)),
        ("d", "headrace surge tank", (4.72, 6.55)),
        ("e", "pressure shaft", (5.48, 3.90)),
        ("f", "high-pressure penstock", (6.14, 2.35)),
        ("g", "underground powerhouse", (6.95, 0.68)),
        ("h", "tailrace surge tank", (8.08, 2.82)),
        ("i", "tailrace tunnel", (8.10, 1.43)),
        ("j", "lower reservoir", (9.35, 0.43)),
    ]
    for tag, text, (x, y) in labels:
        ax.text(
            x,
            y,
            f"{tag} = {text}",
            fontsize=8.8,
            color=MONASH_ELECTRIC_BLUE,
            ha="center",
            va="center",
            linespacing=1.05,
            bbox=dict(facecolor=MONASH_WHITE, edgecolor="none", alpha=0.84, pad=1.0),
        )

    ax.text(0.08, 0.20, "Flow direction: generate downhill; pump uphill when charging storage.", fontsize=9.5, color=MONASH_GREY_1)
    ax.set_xlim(-0.45, 10.8)
    ax.set_ylim(0.0, 6.9)
    ax.axis("off")
    ax.set_title("Schematic layout of a pumped hydropower storage project", loc="left", fontsize=15, color=MONASH_BLUE, pad=10)
    fig.tight_layout()
    return fig


def render_matplotlib_figure(figure: plt.Figure) -> None:
    """Render and close a figure so reruns do not retain native resources."""
    try:
        st.pyplot(figure, clear_figure=False)
    finally:
        plt.close(figure)


def render_overview_page(levels: dict[str, float], design_case: dict[str, float]) -> None:
    render_app_intro()
    st.subheader("The 10-step design workflow")
    st.dataframe(
        pd.DataFrame(
            [
                ["1", "Define MW, hours, site boundary, data confidence, and first GIS constraints.", "Carry candidate head, storage, access, and exclusions into Step 2."],
                ["2", "Map reservoir pairs, compare alternatives, and select the preliminary dam concept.", "Feed levels, storage, dam type, and route length into Steps 3-5."],
                ["3", "Calculate reservoir levels, crest, dam height, gross head, storage energy, and duration.", "Check whether the project target still makes sense before sizing waterways."],
                ["4", "Draw the waterway alignment and first-pass design discharge.", "Return here after Step 5 if losses change the effective head."],
                ["5", "Select conduit diameter, roughness, local losses, velocity, and net head.", "Use the civil net head to verify Step 4 discharge and Step 9 turbine selection."],
                ["6", "Split flow into units, penstocks, branches, intakes, and outlet works.", "Carry unit flow and transient risk into Steps 8-9."],
                ["7", "Check underground cover, jacking, lining, cavern dimensions, access, and support.", "Confirm whether the civil concept is buildable before final recommendation."],
                ["8", "Screen transient events, wave time, pressure severity and surge-control needs.", "Define the connected-system model before sizing or accepting surge control."],
                ["9", "Select pump-turbine family, unit operating point, and efficiency chain.", "Use unit-level head and flow, not only the plant total, near chart boundaries."],
                ["10", "Connect the design to NEM role, operating schedule, economics, risk, and traceability.", "State proceed, proceed with conditions, or reject with named next studies."],
            ],
            columns=["Step", "Student action", "Why it matters later"],
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("PHES project schematic")
    render_matplotlib_figure(phes_overview_schematic_figure())
    st.caption(
        "This drawing is a teaching schematic: real PHES projects vary in tunnel arrangement, surge protection, "
        "powerhouse location, tailwater connection, and whether one or both reservoirs already exist."
    )


def render_app_intro() -> None:
    """Overview hero. Pure guidance; nothing here is calculated from project inputs."""
    st.markdown(
        """
        <section class="hero-card">
            <h1>Pumped Hydro Energy Storage Design</h1>
            <p>
                Build a PHES scheme through a guided 10-step workflow: project brief and site screening,
                reservoir and waterway layout, hydraulic losses, underground works, surge checks, turbine
                selection, NEM integration, and report-ready design evidence.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Begin with a project brief and GIS evidence, then iterate: reservoir and dam concept, levels and "
        "storage, waterway losses, underground works, transients, turbine selection, and final economics/risk. "
        "Benchmark presets are examples only; replace them with project-specific evidence before submission."
    )


def workflow_step_number(page: str) -> int:
    return int(page.split(".", 1)[0].replace("Step", "").strip())


def page_workflow_step(page: str, levels: dict[str, float], design_case: dict[str, float]) -> None:
    step = workflow_step_number(page)
    render_step_header(step, page)

    if step == 1:
        st.caption("Define the project before calculations begin.")
        st.subheader("Project setup")
        active_preset = st.session_state.get("active_preset", NEW_PROJECT_PRESET)
        preset_index = PRESET_OPTIONS.index(active_preset) if active_preset in PRESET_OPTIONS else 0
        c_preset, c_apply = st.columns([2.2, 1.0])
        with c_preset:
            preset = st.selectbox(
                "Starting case",
                PRESET_OPTIONS,
                index=preset_index,
                format_func=lambda name: PRESET_LABELS.get(name, name),
                help="Use a benchmark preset for teaching, or start a blank student-owned PHES design.",
            )
            if preset == NEW_PROJECT_PRESET:
                st.caption("Starts a neutral editable case. Students should replace every value with site evidence from their own design.")
        with c_apply:
            st.write("")
            st.write("")
            if st.button("Create new project" if preset == NEW_PROJECT_PRESET else "Apply preset", width="stretch"):
                apply_preset(preset)
                st.success("New project started." if preset == NEW_PROJECT_PRESET else f"Applied {PRESET_LABELS.get(preset, preset)}.")

        if preset in PRESET_EVIDENCE_NOTES:
            note = PRESET_EVIDENCE_NOTES[preset]
            st.info(f"Public-project layer: {note['public_layer']}")
            st.warning(f"Course-model layer: {note['model_layer']}")
            st.caption(note["source"])

        st.text_input("Project name", key="project_name")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.number_input("Design power (MW)", min_value=1.0, max_value=5000.0, step=10.0, key="design_power_mw")
        with p2:
            st.number_input("Maximum power (MW)", min_value=1.0, max_value=6000.0, step=10.0, key="max_power_mw")
        with p3:
            st.number_input("Target operation duration (h)", min_value=0.1, step=0.5, key="operation_hours")

        if step1_complete():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Power target", metric_value(fnum('design_power_mw'), " MW", 0))
            c2.metric("Duration target", metric_value(fnum('operation_hours'), " h", 1))
            c3.metric("Energy target", metric_value(fnum('design_power_mw') * fnum('operation_hours') / 1000.0, " GWh", 1))
            c4.metric("Maximum power", metric_value(fnum('max_power_mw'), " MW", 0))
        else:
            st.info("Enter the design power and target operation duration to set the energy target for the project.")
        st.subheader("GIS site screening")
        st.caption("Use QGIS or equivalent GIS tools to prove the search area is plausible before mapping reservoir pairs in Step 2.")
        st.markdown("Recommended layers: DEM, contours, hillshade, watercourses, reservoirs, roads, transmission, land use, protected areas, cultural heritage, satellite imagery, and source/confidence notes.")
        render_screening_limitation()
        render_qgis_starter_lab()

        with st.expander("Project brief and reconnaissance checklists"):
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Project identity", st.session_state.get("project_name", "PHES project")],
                        ["Grid/storage role", "Peaking, renewable firming, reserve, black start, or long-duration storage."],
                        ["Reconnaissance boundary", "Reservoir-pair search area, map scale, vertical datum, exclusion zones, and available topographic data."],
                        ["Data source and confidence", "DEM resolution, contour interval, published storage curves, aerial imagery, grid data, and confidence rating."],
                        ["Reservoir availability", "Existing reservoir, new reservoir, mine pit, quarry, or ring-dike opportunity."],
                        ["Water and approvals", "Water source, licence, environmental impact, cultural heritage, land access."],
                        ["Initial viability", "Engineering, economic, environmental, and approvals reasons to continue or reject the site."],
                        ["Constructability", "Roads, portals, spoil disposal, grid/transmission connection, and likely construction-cost drivers."],
                    ],
                    columns=["Design basis item", "Student evidence"],
                ),
                hide_index=True,
                width="stretch",
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        ["1", "Select candidate reservoir pair search area", "DEM/contours, maps, imagery", "Proceed if upper/lower storage opportunity exists"],
                        ["2", "Screen grid access and transmission distance", "Transmission maps, substations, REZ/NEM context", "Proceed if connection pathway is plausible"],
                        ["3", "Screen environmental and social constraints", "Protected areas, waterways, heritage, land tenure", "Reject or reroute if fatal constraint appears"],
                        ["4", "Identify first-pass dam and powerhouse sites", "Contours, access, geology indicators", "Carry alternatives into Step 2"],
                        ["5", "Estimate first-pass head, energy, and discharge range", "Levels, storage opportunity, MW/GWh target", "Carry to Steps 3-4 for calculation"],
                    ],
                    columns=["Order", "Reconnaissance action", "Evidence", "Decision use"],
                ),
                hide_index=True,
                width="stretch",
            )

        with st.expander("Reference tables: criteria, workflow, tools, demos, and links"):
            st.subheader("Evidence-based PHES screening checklist")
            st.dataframe(
                pd.DataFrame(SCREENING_CRITERIA, columns=["Screen", "Scientific / practice basis", "Design use"]),
                hide_index=True,
                width="stretch",
            )
            st.subheader("QGIS screening workflow")
            st.dataframe(
                pd.DataFrame(QGIS_SCREENING_WORKFLOW, columns=["Order", "QGIS task", "How to do the check", "Output for this app"]),
                hide_index=True,
                width="stretch",
            )
            st.subheader("QGIS tool cookbook")
            st.dataframe(
                pd.DataFrame(QGIS_TOOL_COOKBOOK, columns=["Task", "QGIS / plugin tools", "Inputs", "Outputs", "Teaching note"]),
                hide_index=True,
                width="stretch",
            )
            st.subheader("Tutorial demo options")
            st.dataframe(
                pd.DataFrame(QGIS_TUTORIAL_DEMOS, columns=["Demo", "Starting data", "Tool sequence", "Student output", "How it connects to this app"]),
                hide_index=True,
                width="stretch",
            )
            template_df = pd.DataFrame(columns=QGIS_CANDIDATE_TEMPLATE_COLUMNS)
            st.download_button(
                "Download QGIS candidate CSV template",
                data=template_df.to_csv(index=False),
                file_name="phes_qgis_candidate_template.csv",
                mime="text/csv",
                width="stretch",
            )
            st.markdown("**Scientific and practice evidence**")
            render_reference_links(EVIDENCE_SOURCE_LINKS)
            st.markdown("**Open-source tools and adoptable repositories**")
            render_reference_links(OPEN_SOURCE_GIS_LINKS)
    elif step == 2:
        st.caption("Map candidate reservoir pairs, then select the reservoir arrangement and preliminary dam concept. Detailed level and dam-size calculations are in Step 3.")
        if not step1_complete():
            st.info("Complete Step 1 first (design power and operation duration) so screening tables can reference your energy target.")
        snowy_active = st.session_state.get("active_preset") in SNOWY_PRESET_NAMES
        tab_map, tab_evidence, tab_concept, tab_case = st.tabs(["Reservoir mapping", "Evidence screen", "Dam concept & alternatives", "Case study: Plateau vs Ravine"])
        with tab_map:
            st.subheader("Reservoir opportunity mapping")
            st.caption(
                "Use this as the handover from QGIS: one row per reservoir-pair option. "
                "If a value is unknown, leave it blank and mark the confidence low rather than inventing precision."
            )
            target_gwh = fnum('design_power_mw') * fnum('operation_hours') / 1000.0
            current_route_km = safe_div(fnum('penstock_length_m'), 1000.0)
            current_storage_gl = safe_div(fnum('reservoir_volume_m3'), 1e6)
            active_depth = levels["upper_hwl"] - levels["upper_lwl"] if levels_complete() else float("nan")
            current_area_km2 = safe_div(fnum('reservoir_volume_m3'), active_depth) / 1e6 if active_depth > 0 else float("nan")
            current_dam_height = max(levels["upper_hwl"] + st.session_state.freeboard_m + st.session_state.wave_allowance_m + st.session_state.settlement_allowance_m - fnum('dam_foundation_rl'), 0.0) if np.isfinite(fnum('dam_foundation_rl')) else float("nan")
            current_dam_volume_million_m3 = float("nan")
            if np.isfinite(current_dam_height):
                current_section_area = current_dam_height * (
                    st.session_state.dam_crest_width_m
                    + 0.5 * (st.session_state.upstream_slope_hv + st.session_state.downstream_slope_hv) * current_dam_height
                )
                current_dam_volume_million_m3 = current_section_area * st.session_state.dam_crest_length_m / 1e6
            candidate_df = pd.DataFrame(
                [
                    [
                        "Option A",
                        st.session_state.get("selected_screening_category", "Custom / field-mapped"),
                        "Custom",
                        np.nan,
                        np.nan,
                        levels["upper_nwl"],
                        levels["lower_twl"],
                        levels["gross_head"],
                        current_route_km,
                        target_gwh,
                        current_storage_gl,
                        current_area_km2,
                        current_dam_height,
                        st.session_state.dam_crest_length_m,
                        current_dam_volume_million_m3,
                        "Unknown",
                        "Unknown",
                        "Medium",
                        "Current benchmark or mapped student site",
                    ],
                    [
                        "Option B",
                        "Greenfield",
                        "Custom",
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        "Unknown",
                        "Unknown",
                        "Low",
                        "Student alternative",
                    ],
                ],
                columns=[
                    "Option",
                    "Source category",
                    "Storage class",
                    "Latitude",
                    "Longitude",
                    "Upper elevation m",
                    "Lower elevation m",
                    "Head m",
                    "Separation km",
                    "Storage GWh",
                    "Usable volume GL",
                    "Reservoir area (km²)",
                    "Dam height m",
                    "Dam length m",
                    "Dam rock volume (million m³)",
                    "Cost rank",
                    "Constraint flag",
                    "Data confidence",
                    "Notes",
                ],
            )
            edited = st.data_editor(
                candidate_df,
                num_rows="dynamic",
                width="stretch",
                key="candidate_site_screening_editor",
                column_config={
                    "Source category": st.column_config.SelectboxColumn("Source category", options=RESERVOIR_SOURCE_CATEGORIES),
                    "Storage class": st.column_config.SelectboxColumn("Storage class", options=STORAGE_DURATION_CLASSES),
                    "Cost rank": st.column_config.SelectboxColumn("Cost rank", options=SCREENING_COST_CLASSES),
                    "Constraint flag": st.column_config.SelectboxColumn(
                        "Constraint flag",
                        options=["Unknown", "No obvious fatal flag", "Protected area", "Heritage/cultural", "Waterway/ecology", "Land tenure/access", "Transmission/grid", "Other"],
                    ),
                    "Data confidence": st.column_config.SelectboxColumn("Data confidence", options=["Low", "Medium", "High"]),
                },
            )
            st.caption("Use the Evidence screen tab to tick the supporting evidence and see the readiness score.")
            screened = edited.copy()
            for column in ["Head m", "Separation km", "Usable volume GL", "Dam rock volume (million m³)"]:
                screened[column] = pd.to_numeric(screened[column], errors="coerce")
            screened["Head-to-distance m/km"] = np.where(screened["Separation km"] > 0, screened["Head m"] / screened["Separation km"], np.nan)
            screened["Water-to-rock ratio"] = np.where(
                screened["Dam rock volume (million m³)"] > 0,
                screened["Usable volume GL"] / screened["Dam rock volume (million m³)"],
                np.nan,
            )
            if np.isfinite(target_gwh):
                screened["Meets energy target"] = np.where(pd.to_numeric(screened["Storage GWh"], errors="coerce") >= target_gwh, "Yes", "No / unclear")
            metric_cols = ["Option", "Head-to-distance m/km", "Water-to-rock ratio"]
            if "Meets energy target" in screened.columns:
                metric_cols.append("Meets energy target")
            metric_cols.extend(["Cost rank", "Constraint flag", "Data confidence"])
            derived_metrics = screened[[col for col in metric_cols if col in screened.columns]].copy()
            st.caption("Derived metrics below use the edited table above. They are screening outputs, not another input table.")
            st.dataframe(
                derived_metrics,
                hide_index=True,
                width="stretch",
                column_config={
                    "Head-to-distance m/km": st.column_config.NumberColumn(format="%.2f"),
                    "Water-to-rock ratio": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.subheader("GIS screening deliverables")
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Reservoir-pair map", "Upper/lower reservoir candidates, footprints, contours, common datum, map scale"],
                        ["Dam and powerhouse siting", "Candidate dam axes, intake/outlet positions, powerhouse location, access/portal constraints"],
                        ["Waterway corridor", "At least one main alignment and one alternative with route length and environmental conflicts"],
                        ["Maximum plant discharge range", "Initial Q range from MW/head/efficiency to test route and intake feasibility"],
                        ["Data confidence", "DEM source, contour interval, storage data source, uncertainty and missing survey/geology data"],
                    ],
                    columns=["Deliverable", "Required content"],
                ),
                hide_index=True,
                width="stretch",
            )
        with tab_evidence:
            st.subheader("Screening categories and evidence fields")
            render_screening_limitation()
            render_step2_evidence_checklist()
            with st.expander("Reference table: screening categories"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            ["Greenfield", "New off-river upper and lower reservoirs", "Use for undeveloped reservoir pairs found from DEM/contours."],
                            ["Bluefield", "Existing reservoir paired with a new reservoir", "Useful where one storage already exists and can reduce civil/environmental scope."],
                            ["Brownfield", "Existing dam, hydro, mine pit, quarry, or modified storage", "Screen reuse opportunities before proposing wholly new reservoirs."],
                            ["Ocean", "Ocean used as one reservoir", "Coastal concepts need marine, corrosion, intake, and environmental checks."],
                            ["Seasonal", "Long-duration or seasonal storage concept", "Check whether the storage duration is actually required by the grid role."],
                            ["Turkey's Nest", "Ring-dike or perimeter embankment reservoir", "Relevant on plateau/ridge terrain with suitable foundation and fill."],
                        ],
                        columns=["Category", "Screening meaning", "How to use in this app"],
                    ),
                    hide_index=True,
                    width="stretch",
                )
            with st.expander("Reference table: storage-duration classes"):
                st.dataframe(
                    pd.DataFrame({"Storage class": STORAGE_DURATION_CLASSES[:-1]}),
                    hide_index=True,
                    width="stretch",
                )
            with st.expander("Reference table: open-source QGIS adoption path"):
                st.dataframe(
                    pd.DataFrame(QGIS_SCREENING_WORKFLOW, columns=["Order", "QGIS task", "How to do the check", "Output for this app"]),
                    hide_index=True,
                    width="stretch",
                )
            with st.expander("Reference table: tool cookbook for the lab"):
                st.dataframe(
                    pd.DataFrame(QGIS_TOOL_COOKBOOK, columns=["Task", "QGIS / plugin tools", "Inputs", "Outputs", "Teaching note"]),
                    hide_index=True,
                    width="stretch",
                )
            with st.expander("Reference table: working demo exercises"):
                st.dataframe(
                    pd.DataFrame(QGIS_TUTORIAL_DEMOS, columns=["Demo", "Starting data", "Tool sequence", "Student output", "How it connects to this app"]),
                    hide_index=True,
                    width="stretch",
                )
            st.download_button(
                "Download candidate CSV template",
                data=pd.DataFrame(columns=QGIS_CANDIDATE_TEMPLATE_COLUMNS).to_csv(index=False),
                file_name="phes_qgis_candidate_template.csv",
                mime="text/csv",
                width="stretch",
            )
            with st.expander("Scientific, practice, and open-source links"):
                st.markdown("**Scientific and practice evidence**")
                render_reference_links(EVIDENCE_SOURCE_LINKS)
                st.markdown("**Open-source tools and adoptable repositories**")
                render_reference_links(OPEN_SOURCE_GIS_LINKS)
        with tab_concept:
            st.subheader("Reservoir and dam concept selection")
            st.caption("Select the reservoir and dam concept from site evidence.")
            for key, options in {
                "reservoir_arrangement": RESERVOIR_ARRANGEMENTS,
                "valley_geometry": VALLEY_GEOMETRIES,
                "foundation_quality": FOUNDATION_QUALITIES,
                "construction_material": CONSTRUCTION_MATERIALS,
            }.items():
                ensure_option_state(key, options)
            ensure_option_state("selected_screening_category", RESERVOIR_SOURCE_CATEGORIES)
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.selectbox(
                    "Reservoir arrangement",
                    RESERVOIR_ARRANGEMENTS,
                    key="reservoir_arrangement",
                    help="Choose whether the project reuses storage, creates a new valley/off-river reservoir, uses a mine/quarry void, or forms a ring-dike storage.",
                )
            with c2:
                st.selectbox(
                    "Valley geometry",
                    VALLEY_GEOMETRIES,
                    key="valley_geometry",
                    help="This controls which dam families are physically plausible: broad valleys favour embankments; narrow competent valleys favour concrete options.",
                )
            with c3:
                st.selectbox(
                    "Foundation quality",
                    FOUNDATION_QUALITIES,
                    key="foundation_quality",
                    help="A new-dam concept needs at least a desktop foundation assumption. Unknown foundation quality blocks a confident dam-family recommendation.",
                )
            with c4:
                st.selectbox(
                    "Construction material",
                    CONSTRUCTION_MATERIALS,
                    key="construction_material",
                    help="Material availability controls whether earthfill, rockfill/CFRD, RCC/gravity, or existing-asset reuse is plausible.",
                )
            with c5:
                st.selectbox("Source category", RESERVOIR_SOURCE_CATEGORIES, key="selected_screening_category")
            concept_dam_height = max(levels["upper_hwl"] - fnum('dam_foundation_rl'), 0.0)
            dam_type_choice = recommend_dam_type(
                st.session_state.reservoir_arrangement,
                st.session_state.valley_geometry,
                st.session_state.foundation_quality,
                st.session_state.construction_material,
                concept_dam_height,
                fnum('reservoir_volume_m3'),
            )
            st.metric("Preliminary dam type", dam_type_choice)
            st.caption(DAM_TYPE_GUIDANCE[dam_type_choice])
            if dam_type_choice == "Further site investigation required":
                st.warning(
                    "The app is withholding a dam-family recommendation because the current evidence is not strong enough. "
                    "Update foundation quality, dam axis evidence, material source and Step 2 evidence readiness before treating a dam type as selected."
                )
            st.markdown("**Why this concept was selected**")
            st.caption(
                "The app applies these rules in priority order. The first row marked 'Governing rule' determines the preliminary dam type; later rows are useful checks but do not override it."
            )
            st.dataframe(
                pd.DataFrame(
                    dam_selection_trace_rows(
                        st.session_state.reservoir_arrangement,
                        st.session_state.valley_geometry,
                        st.session_state.foundation_quality,
                        st.session_state.construction_material,
                        concept_dam_height,
                        fnum('reservoir_volume_m3'),
                    ),
                    columns=["Priority", "Criterion", "Current input", "Rule result", "Selected outcome", "Teaching note"],
                ),
                hide_index=True,
                width="stretch",
            )
            with st.expander("Selection criteria reference"):
                st.dataframe(
                    pd.DataFrame(DAM_SELECTION_CRITERIA, columns=["Criterion", "Why it matters", "Decision effect"]),
                    hide_index=True,
                    width="stretch",
                )
            render_matplotlib_figure(dam_concept_schematic_figure(dam_type_choice))
            st.dataframe(
                pd.DataFrame(
                    DAM_CONCEPT_DEFINITIONS,
                    columns=["Dam concept", "Plain-language definition", "When it is plausible", "Main checks later"],
                ),
                hide_index=True,
                width="stretch",
            )
            st.subheader("Comparative optimisation of alternatives")
            if snowy_active:
                alt_df = pd.DataFrame(
                    [
                        ["Snowy Plateau teaching model", "Brownfield", "Tantangara - Talbingo", "Existing reservoirs reused", "Course-defined plateau concept", fnum('penstock_length_m'), design_case["total_discharge_m3_s"], np.nan, np.nan, "Unknown", "No rank: geology, transients, access, impacts and cost require comparable evidence"],
                        ["Snowy Ravine teaching alternative", "Brownfield", "Tantangara - Talbingo", "Existing reservoirs reused", "Course-defined ravine concept", fnum('penstock_length_m'), design_case["total_discharge_m3_s"], np.nan, np.nan, "Unknown", "No rank: geometry and development claims require primary evidence"],
                    ],
                    columns=["Option", "Source category", "Reservoir pair", "Dam site", "Powerhouse site", "Waterway length m", "Max plant Q (m³/s)", "Cost $/kW", "Cost $/kWh", "Desktop cost rank", "Economic evidence / note"],
                )
                st.caption("Seeded with the course-defined Plateau-versus-Ravine comparison. Complete the evidence columns before ranking either option.")
            else:
                alt_df = pd.DataFrame(
                    [
                        ["Option A", st.session_state.selected_screening_category, "Selected upper/lower pair", "Preferred dam axis", "Underground / surface", fnum('penstock_length_m'), design_case["total_discharge_m3_s"], np.nan, np.nan, "Unknown", "Current benchmark"],
                        ["Option B", "Greenfield", "Alternative pair", "Alternative dam axis", "Alternative powerhouse", np.nan, np.nan, np.nan, np.nan, "Unknown", "Student alternative"],
                    ],
                    columns=["Option", "Source category", "Reservoir pair", "Dam site", "Powerhouse site", "Waterway length m", "Max plant Q (m³/s)", "Cost $/kW", "Cost $/kWh", "Desktop cost rank", "Economic evidence / note"],
                )
            st.data_editor(
                alt_df,
                num_rows="dynamic",
                width="stretch",
                key="concept_alternative_editor",
                column_config={
                    "Waterway length m": st.column_config.NumberColumn(format="%.0f"),
                    "Max plant Q (m³/s)": st.column_config.NumberColumn(format="%.1f"),
                    "Cost $/kW": st.column_config.NumberColumn(format="%.0f"),
                    "Cost $/kWh": st.column_config.NumberColumn(format="%.0f"),
                    "Source category": st.column_config.SelectboxColumn("Source category", options=RESERVOIR_SOURCE_CATEGORIES),
                    "Desktop cost rank": st.column_config.SelectboxColumn("Desktop cost rank", options=SCREENING_COST_CLASSES),
                },
            )
            st.caption("Use this table to justify why the selected dam, powerhouse, waterway route, discharge and storage duration are better than the main alternative.")
        with tab_case:
            st.subheader("Teaching comparison: Snowy Plateau vs Ravine concepts")
            if not snowy_active:
                st.info(
                    "Apply one of the Snowy 2.0 presets in Step 1 to activate this worked example. "
                    "The two presets differ only in the lower tailwater level (538.8 m vs 534.4 m), "
                    "so the choice between them cannot be made on the energy numbers alone."
                )
            else:
                plateau_twl = PRESETS["Snowy 2.0 - Plateau"]["lower_twl"]
                ravine_twl = PRESETS["Snowy 2.0 - Ravine"]["lower_twl"]
                hg_plateau = levels["upper_nwl"] - plateau_twl
                hg_ravine = levels["upper_nwl"] - ravine_twl
                m1, m2, m3 = st.columns(3)
                m1.metric("Representative head - Plateau", metric_value(hg_plateau, " m", 1))
                m2.metric("Representative head - Ravine", metric_value(hg_ravine, " m", 1))
                m3.metric("Difference", metric_value(hg_ravine - hg_plateau, " m", 1))
                st.caption(
                    "Both options deliver the same power and storage from the same reservoir pair, so the "
                    "head and energy arithmetic alone cannot select an option. Any ranking must use comparable "
                    "geology (Step 7), connected-system transients (Step 8), approvals, access and dated cost evidence (Step 10)."
                )
                st.dataframe(
                    pd.DataFrame(SNOWY_CASE_STUDY_ROWS, columns=["Priority", "Factor", "Plateau option", "Ravine option", "Design lesson"]),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Course-defined comparison only. No preferred option is established until the evidence fields are completed; "
                    "students should build the same traceable table for their own alternatives in the dam concept tab."
                )
    elif step == 3:
        st.caption("Quantify the Step 2 concept: reservoir levels, dam sizing, gross head, storage energy, and duration.")
        evidence_summary = current_step2_evidence_summary()
        if int(evidence_summary["ready_count"]) < 4:
            st.warning(
                f"Step 2 evidence readiness is {int(evidence_summary['ready_count'])}/{int(evidence_summary['total'])}. "
                "Treat the Step 3 dimensions as provisional until reservoir footprints, operating levels, storage and route evidence are ticked in Step 2."
            )
        elif int(evidence_summary["ready_count"]) < 6:
            st.info(
                f"Step 2 evidence readiness is {int(evidence_summary['ready_count'])}/{int(evidence_summary['total'])}. "
                "Proceed with Step 3, but keep missing evidence visible in the report and risk register."
            )
        else:
            st.caption(f"Step 2 evidence readiness: {int(evidence_summary['ready_count'])}/{int(evidence_summary['total'])} ({evidence_summary['level']}).")
        with section_panel("input", "Reservoir level and storage inputs"):
            l1, l2, l3, l4, l5 = st.columns(5)
            with l1:
                st.number_input("Upper HWL (m)", min_value=0.0, max_value=3000.0, step=1.0, key="upper_hwl")
            with l2:
                st.number_input("Upper LWL (m)", min_value=0.0, max_value=3000.0, step=1.0, key="upper_lwl")
            with l3:
                st.number_input(
                    "Lower HWL / FSL (m)",
                    min_value=0.0,
                    max_value=3000.0,
                    step=1.0,
                    key="lower_hwl",
                    help="Lower reservoir high water level or full supply level used for operating-range checks.",
                )
            with l4:
                st.number_input(
                    "Lower TWL (m)",
                    min_value=0.0,
                    max_value=3000.0,
                    step=1.0,
                    key="lower_twl",
                    help="Tailwater level used directly in the gross-head calculation. If no tailwater curve is available, estimate it from lower FSL and MOL before entering it here.",
                )
            with l5:
                st.number_input("Active storage volume (m³)", min_value=0.0, step=1_000_000.0, format="%.0f", key="reservoir_volume_m3")
            levels = reservoir_levels()

        if not (levels_complete() and has_values("reservoir_volume_m3")):
            st.info("Enter both reservoirs' high and low operating levels plus active storage to calculate the simultaneous head envelope, a labelled representative head, storage energy, duration and HFR.")
            render_step_guidance(step)
            st.stop()

        active_depth = max(levels["upper_hwl"] - levels["upper_lwl"], 0.0)
        tailwater_average = 0.5 * (levels["lower_hwl"] + levels["lower_twl"])
        eta_gen = st.session_state.eta_turbine * st.session_state.eta_generator * st.session_state.eta_transformer
        e_gross_gwh = RHO * G * fnum('reservoir_volume_m3') * levels["gross_head"] * eta_gen / 3.6e12
        e_deliverable_gwh = RHO * G * fnum('reservoir_volume_m3') * fnum('teaching_effective_head_m') * eta_gen / 3.6e12
        duration = safe_div(e_deliverable_gwh * 1000.0, fnum('design_power_mw'))

        with section_panel(
            "output",
            "Reservoir level outputs",
            "Equations are consolidated in the bottom Equations, annotations, and reference table. The tables below show the calculated values that feed Step 4 discharge and Step 5 hydraulic-loss checks.",
        ):
            level_output_rows = pd.DataFrame(
                [
                    ["Representative upper level", metric_value(levels["upper_representative"], " m", 1), "One-third upper drawdown placeholder; replace when an operating rule or storage curve supports another level."],
                    ["Representative lower level", metric_value(levels["lower_representative"], " m", 1), "Current teaching case uses entered lower TWL; state the operating condition."],
                    ["Representative gross head", metric_value(levels["gross_head"], " m", 1), "Representative Step 3 output carried into Step 4."],
                    ["Minimum simultaneous gross head", metric_value(levels["gross_head_min"], " m", 1), "Upper LWL minus lower HWL."],
                    ["Maximum simultaneous gross head", metric_value(levels["gross_head_max"], " m", 1), "Upper HWL minus lower LWL/TWL."],
                    ["TWL approximation check", metric_value(tailwater_average, " m", 1), "Use only as a check if TWL is estimated from lower FSL/MOL."],
                    ["Head-range ratio HFR", metric_value(levels["head_fluctuation_ratio"], "", 3), "Minimum simultaneous gross head divided by maximum simultaneous gross head."],
                ],
                columns=["Output", "Value", "Design use"],
            )
            storage_output_rows = pd.DataFrame(
                [
                    ["Active storage", metric_value(fnum('reservoir_volume_m3') / 1e6, " GL", 1), "Water volume available between operating levels."],
                    ["Gross-head energy", metric_value(e_gross_gwh, " GWh", 1), "Storage energy using Step 3 gross head."],
                    ["First-pass effective-head energy", metric_value(e_deliverable_gwh, " GWh", 1), "Pre-loss energy using the current Step 4 head; Step 5 net head controls the final headline value."],
                    ["First-pass full-load duration", metric_value(duration, " h", 1), "Pre-loss duration check; update after Step 5."],
                ],
                columns=["Output", "Value", "Design use"],
            )
            output_column_config = {
                "Output": st.column_config.TextColumn(width="medium"),
                "Value": st.column_config.TextColumn(width="small"),
                "Design use": st.column_config.TextColumn(width="large"),
            }
            st.markdown("**Level and head outputs**")
            st.dataframe(level_output_rows, hide_index=True, width="stretch", column_config=output_column_config)
            st.markdown("**Storage and energy outputs**")
            st.dataframe(storage_output_rows, hide_index=True, width="stretch", column_config=output_column_config)
            with st.expander("Open calculation trace"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            ["Available drawdown", f"{levels['upper_hwl']:.1f} - {levels['upper_lwl']:.1f}", metric_value(active_depth, " m", 1), "Upper-reservoir operating band."],
                            ["Representative upper level", f"{levels['upper_hwl']:.1f} - {active_depth:.1f}/3", metric_value(levels["upper_representative"], " m", 1), "Course placeholder only; not a universal definition of NWL."],
                            ["Tailwater approximation", f"({levels['lower_hwl']:.1f} + {levels['lower_twl']:.1f})/2", metric_value(tailwater_average, " m", 1), "Reference check when a tailwater curve is unavailable."],
                            ["Representative gross head", f"{levels['upper_representative']:.1f} - {levels['lower_representative']:.1f}", metric_value(levels["gross_head"], " m", 1), "Carried to Step 4 as the representative gross-head screen."],
                            ["Maximum simultaneous head", f"{levels['upper_hwl']:.1f} - {levels['lower_twl']:.1f}", metric_value(levels["gross_head_max"], " m", 1), "Upper high and lower low at the same operating condition."],
                            ["Minimum simultaneous head", f"{levels['upper_lwl']:.1f} - {levels['lower_hwl']:.1f}", metric_value(levels["gross_head_min"], " m", 1), "Upper low and lower high at the same operating condition."],
                            ["Head-range ratio", f"{levels['gross_head_min']:.1f}/{levels['gross_head_max']:.1f}", metric_value(levels["head_fluctuation_ratio"], "", 3), "Simultaneous minimum-to-maximum head ratio."],
                            ["First-pass stored energy", f"V={fnum('reservoir_volume_m3')/1e6:.1f} GL; H={fnum('teaching_effective_head_m'):.1f} m", metric_value(e_deliverable_gwh, " GWh", 1), "Uses the Step 4 selected head; replace with Step 5 net-head energy in the final report."],
                        ],
                        columns=["Output", "Current substitution", "Calculated value", "Design use"],
                    ),
                    hide_index=True,
                    width="stretch",
                )

        with section_panel("input", "Dam sizing inputs"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.number_input("Dam foundation RL (m)", min_value=0.0, max_value=3000.0, step=1.0, key="dam_foundation_rl")
            with c2:
                st.number_input("Freeboard Fb (m)", min_value=0.0, max_value=20.0, step=0.1, key="freeboard_m")
            with c3:
                st.number_input("Wave allowance Hw (m)", min_value=0.0, max_value=10.0, step=0.1, key="wave_allowance_m")
            with c4:
                st.number_input("Settlement allowance Sa (m)", min_value=0.0, max_value=10.0, step=0.1, key="settlement_allowance_m")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.number_input("Dam crest length (m)", min_value=1.0, max_value=10_000.0, step=10.0, key="dam_crest_length_m")
            with g2:
                st.number_input("Dam crest width (m)", min_value=1.0, max_value=100.0, step=0.5, key="dam_crest_width_m")
            with g3:
                st.number_input("Upstream slope H:V", min_value=0.1, max_value=10.0, step=0.1, key="upstream_slope_hv")
            with g4:
                st.number_input("Downstream slope H:V", min_value=0.1, max_value=10.0, step=0.1, key="downstream_slope_hv")
            with st.expander("Open dam-input guidance"):
                st.dataframe(
                    pd.DataFrame(DAM_DIMENSION_INPUT_GUIDANCE, columns=["Input", "How to choose it", "Teaching note"]),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "The app volume is a trapezoidal screening quantity, not a detailed bill of quantities. "
                    "Use it to compare options, then verify the selected dam with civil/geotechnical modelling."
                )
        if not has_values("dam_foundation_rl"):
            st.info("Enter the dam foundation RL to calculate crest level, dam height, dam volume and civil screening quantities.")
            st.stop()
        crest_level = levels["upper_hwl"] + st.session_state.freeboard_m + st.session_state.wave_allowance_m + st.session_state.settlement_allowance_m
        dam_height = max(crest_level - fnum('dam_foundation_rl'), 0.0)
        reservoir_area = safe_div(fnum('reservoir_volume_m3'), active_depth)
        section_area = dam_height * (st.session_state.dam_crest_width_m + 0.5 * (st.session_state.upstream_slope_hv + st.session_state.downstream_slope_hv) * dam_height)
        dam_volume = section_area * st.session_state.dam_crest_length_m
        with section_panel("output", "Dam sizing outputs"):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Crest level", metric_value(crest_level, " m", 1))
            m2.metric("Dam height", metric_value(dam_height, " m", 1))
            m3.metric("Active area", metric_value(reservoir_area / 1e6, " km²", 2))
            m4.metric("Dam volume", metric_value(dam_volume / 1e6, " million m³", 2))
            st.caption(
                "Dam volume is a trapezoidal screening estimate from crest width, side slopes, height and crest length. "
                "Use it for concept comparison only; final dimensions require dam-type-specific stability, seepage and foundation verification."
            )
            st.markdown("**Desktop storage and civil screening**")
            water_to_rock = safe_div(fnum('reservoir_volume_m3'), dam_volume)
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Storage GWh", e_deliverable_gwh, "GWh", "Compare with the storage-duration class selected in Step 2."],
                        ["Usable volume", fnum('reservoir_volume_m3') / 1e6, "GL", "Candidate-table usable water volume."],
                        ["Reservoir area", reservoir_area / 1e6, "km²", "Screen against mapped reservoir footprint from contours or DEM."],
                        ["Dam rock/fill volume", dam_volume / 1e6, "million m³", "Civil quantity proxy for cost ranking."],
                        ["Water-to-rock ratio", water_to_rock, "-", "Higher values generally indicate a more efficient storage geometry."],
                        ["Desktop cost rank", np.nan, "-", "Enter the desktop or project-screening rank in Step 2 and Step 10."],
                    ],
                    columns=["Screening field", "Value", "Unit", "Design use"],
                ),
                hide_index=True,
                width="stretch",
                column_config={"Value": st.column_config.NumberColumn(format="%.2f")},
            )
        st.subheader("Reservoir level schematic")
        render_reservoir_level_schematic(levels, crest_level)
        st.subheader("Dam dimension schematic")
        render_matplotlib_figure(dam_dimension_schematic_figure(levels, crest_level, dam_height, dam_volume))
        with st.expander("Dam design verification after the screening dimensions"):
            st.dataframe(
                pd.DataFrame(DAM_MODELLING_VERIFICATION, columns=["Verification", "Use when", "Output students should report"]),
                hide_index=True,
                width="stretch",
            )
            st.info(
                "Before finalising dimensions or stability, verify the selected dam type with numerical or limit-equilibrium modelling. "
                "For non-civil design reports, state the modelling method that would be required and which assumptions are still unverified."
            )
    elif step == 4:
        st.caption("Draw the ground profile and the selected waterway alignment first, then size the design discharge. Use the waterway RL line for hydraulic length, not only the ground topography.")
        if not levels_complete():
            st.info("Enter the reservoir levels in Step 3 first: the waterway alignment template uses the representative upper level, lower tailwater and turbine-setting geometry.")
            render_step_guidance(step)
            st.stop()
        st.markdown(
            '<div class="equation-label">Turbine centreline setting below tailwater <span>h<sub>set</sub></span> (m)</div>',
            unsafe_allow_html=True,
        )
        st.number_input(
            "Turbine centreline setting below tailwater",
            min_value=0.0,
            max_value=300.0,
            step=1.0,
            key="draft_head_m",
            help="Use tailwater level minus turbine centreline. This is a geometry/cavitation input and is not subtracted as a hydraulic loss.",
            label_visibility="collapsed",
        )
        with st.expander("How to choose the turbine setting"):
            st.caption("The turbine-setting relationship is listed in the bottom Step 4 Equations, annotations, and reference table.")
            st.dataframe(
                pd.DataFrame(DRAFT_HEAD_GUIDANCE, columns=["Situation", "Suggested value", "How to report it"]),
                hide_index=True,
                width="stretch",
            )
            st.markdown(
                r"In this app, $h_{set}$ locates the turbine centreline for powerhouse and cavitation checks. "
                r"It is not a Darcy, local, or other hydraulic loss. For a new project, use 0 m only as an explicit geometry placeholder until a layout or vendor basis exists."
            )
        profile = pd.DataFrame(
            [
                ["Upper intake", 0.0, levels["upper_nwl"] + 15.0, levels["upper_nwl"]],
                ["Headrace tunnel", 3500.0, levels["upper_nwl"] - 80.0, levels["upper_nwl"] - 25.0],
                ["Pressure shaft", 8500.0, levels["upper_nwl"] - 260.0, levels["lower_twl"] + 210.0],
                ["Powerhouse", 13500.0, levels["lower_twl"] + 120.0, levels["lower_twl"] + 45.0],
                ["Tailrace outlet", 18000.0, levels["lower_twl"] + 25.0, levels["lower_twl"]],
            ],
            columns=["Element", "Chainage_m", "Ground_RL_m", "Waterway_RL_m"],
        )
        profile = st.data_editor(
            profile,
            num_rows="dynamic",
            width="stretch",
            key="waterway_alignment_editor",
            column_config={
                "Element": st.column_config.SelectboxColumn(
                    "Element",
                    options=["Upper intake", "Headrace tunnel", "Surge tank", "Pressure shaft", "Powerhouse", "Tailrace tunnel", "Tailrace outlet", "Access adit", "Other"],
                ),
                "Chainage_m": st.column_config.NumberColumn("Chainage (m)", min_value=0.0, step=100.0, format="%.0f"),
                "Ground_RL_m": st.column_config.NumberColumn("Ground RL (m)", step=1.0, format="%.1f"),
                "Waterway_RL_m": st.column_config.NumberColumn("Waterway RL (m)", step=1.0, format="%.1f"),
            },
        )
        alignment = render_waterway_alignment_schematic(profile)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Waterway length", metric_value(alignment["profile_length_m"], " m", 1))
        m2.metric("Plan chainage", metric_value(alignment["plan_length_m"], " m", 1))
        m3.metric("Alignment drop", metric_value(alignment["drop_m"], " m", 1))
        m4.metric("Average gradient", metric_value(alignment["average_gradient"] * 100.0, "%", 2))
        turbine_cl = levels["lower_twl"] - fnum('draft_head_m')
        cover_rows = []
        clean_profile = profile.copy()
        for column in ["Chainage_m", "Ground_RL_m", "Waterway_RL_m"]:
            clean_profile[column] = pd.to_numeric(clean_profile[column], errors="coerce")
        clean_profile = clean_profile.dropna(subset=["Ground_RL_m", "Waterway_RL_m"])
        if not clean_profile.empty:
            clean_profile["Cover_m"] = clean_profile["Ground_RL_m"] - clean_profile["Waterway_RL_m"]
            critical = clean_profile.loc[clean_profile["Cover_m"].idxmin()]
            cover_rows.append(["Minimum cover from edited profile", critical["Cover_m"], "m", f"At {critical['Element']}"])
        cover_rows.extend(
            [
                ["Turbine centreline CL", turbine_cl, "m", "Uses lower tailwater and the centreline-setting depth."],
                ["Turbine setting below tailwater", fnum('draft_head_m'), "m", "Geometry/cavitation input only; not deducted from hydraulic head"],
                ["Required GIS evidence", np.nan, "-", "Centreline polyline should follow chosen horizontal and vertical alignment, not just terrain."],
            ]
        )
        st.dataframe(pd.DataFrame(cover_rows, columns=["Check", "Value", "Unit", "Annotation"]), hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Powerhouse datum relationship is listed in the bottom Step 4 equation table.")
        st.subheader("Design discharge and head basis")
        page_design_discharge(levels, embedded=True)
    elif step == 5:
        st.caption("Shared conduit losses use total plant discharge. Select roughness and local-loss components from tables, then calculate Reynolds number and Darcy friction factor.")
        with section_panel("input", "Shared conduit and roughness inputs"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.number_input("Conduit length L (m)", min_value=1.0, max_value=100_000.0, step=10.0, key="penstock_length_m")
            with c2:
                st.number_input("Shared diameter D (m)", min_value=0.2, max_value=30.0, step=0.1, key="penstock_diameter_m")
            with c3:
                st.number_input("Water temperature (deg C)", min_value=0.0, max_value=60.0, step=0.5, key="temperature_c")
            with c4:
                st.radio("Flow area basis", ["Shared conduit", "Per penstock"], horizontal=False, key="flow_area_mode")

            material = st.selectbox("Material roughness for Darcy f", list(ROUGHNESS.keys()), key="roughness_material")
            if material == "Custom":
                st.number_input("Custom absolute roughness ε (m)", min_value=0.0, max_value=0.05, step=0.00001, format="%.6f", key="roughness_m")
            else:
                st.session_state.roughness_m = ROUGHNESS[material]

            st.number_input(
                "Other/non-conduit hydraulic loss allowance h_other (m)",
                min_value=0.0,
                max_value=500.0,
                step=0.1,
                key="other_head_loss_m",
                help="Use only for a separately justified hydraulic allowance not already represented by Darcy friction or the selected local-loss coefficients. Do not enter turbine setting depth here.",
            )

            with st.expander("Open roughness reference table"):
                roughness_df = pd.DataFrame(
                    {
                        "Material": [key for key, value in ROUGHNESS.items() if value is not None],
                        "Absolute roughness ε (m)": [value for value in ROUGHNESS.values() if value is not None],
                    }
                )
                st.dataframe(roughness_df, hide_index=True, width="stretch", column_config={"Absolute roughness ε (m)": st.column_config.NumberColumn(format="%.6f")})

        with section_panel("input", "Local loss component inputs"):
            with st.expander("Open local-loss and velocity guidance"):
                st.dataframe(
                    pd.DataFrame(LOSS_LAYOUT_GUIDANCE, columns=["Layout", "Typical selection", "How to use"]),
                    hide_index=True,
                    width="stretch",
                )
                st.dataframe(
                    pd.DataFrame(VELOCITY_GUIDANCE, columns=["Waterway part", "Reference velocity range", "Design note"]),
                    hide_index=True,
                    width="stretch",
                )
                st.caption("Start with the nearest preset below, then delete components that are not present in your own Step 4 layout.")
            preset_cols = st.columns(len(LOSS_COMPONENT_PRESETS))
            for i, preset_name in enumerate(LOSS_COMPONENT_PRESETS):
                with preset_cols[i]:
                    if st.button(preset_name, key=f"loss_preset_{i}", width="stretch"):
                        apply_loss_component_preset(preset_name)
                        st.success(f"Applied {preset_name}.")
            k_sum = render_loss_component_editor()
        design_case = hydraulic_snapshot(fnum('design_power_mw'), levels["gross_head"])
        if not (hydraulics_ready() and np.isfinite(design_case["velocity_m_s"])):
            st.info("Loss results appear once the design power (Step 1), reservoir levels (Step 3), effective head (Step 4), and the conduit length and diameter above are all set.")
            render_step_guidance(step)
            st.stop()
        rel_roughness = safe_div(st.session_state.roughness_m, fnum('penstock_diameter_m'))
        regime = reynolds_regime(design_case["reynolds"])
        method = friction_method_label(design_case["reynolds"])
        with section_panel("output", "Hydraulic loss outputs"):
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("ΣK", metric_value(k_sum, "", 2))
            m2.metric("Velocity", metric_value(design_case["velocity_m_s"], " m/s", 2))
            m3.metric("Reynolds Re", metric_value(design_case["reynolds"], "", 0))
            m4.metric("Darcy f", metric_value(design_case["friction"], "", 4))
            m5.metric("Conduit loss", metric_value(design_case["conduit_loss_m"], " m", 2))
            m6.metric("Total deduction", metric_value(design_case["total_loss_m"], " m", 2))
            m7.metric("Net head", metric_value(design_case["net_head_m"], " m", 1))
            if np.isfinite(design_case["net_head_m"]):
                head_delta_percent = safe_div(design_case["net_head_m"] - fnum('teaching_effective_head_m'), fnum('teaching_effective_head_m')) * 100.0
                st.info(
                    f"Step 5 civil net head is {metric_value(design_case['net_head_m'], ' m', 1)}. "
                    f"It differs from the Step 4 selected first-pass head by {metric_value(head_delta_percent, '%', 1)}. "
                    "If the difference is more than about 5-10%, go back to Step 4 and recalculate design discharge."
                )

            calc_rows = pd.DataFrame(
                [
                    ["1. Flow area", "Area from the selected conduit diameter.", area_circle(fnum('penstock_diameter_m')), "m²", "Shared conduit area unless per-penstock basis is selected."],
                    ["2. Flow for velocity", "Flow basis selected for this conduit.", design_case["flow_for_velocity_m3_s"], "m³/s", design_case["flow_area_mode"]],
                    ["3. Velocity", "Selected flow divided by conduit area.", design_case["velocity_m_s"], "m/s", "Velocity drives both friction and local losses."],
                    ["4. Reynolds number", "Velocity, diameter and water viscosity.", design_case["reynolds"], "-", regime],
                    ["5. Relative roughness", "Absolute roughness divided by diameter.", rel_roughness, "-", f"Roughness = {st.session_state.roughness_m:.6f} m"],
                    ["6. Darcy friction factor", "Laminar rule or turbulent Swamee-Jain/Colebrook approximation.", design_case["friction"], "-", method],
                    ["7. Major loss", "Friction loss along the conduit length.", design_case["major_loss_m"], "m", "Pipe/tunnel friction loss."],
                    ["8. Minor/local loss", "Local-loss coefficient sum for selected components.", design_case["minor_loss_m"], "m", "Selected fittings and transitions."],
                    ["9. Other hydraulic loss", "Separately declared allowance not represented above.", design_case["other_loss_m"], "m", "Must have a distinct source; turbine setting is not a loss."],
                    ["10. Net head", "Selected Step 4 head minus major, minor and other hydraulic losses.", design_case["net_head_m"], "m", "Operating head carried into Step 9."],
                ],
                columns=["Calculation step", "Calculation basis", "Value", "Unit", "Annotation"],
            )
            st.caption(
                "The full Step 5 equations are kept in the bottom Equations, annotations, and reference table; "
                "this table shows the numerical trace without unrendered formula text."
            )
            st.dataframe(
                calc_rows,
                hide_index=True,
                width="stretch",
                column_config={"Value": st.column_config.NumberColumn(format="%.4g")},
            )
        st.subheader("Reach-by-reach loss schedule")
        reach_df = pd.DataFrame(
            [
                ["Headrace tunnel", "Total plant Q", np.nan, fnum('penstock_length_m') * 0.55, fnum('penstock_diameter_m'), st.session_state.roughness_m, 1.2],
                ["Pressure shaft", "Total plant Q", np.nan, fnum('penstock_length_m') * 0.25, max(fnum('penstock_diameter_m') * 0.8, 0.2), st.session_state.roughness_m, 1.0],
                ["Unit penstock/branch", "Per unit Q", np.nan, fnum('penstock_length_m') * 0.20, max(fnum('unit_penstock_diameter_m'), 0.2), 0.000045, 1.5],
            ],
            columns=["Reach", "Flow basis", "Custom Q (m³/s)", "Length m", "Diameter m", "Roughness ε (m)", "Local K"],
        )
        reach_df = st.data_editor(
            reach_df,
            num_rows="dynamic",
            width="stretch",
            key="reach_loss_editor",
            column_config={
                "Flow basis": st.column_config.SelectboxColumn(
                    "Flow basis",
                    options=["Total plant Q", "Per unit Q", "Per penstock Q", "Custom Q"],
                    help="Use total Q for shared waterways and per-unit/per-penstock Q after branches.",
                ),
                "Custom Q (m³/s)": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.2f"),
                "Length m": st.column_config.NumberColumn(min_value=0.0, step=100.0, format="%.0f"),
                "Diameter m": st.column_config.NumberColumn(min_value=0.2, step=0.1, format="%.2f"),
                "Roughness ε (m)": st.column_config.NumberColumn(min_value=0.0, step=0.00001, format="%.6f"),
                "Local K": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.2f"),
            },
        )
        reach_rows = []
        for _, row in reach_df.iterrows():
            length = float(row["Length m"]) if pd.notna(row["Length m"]) else 0.0
            diameter = float(row["Diameter m"]) if pd.notna(row["Diameter m"]) else float("nan")
            roughness = float(row["Roughness ε (m)"]) if pd.notna(row["Roughness ε (m)"]) else 0.0
            k_local = float(row["Local K"]) if pd.notna(row["Local K"]) else 0.0
            custom_q = float(row["Custom Q (m³/s)"]) if "Custom Q (m³/s)" in row and pd.notna(row["Custom Q (m³/s)"]) else float("nan")
            q_reach = flow_for_reach_basis(str(row.get("Flow basis", "Total plant Q")), design_case["total_discharge_m3_s"], custom_q)
            velocity = safe_div(q_reach, area_circle(diameter))
            re = safe_div(velocity * diameter, water_nu_kinematic_m2_s(st.session_state.temperature_c))
            f = f_swamee_jain(re, safe_div(roughness, diameter))
            major = head_loss(f, length, diameter, velocity, 0.0)
            minor = head_loss(0.0, length, diameter, velocity, k_local)
            reach_rows.append([row["Reach"], row.get("Flow basis", "Total plant Q"), q_reach, velocity, re, f, major, minor, major + minor])
        st.dataframe(
            pd.DataFrame(reach_rows, columns=["Reach", "Flow basis", "Q (m³/s)", "Velocity m/s", "Re", "Darcy f", "Major loss m", "Minor loss m", "Total loss m"]),
            hide_index=True,
            width="stretch",
            column_config={
                "Q (m³/s)": st.column_config.NumberColumn(format="%.2f"),
                "Velocity m/s": st.column_config.NumberColumn(format="%.2f"),
                "Re": st.column_config.NumberColumn(format="%.2e"),
                "Darcy f": st.column_config.NumberColumn(format="%.5f"),
                "Major loss m": st.column_config.NumberColumn(format="%.2f"),
                "Minor loss m": st.column_config.NumberColumn(format="%.2f"),
                "Total loss m": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption("Use Re < 2000 as laminar, 2000-4000 as transitional, and Re > 4000 as turbulent. Most PHES waterways will be strongly turbulent.")
        with section_panel("check", "Diameter sensitivity checks"):
            q_for_diameter = design_case["flow_for_velocity_m3_s"]
            gross_head = float(levels.get("gross_head", float("nan")))
            hf_2_percent = 0.02 * gross_head
            hf_5_percent = 0.05 * gross_head
            hf_10_percent = 0.10 * gross_head
            with st.expander("Open target-velocity and head-loss guidance"):
                st.markdown("**Target velocity**")
                st.dataframe(
                    pd.DataFrame(VELOCITY_GUIDANCE, columns=["Waterway part", "Reference velocity range", "Design note"]),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "For the Step 5 shared pressure waterway, start around 3.5-5.5 m/s unless the layout is clearly a low-pressure headrace/tailrace. "
                    "Lower velocity gives a larger, more expensive conduit but lower losses and smaller water-hammer sensitivity; higher velocity gives a smaller conduit but higher losses and transient risk."
                )
                st.markdown("**Allowable head loss**")
                st.dataframe(
                    pd.DataFrame(HEAD_LOSS_ALLOWANCE_GUIDANCE, columns=["Screening class", "Suggested allowance", "How to use it"]),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "A practical classroom method is to set an allowable head-loss budget as a percentage of Step 3 gross head, "
                    "then size the diameter and go back to Step 4 if the net-head change is material."
                )
            suggestion_cols = st.columns(4)
            with suggestion_cols[0]:
                render_symbol_value_card("Pressure-waterway velocity start", "3.5-5.5 m/s")
            with suggestion_cols[1]:
                render_symbol_value_card("Strict loss budget, 2% H<sub>g</sub>", metric_value(hf_2_percent, " m", 1))
            with suggestion_cols[2]:
                render_symbol_value_card("Typical loss budget, 5% H<sub>g</sub>", metric_value(hf_5_percent, " m", 1))
            with suggestion_cols[3]:
                render_symbol_value_card("Upper screen, 10% H<sub>g</sub>", metric_value(hf_10_percent, " m", 1))
            d_velocity = diameter_from_velocity(q_for_diameter, st.session_state.target_velocity)
            c1, c2 = st.columns(2)
            with c1:
                st.slider(
                    "Target velocity (m/s)",
                    1.0,
                    10.0,
                    step=0.1,
                    key="target_velocity",
                    help="Use the waterway-type table above. For a shared pressure waterway, 3.5-5.5 m/s is a good first pass; justify values outside that range with loss, transient, and cost checks.",
                )
                st.metric("Diameter from target velocity", metric_value(d_velocity, " m", 2))
                st.caption(f"Using {st.session_state.flow_area_mode.lower()} flow: {metric_value(q_for_diameter, ' m³/s', 1)}.")
            with c2:
                target_hf = st.number_input(
                    "Allowable head loss (m)",
                    min_value=0.5,
                    max_value=300.0,
                    value=15.0,
                    step=0.5,
                    key="step7_allowable_headloss_m",
                    help="Choose as a percentage of gross head: about 2% for a strict efficient waterway, 5% for a typical first screen, and 10% as an upper concept screen.",
                )
                d_hf, f_hf, re_hf, v_hf, hf_hf = diameter_from_headloss(
                    q_for_diameter,
                    fnum('penstock_length_m'),
                    target_hf,
                    st.session_state.roughness_m,
                    st.session_state.temperature_c,
                    k_sum,
                )
                st.metric("Diameter from head-loss target", metric_value(d_hf, " m", 2))
                st.caption(f"f={metric_value(f_hf, '', 4)}, Re={metric_value(re_hf, '', 0)}, v={metric_value(v_hf, ' m/s', 2)}, hf={metric_value(hf_hf, ' m', 2)}.")
            velocity_status_label, velocity_interpretation = velocity_status(st.session_state.target_velocity)
            hf_percent = safe_div(target_hf, gross_head) * 100.0
            if hf_percent <= 3.0:
                hf_interpretation = "Strict/efficient loss budget; diameter may become large but energy loss is low."
            elif hf_percent <= 5.0:
                hf_interpretation = "Typical first-screening loss budget."
            elif hf_percent <= 10.0:
                hf_interpretation = "Broad concept upper screen; verify Step 4 discharge and Step 9 turbine selection."
            else:
                hf_interpretation = "High loss budget; revise diameter or explain why this loss is acceptable."
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Target velocity", st.session_state.target_velocity, "m/s", velocity_interpretation],
                        ["Allowable head loss", target_hf, "m", f"{metric_value(hf_percent, '% of gross head', 1)}. {hf_interpretation}"],
                        ["Step 5 calculated loss", design_case["total_loss_m"], "m", f"{metric_value(safe_div(design_case['total_loss_m'], gross_head) * 100.0, '% of gross head', 1)} from the current diameter and local losses."],
                    ],
                    columns=["Input/check", "Value", "Unit", "Interpretation"],
                ),
                hide_index=True,
                width="stretch",
                column_config={"Value": st.column_config.NumberColumn(format="%.3g")},
            )
            if hf_percent > 10.0:
                st.warning("The allowable head-loss target is above 10% of gross head. This should trigger a Step 4 head/discharge re-check and a turbine-selection sensitivity check.")
            elif st.session_state.target_velocity < 3.5 or st.session_state.target_velocity > 5.5:
                getattr(st, velocity_status_label)(velocity_interpretation)
    elif step == 6:
        st.caption("Select unit branch velocity by comparing diameter, loss, transient risk, and constructability.")
        u1, u2, u3 = st.columns(3)
        with u1:
            st.number_input("Units", min_value=1, max_value=20, step=1, key="units")
        with u2:
            st.number_input("Penstocks", min_value=1, max_value=20, step=1, key="penstocks")
        with u3:
            st.number_input("Selected unit branch diameter (m)", min_value=0.2, max_value=30.0, step=0.1, key="unit_penstock_diameter_m")
        design_case = hydraulic_snapshot(fnum('design_power_mw'), levels["gross_head"])
        q_unit = discharge_per_unit(design_case["total_discharge_m3_s"])
        if not np.isfinite(q_unit):
            st.info("Unit branch sizing appears once the total design discharge exists: complete Step 1 (power), Step 3 (levels), Step 4 (effective head), and Step 5 (shared conduit) first.")
            render_step_guidance(step)
            st.stop()
        pressure_length = fnum('penstock_length_m')
        suggested_branch_length = max(100.0, min(500.0, 0.10 * pressure_length)) if np.isfinite(pressure_length) and pressure_length > 0 else 500.0
        branch_length_range = (
            f"{metric_value(max(50.0, 0.05 * pressure_length), ' m', 0)} to {metric_value(max(100.0, 0.20 * pressure_length), ' m', 0)}"
            if np.isfinite(pressure_length) and pressure_length > 0
            else "100 m to 500 m"
        )
        material_label = str(st.session_state.get("roughness_material", "Unknown concept"))
        material_key = material_label.lower()
        if "steel" in material_key:
            suggested_wave_speed = 1050.0
        elif "concrete" in material_key:
            suggested_wave_speed = 950.0
        elif "rock" in material_key:
            suggested_wave_speed = 1100.0
        elif any(token in material_key for token in ("pvc", "hdpe", "pe")):
            suggested_wave_speed = 500.0
        elif "ductile" in material_key or "iron" in material_key:
            suggested_wave_speed = 1000.0
        else:
            suggested_wave_speed = 1000.0
        with st.expander("How to choose units, branches, intakes, and wave speed"):
            st.markdown("**How to choose the three unclear Step 6 numeric inputs**")
            st.dataframe(
                pd.DataFrame(STEP6_NUMERIC_INPUT_GUIDANCE, columns=["Input", "How to choose it", "How to report it"]),
                hide_index=True,
                width="stretch",
            )
            st.markdown("**Unit branch local-loss coefficient ranges**")
            st.dataframe(
                pd.DataFrame(UNIT_BRANCH_K_GUIDANCE, columns=["Branch layout", "Typical K range", "Use when"]),
                hide_index=True,
                width="stretch",
            )
            st.markdown("**General branch and opening choices**")
            st.dataframe(
                pd.DataFrame(BRANCH_SELECTION_GUIDANCE, columns=["Input", "Starting choice", "Design note"]),
                hide_index=True,
                width="stretch",
            )
            st.markdown("**Wave-speed ranges for the water-hammer sense check**")
            st.dataframe(
                pd.DataFrame(WAVE_SPEED_GUIDANCE, columns=["Material / waterway", "Teaching wave-speed range", "Why it matters"]),
                hide_index=True,
                width="stretch",
            )
            st.caption("For a simple teaching layout, use one branch and one intake/outlet opening per turbine unit unless the Step 4 layout clearly shows a shared intake or manifold.")
        st.markdown("**Suggested starting values from the current layout**")
        s1, s2, s3 = st.columns(3)
        with s1:
            render_symbol_value_card("Unit branch length", metric_value(suggested_branch_length, " m", 0))
        with s2:
            render_symbol_value_card("Unit branch local K", "2.0")
        with s3:
            render_symbol_value_card("Wave speed <span>a</span>", metric_value(suggested_wave_speed, " m/s", 0))
        st.caption(
            f"Branch length suggestion uses about 10% of the Step 5 conduit length; a broad concept range is {branch_length_range}. "
            f"Wave speed suggestion is based on the current roughness/material selection: {material_label}."
        )
        i1, i2 = st.columns(2)
        with i1:
            intake_count = st.number_input("Intake/outlet openings", min_value=1, max_value=50, value=max(int(st.session_state.penstocks), 1), step=1)
        with i2:
            st.metric("Unit discharge", metric_value(q_unit, " m³/s", 1))
        b1, b2, b3 = st.columns(3)
        with b1:
            branch_length = st.number_input(
                "Unit branch length (m)",
                min_value=1.0,
                value=float(suggested_branch_length),
                step=25.0,
                help="Measure the centreline length from the branch/manifold split to the turbine inlet. If no branch is drawn, use the suggested value and show a sensitivity range.",
            )
        with b2:
            unit_k = st.number_input(
                "Unit branch local loss K",
                min_value=0.0,
                value=2.0,
                step=0.1,
                help="Sum only the branch, valve, bend and transition losses that are not already counted in Step 5. Smooth concept branch: about 1-3; complex branch: 3-6 or higher.",
            )
        with b3:
            wave_speed = st.number_input(
                "Wave speed for water-hammer sense check (m/s)",
                min_value=100.0,
                value=float(suggested_wave_speed),
                step=50.0,
                help="Choose from the material/lining table. This only screens instantaneous water-hammer rise; Step 8 revisits closure time and surge control.",
            )
        rows = []
        nu = water_nu_kinematic_m2_s(st.session_state.temperature_c)
        for velocity in [3.0, 4.0, 5.0, 6.0, 7.0]:
            diameter = diameter_from_velocity(q_unit, velocity)
            reynolds = safe_div(velocity * diameter, nu)
            friction = f_swamee_jain(reynolds, safe_div(st.session_state.roughness_m, diameter))
            loss = head_loss(friction, branch_length, diameter, velocity, unit_k)
            hammer = safe_div(wave_speed * velocity, G)
            rows.append([velocity, diameter, loss, hammer])
        st.dataframe(
            pd.DataFrame(rows, columns=["Velocity (m/s)", "Unit diameter (m)", "Unit loss (m)", "Water-hammer ΔH (m)"]),
            hide_index=True,
            width="stretch",
            column_config={col: st.column_config.NumberColumn(format="%.2f") for col in ["Velocity (m/s)", "Unit diameter (m)", "Unit loss (m)", "Water-hammer ΔH (m)"]},
        )
        empirical_d = 0.802 * max(q_unit, 0.0) ** 0.437 if np.isfinite(q_unit) else float("nan")
        selected_velocity = safe_div(q_unit, area_circle(fnum('unit_penstock_diameter_m')))
        selected_re = safe_div(selected_velocity * fnum('unit_penstock_diameter_m'), nu)
        selected_f = f_swamee_jain(selected_re, safe_div(st.session_state.roughness_m, fnum('unit_penstock_diameter_m')))
        selected_loss = head_loss(selected_f, branch_length, fnum('unit_penstock_diameter_m'), selected_velocity, unit_k)
        loss_allow_low = 0.05 * levels["gross_head"]
        loss_allow_high = 0.10 * levels["gross_head"]
        deviation = safe_div(fnum('unit_penstock_diameter_m') - empirical_d, empirical_d) * 100.0
        verify_rows = pd.DataFrame(
            [
                ["Selected unit diameter", fnum('unit_penstock_diameter_m'), "m", "Current Step 6 value"],
                ["Selected unit velocity", selected_velocity, "m/s", "Check against about 4-6 m/s, or justify 3.5-7 m/s"],
                ["Intake/outlet openings", intake_count, "-", "Concept count; usually one per penstock/unit or a shared intake with separate gated passages"],
                ["Empirical diameter", empirical_d, "m", "D = 0.802 Q_u^0.437, used as guidance only"],
                ["Deviation from empirical", deviation, "%", "Deviation is acceptable if velocity, losses and transients are justified"],
                ["Selected branch loss", selected_loss, "m", "Candidate unit branch loss"],
                ["Loss budget lower", loss_allow_low, "m", "5% of gross head"],
                ["Loss budget upper", loss_allow_high, "m", "10% of gross head"],
            ],
            columns=["Check", "Value", "Unit", "Interpretation"],
        )
        st.dataframe(verify_rows, hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.3g")})
    elif step == 7:
        page_underground_civil(levels, design_case, embedded=True)
    elif step == 8:
        page_surge_transient(design_case, embedded=True)
    elif step == 9:
        page_turbine_power(levels, design_case, embedded=True)
    elif step == 10:
        page_report_risks(levels, design_case, embedded=True)

    render_step_guidance(step)


def page_design_discharge(levels: dict[str, float], embedded: bool = False) -> None:
    if not embedded:
        st.title("Design Discharge")
    st.markdown(
        r"Start from the Step 3 gross head $H_g$. Use $H_g$ for the ideal first-pass discharge, or choose a lower "
        r"first-pass effective head $H_e$ only when you intentionally allow for hydraulic losses, operating-level effects, or a known benchmark net-head basis. "
        r"After Step 5 calculates civil net head, return here and check the discharge again."
    )

    gross_head = float(levels.get("gross_head", float("nan")))
    if st.session_state.get("step4_head_basis") in STEP4_HEAD_BASIS_LEGACY:
        st.session_state["step4_head_basis"] = STEP4_HEAD_BASIS_LEGACY[st.session_state["step4_head_basis"]]
    if st.session_state.get("step4_head_basis") not in STEP4_HEAD_BASIS_OPTIONS:
        active_preset = st.session_state.get("active_preset", NEW_PROJECT_PRESET)
        st.session_state["step4_head_basis"] = (
            STEP4_HEAD_BASIS_MANUAL
            if active_preset != NEW_PROJECT_PRESET and has_values("teaching_effective_head_m")
            else STEP4_HEAD_BASIS_GROSS
        )

    c0, c1, c2, c3 = st.columns([1.45, 1.0, 1.0, 1.0])
    with c0:
        head_basis = st.selectbox(
            "Head basis for discharge",
            STEP4_HEAD_BASIS_OPTIONS,
            key="step4_head_basis",
            help="For a new concept, start with the Step 3 gross head. Use manual effective head only when you have a known benchmark or a deliberate first-pass net-head allowance.",
        )
    with c1:
        render_symbol_value_card("Gross head <span>H<sub>g</sub></span>", metric_value(gross_head, " m", 1))
    with c2:
        st.number_input("Required operation time (hours)", min_value=0.1, step=0.5, key="operation_hours")
    with c3:
        st.slider("First-pass sizing efficiency η", 0.45, 0.96, step=0.01, key="sizing_efficiency")

    if head_basis == STEP4_HEAD_BASIS_GROSS:
        selected_head = gross_head
        if np.isfinite(selected_head) and selected_head > 0:
            st.session_state["teaching_effective_head_m"] = selected_head
        st.info(r"Using $H_g$ directly for the first-pass discharge. This is the cleanest early concept check before Step 5 losses are known.")
    elif head_basis == STEP4_HEAD_BASIS_REDUCED:
        default_allowance = 0.0
        if np.isfinite(gross_head) and gross_head > 0:
            default_allowance = 0.05 * gross_head
        allowance_max = max(gross_head - 1.0, 1.0) if np.isfinite(gross_head) else 1000.0
        st.markdown(
            '<div class="equation-label">First-pass head allowance below <span>H<sub>g</sub></span> (m)</div>',
            unsafe_allow_html=True,
        )
        allowance = st.number_input(
            "First-pass head allowance below gross head",
            min_value=0.0,
            max_value=float(allowance_max),
            value=max(0.0, min(num_or(st.session_state.get("step4_head_allowance_m"), default_allowance), float(allowance_max))),
            step=1.0,
            key="step4_head_allowance_m",
            help="Use this for an early hydraulic-loss or operating-head allowance before Step 5 calculates the explicit loss components.",
            label_visibility="collapsed",
        )
        selected_head = max(gross_head - allowance, 1.0) if np.isfinite(gross_head) else float("nan")
        if np.isfinite(selected_head) and selected_head > 0:
            st.session_state["teaching_effective_head_m"] = selected_head
        st.warning(r"You are using a reduced first-pass $H_e$. Record the allowance basis and verify it against Step 5 civil net head.")
    else:
        st.markdown(
            '<div class="equation-label">Preset/manual effective head <span>H<sub>e</sub></span> (m)</div>',
            unsafe_allow_html=True,
        )
        st.number_input(
            "Preset/manual effective head",
            min_value=1.0,
            max_value=2000.0,
            step=1.0,
            key="teaching_effective_head_m",
            help="Use for benchmark cases or a deliberate net/effective head from prior calculations. Explain why it differs from gross head.",
            label_visibility="collapsed",
        )
        selected_head = float(fnum('teaching_effective_head_m'))

    h_cols = st.columns(4)
    with h_cols[0]:
        render_symbol_value_card("Selected first-pass <span>H</span>", metric_value(selected_head, " m", 1))
    with h_cols[1]:
        render_symbol_value_card("Head reduction from <span>H<sub>g</sub></span>", metric_value(gross_head - selected_head, " m", 1))
    with h_cols[2]:
        render_symbol_value_card("<span>H</span> / <span>H<sub>g</sub></span>", metric_value(safe_div(selected_head, gross_head), "", 3))
    with h_cols[3]:
        render_symbol_value_card("Active storage from Step 3", metric_value(fnum('reservoir_volume_m3') / 1e6, " GL", 1))
    st.caption(
        "The design-discharge equation is listed in the bottom Step 4 Equations, annotations, and reference table. "
        "Use gross head for an ideal first pass, or effective head when a preliminary allowance is deliberately applied."
    )
    with st.expander("How to choose the first-pass efficiency"):
        st.dataframe(
            pd.DataFrame(EFFICIENCY_GUIDANCE, columns=["η range", "Use case", "Teaching guidance"]),
            hide_index=True,
            width="stretch",
        )
        st.info("After Step 5 calculates civil net head and Step 9 estimates equipment efficiency, return here and check whether Q still supports the target MW and storage duration.")

    d1, d2 = st.columns([1.2, 1.0])
    with d1:
        discharge_basis = st.selectbox(
            "Operating-discharge basis",
            DISCHARGE_BASIS_OPTIONS,
            key="discharge_basis",
            help="Use the power-target equation for a new design. Use a declared Q only for a benchmark, measured operating point, or deliberately selected design flow.",
        )
    with d2:
        target_q = q_from_power(fnum('design_power_mw'), selected_head, st.session_state.sizing_efficiency)
        if discharge_basis == DISCHARGE_BASIS_DECLARED:
            if not np.isfinite(fnum('operating_discharge_m3_s')) or fnum('operating_discharge_m3_s') <= 0:
                st.session_state["operating_discharge_m3_s"] = max(num_or(target_q, 1.0), 0.1)
            st.number_input(
                "Declared operating Q (m³/s)",
                min_value=0.1,
                step=1.0,
                key="operating_discharge_m3_s",
                help="Record the source and operating condition. Step 9 will show the power actually achieved at the net head.",
            )
        else:
            st.metric("Power-target sizing Q", metric_value(target_q, " m³/s", 1))

    if not (step1_complete() and head_defined()):
        st.info("Enter the design power, operation duration and head basis to calculate the design discharge.")
        return
    head = float(fnum('teaching_effective_head_m'))
    q_gross = q_from_power(fnum('design_power_mw'), gross_head, st.session_state.sizing_efficiency)
    q_target = q_from_power(fnum('design_power_mw'), head, st.session_state.sizing_efficiency)
    q_req = selected_design_discharge(head)
    q_report = q_req
    storage_required = q_req * fnum('operation_hours') * 3600.0
    operation_with_storage = safe_div(fnum('reservoir_volume_m3'), q_req * 3600.0)
    q_per_penstock = safe_div(q_req, st.session_state.penstocks)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        render_symbol_value_card("<span>Q</span> using <span>H<sub>g</sub></span>", metric_value(q_gross, " m³/s", 1))
    with m2:
        render_symbol_value_card("Selected design <span>Q</span>", metric_value(q_req, " m³/s", 1))
    with m3:
        render_symbol_value_card("Selected vs target-derived <span>Q</span>", metric_value(safe_div(q_req - q_target, q_target) * 100.0, "%", 1))
    with m4:
        render_symbol_value_card("Per penstock", metric_value(q_per_penstock, " m³/s", 1))
    with m5:
        render_symbol_value_card("Required storage", metric_value(storage_required / 1e6, " ML", 1))
    with m6:
        render_symbol_value_card("Provided duration", metric_value(operation_with_storage, " h", 1))

    snowy_reference_q = q_from_power(2000.0, head, st.session_state.sizing_efficiency)
    st.caption(
        "The plot uses total plant discharge. For comparison, at the currently selected head and 2000 MW, "
        f"Q is about {metric_value(snowy_reference_q, ' m³/s', 0)} total, "
        f"or {metric_value(safe_div(snowy_reference_q, 6), ' m³/s', 0)} per penstock for six units. "
        "If a lecture or benchmark report uses a different value, state whether it used gross head, net head, overload capacity, or a different efficiency."
    )

    if fnum('reservoir_volume_m3') >= storage_required:
        st.success("Provided active storage satisfies the selected operating duration.")
    else:
        st.warning("Provided active storage is smaller than the selected operating duration requires.")

    with st.expander("Annual energy and capacity-factor discharge cross-check"):
        c1, c2 = st.columns(2)
        with c1:
            annual_energy_gwh = st.number_input(
                "Annual generation target (GWh/year)",
                min_value=0.0,
                value=max(num_or(fnum('design_power_mw') * fnum('operation_hours') / 1000.0, 0.0), 0.0),
                step=100.0,
            )
        with c2:
            capacity_factor = st.number_input("Capacity factor CF", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        seconds_per_year = 365.0 * 24.0 * 3600.0
        q_ave = safe_div(annual_energy_gwh * 3.6e12, RHO * G * max(head, 0.001) * st.session_state.sizing_efficiency * seconds_per_year)
        q_max_cf = safe_div(q_ave, capacity_factor)
        cf_rows = pd.DataFrame(
            [
                ["Average discharge", q_ave, "m³/s", "Q_ave from annual energy spread over one year"],
                ["Maximum discharge from CF", q_max_cf, "m³/s", "Q_max = Q_ave / CF"],
                ["Power-based design discharge", q_req, "m³/s", "Q from design power, head and efficiency"],
                ["Difference from power method", safe_div(q_max_cf - q_req, q_req) * 100.0, "%", "Large differences mean the annual-energy target or CF assumption needs review"],
            ],
            columns=["Check", "Value", "Unit", "Interpretation"],
        )
        st.dataframe(cf_rows, hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.3g")})

    st.subheader("Reservoir levels and rating head")
    fig_levels = go.Figure()
    x_positions = ["Upper reservoir", "Lower reservoir"]
    fig_levels.add_trace(go.Scatter(x=x_positions, y=[levels["upper_hwl"], levels["lower_hwl"]], mode="markers+text", text=["HWL", "HWL"], textposition="top center", name="High water level", marker=dict(size=12, color=MONASH_BLUE)))
    fig_levels.add_trace(go.Scatter(x=x_positions, y=[levels["upper_lwl"], levels["lower_twl"]], mode="markers+text", text=["LWL", "TWL"], textposition="bottom center", name="Low/tailwater level", marker=dict(size=12, color=MONASH_BLUEBERRY)))
    fig_levels.add_trace(go.Scatter(x=["Upper reservoir"], y=[levels["upper_representative"]], mode="markers+text", text=["Representative"], textposition="middle right", name="Representative upper level", marker=dict(size=14, color=MONASH_ELECTRIC_BLUE)))
    fig_levels.add_shape(type="line", x0="Upper reservoir", x1="Lower reservoir", y0=levels["upper_representative"], y1=levels["lower_representative"], line=dict(color=MONASH_GREY_1, dash="dash"))
    fig_levels.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Elevation (m)")
    st.plotly_chart(fig_levels, width="stretch")

    st.subheader("Discharge, storage and power relationship")
    q_min = max(1.0, q_req * 0.25 if np.isfinite(q_req) else 50.0)
    q_max = max(q_min + 1.0, max(q_req * 1.35, q_report * 1.15) if np.isfinite(q_req) else 500.0)
    q_range = np.linspace(q_min, q_max, 80)
    p_curve = power_from_q_head(q_range, max(head, 0.001), st.session_state.sizing_efficiency)
    t_curve = fnum('reservoir_volume_m3') / (q_range * 3600.0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=q_range, y=p_curve, name="Power", line=dict(color=MONASH_BLUE, width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=q_range, y=t_curve, name="Duration from storage", line=dict(color=MONASH_ELECTRIC_BLUE, width=3)), secondary_y=True)
    fig.add_vline(x=q_req, line_dash="dash", line_color=MONASH_BLUEBERRY)
    if np.isfinite(q_report):
        fig.add_vline(x=q_report, line_dash="dot", line_color=MONASH_GREY_1, annotation_text="Q_report")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    fig.update_xaxes(title="Discharge Q (m³/s)")
    fig.update_yaxes(title="Power (MW)", secondary_y=False)
    fig.update_yaxes(title="Operating duration (h)", secondary_y=True)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Dam type suggestion")
    crest_level = (
        levels["upper_hwl"]
        + st.session_state.freeboard_m
        + st.session_state.wave_allowance_m
        + st.session_state.settlement_allowance_m
    )
    dam_height = max(crest_level - fnum('dam_foundation_rl'), 0.0)
    suggestion = recommend_dam_type(
        st.session_state.reservoir_arrangement,
        st.session_state.valley_geometry,
        st.session_state.foundation_quality,
        st.session_state.construction_material,
        dam_height,
        fnum('reservoir_volume_m3'),
    )
    st.metric("Suggested preliminary dam type", suggestion)
    with st.expander("Selection heuristics"):
        st.markdown(
            """
            - Existing reservoir reuse: identify operating constraints and modification works.
            - Earthfill or rockfill embankment: broad valleys and locally available fill materials.
            - CFRD: rockfill option where seepage control is important.
            - RCC/concrete gravity: narrower valley with competent rock foundation.
            - Arch dam: narrow gorge with very strong abutments.
            - Ring-dike/turkey-nest: off-river plateau or ridge-top PHES storage.
            """
        )
        st.caption("These are classroom screening rules only. Final dam type depends on geology, hydrology, foundation conditions, materials and construction method.")

def page_turbine_power(levels: dict[str, float], design_case: dict[str, float], embedded: bool = False) -> None:
    if not embedded:
        st.title("Turbine And Power")
    st.caption("Step 9: select the pump-turbine operating envelope after head, discharge, losses, and unit flows are known.")
    if not discharge_ready():
        st.info("Turbine and pumping results appear once the design power and duration (Step 1) and the effective head (Step 4) are set.")
        return

    st.subheader("Efficiency assumptions")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.slider("Teaching turbine efficiency", 0.70, 0.98, step=0.005, key="eta_turbine")
    with e2:
        st.slider("Generator efficiency", 0.90, 0.99, step=0.01, key="eta_generator")
    with e3:
        st.slider("Transformer efficiency", 0.98, 0.995, step=0.001, key="eta_transformer")
    with e4:
        st.slider("Pump efficiency", 0.70, 0.92, step=0.01, key="eta_pump")
    st.caption("The Step 4 sizing efficiency is a separate first-pass assumption. Step 9 power uses the turbine, generator and transformer efficiencies entered here; the curve value is shown only as a screening cross-check.")
    with st.expander("Why there are multiple head and discharge inputs"):
        st.dataframe(
            pd.DataFrame(HEAD_BASIS_GUIDANCE, columns=["Option", "Meaning", "When to use"]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "For the final design trail, use civil net head and civil design Q after Step 5 unless you are deliberately running a sensitivity case. "
            "Near turbine chart boundaries, compare both plant-total Q and unit Q."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        head_basis = st.radio("Head input", ["Use Step 4 selected head", "Use civil net head", "Manual head"], index=1, horizontal=True)
        if head_basis == "Manual head":
            head = st.number_input("Manual head (m)", min_value=1.0, max_value=2000.0, value=max(num_or(design_case["net_head_m"], 100.0), 1.0), step=1.0)
        elif head_basis == "Use Step 4 selected head":
            head = float(fnum('teaching_effective_head_m'))
            st.metric("Head used", metric_value(head, " m", 1))
        else:
            head = design_case["net_head_m"]
            st.metric("Head used", metric_value(head, " m", 1))
    with c2:
        q_basis = st.radio("Discharge input", ["Use Step 4 selected Q", "Use power-target Q", "Manual Q"], horizontal=True)
        if q_basis == "Manual Q":
            discharge = st.number_input("Manual discharge (m³/s)", min_value=0.1, max_value=3000.0, value=max(num_or(design_case["total_discharge_m3_s"], 100.0), 0.1), step=1.0)
        elif q_basis == "Use Step 4 selected Q":
            discharge = selected_design_discharge(fnum('teaching_effective_head_m'))
            st.metric("Discharge used", metric_value(discharge, " m³/s", 1))
        else:
            discharge = q_from_power(fnum('design_power_mw'), float(fnum('teaching_effective_head_m')), st.session_state.sizing_efficiency)
            st.metric("Discharge used", metric_value(discharge, " m³/s", 1))
    with c3:
        st.number_input("Runner speed N (rpm)", min_value=50, max_value=1500, step=25, key="runner_speed_rpm")
        st.slider("Q / Qmax factor", 1.0, 3.0, step=0.05, key="qmax_factor")
        st.caption("For Pelton/Kaplan/Bulb teaching curves, the slider sets the plotted operating ratio Q/Qmax.")

    recommended = select_turbine(head, discharge)
    eta_turbine_curve = turbine_efficiency(recommended, head, discharge)
    eta_total = st.session_state.eta_turbine * st.session_state.eta_generator * st.session_state.eta_transformer
    p_hydraulic = power_from_q_head(discharge, head, 1.0)
    p_generation = power_from_q_head(discharge, head, eta_total)
    p_pump = (RHO * G * discharge * head) / (st.session_state.eta_pump * 1e6) if st.session_state.eta_pump > 0 else float("nan")
    energy_mwh = p_generation * fnum('operation_hours')
    q_unit = discharge_per_unit(discharge)
    p_unit = safe_div(fnum('design_power_mw'), st.session_state.units)
    n_q = st.session_state.runner_speed_rpm * math.sqrt(max(q_unit, 0.001)) / max(head, 0.001) ** 0.75
    unit_family_check = select_turbine(head, q_unit)
    zone_kind, zone_text = turbine_application_message(head, discharge, q_unit)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Recommended turbine", recommended)
    m2.metric("Hydraulic power", metric_value(p_hydraulic, " MW", 1))
    m3.metric("Assumed turbine η", metric_value(st.session_state.eta_turbine * 100, "%", 1))
    m4.metric("Curve η check", metric_value(eta_turbine_curve * 100, "%", 1))
    m5.metric("Generation", metric_value(p_generation, " MW", 1))
    m6.metric("Pumping power", metric_value(p_pump, " MW", 1))

    st.caption(f"Energy for selected operation duration: {metric_value(energy_mwh, ' MWh', 1)}")
    st.caption("The pumping-power value is a same-head screening comparison. Final pump-mode input must use the pumping operating head, including the appropriate static level combination and pumping-direction losses.")
    power_gap_percent = safe_div(p_generation - fnum('design_power_mw'), fnum('design_power_mw')) * 100.0
    if np.isfinite(power_gap_percent) and abs(power_gap_percent) > 2.0:
        st.warning(f"The selected Q and net head produce {metric_value(p_generation, ' MW', 1)}, which is {metric_value(power_gap_percent, '%', 1)} relative to the design-power target. Revise Q, head, efficiency or the target; do not silently force the calculation to close.")
    getattr(st, zone_kind)(zone_text)
    st.subheader("Plant and unit-level verification")
    st.dataframe(
        pd.DataFrame(
            [
                ["Plant-level chart point", discharge, fnum('design_power_mw'), recommended, "Use total Q and head for first turbine-family screening"],
                ["Unit-level check", q_unit, p_unit, unit_family_check, "Use Q_u and P_u for manufacturability and specific speed"],
                ["Specific speed n_q", n_q, np.nan, "SI check", "Calculate at unit level; plant-level n_q is not representative"],
            ],
            columns=["Check", "Q (m³/s)", "Power (MW)", "Result", "Interpretation"],
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Q (m³/s)": st.column_config.NumberColumn(format="%.2f"),
            "Power (MW)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption("If the plant-level point sits near a chart boundary, confirm the selection using per-unit discharge, per-unit power, runner speed and specific-speed guidance.")

    c1, c2 = st.columns([1.05, 1.0])
    with c1:
        st.subheader("Turbine application zones")
        render_matplotlib_figure(turbine_application_figure(discharge, head))

    with c2:
        st.subheader("Efficiency curve")
        efficiency_curve_plot(recommended, head, discharge, eta_turbine_curve)

    with st.expander("Reference images from previous teaching app"):
        image_names = [
            "Turbine Selection.png",
            "Specific speed.png",
            "Efficiency as per specific speed.png",
            "Turbine efficiency as per flow.png",
        ]
        for image_name in image_names:
            image_path = APP_DIR / image_name
            if image_path.exists():
                st.image(str(image_path), caption=image_name, width="stretch")

def turbine_efficiency(turbine: str, head_m: float, discharge_m3_s: float) -> float:
    if turbine == "Francis":
        ns_vals = np.array([5, 10, 20, 40, 60, 80, 100], dtype=float)
        eta_vals = np.array([0.82, 0.88, 0.92, 0.93, 0.91, 0.88, 0.85], dtype=float)
        q_unit = discharge_per_unit(discharge_m3_s)
        n_q = st.session_state.runner_speed_rpm * math.sqrt(max(q_unit, 0.001)) / max(head_m, 0.001) ** 0.75
        return float(np.interp(n_q, ns_vals, eta_vals))

    q_ratio_vals = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    if turbine == "Pelton":
        eta_vals = np.array([0.70, 0.85, 0.90, 0.93, 0.94, 0.94], dtype=float)
    else:
        eta_vals = np.array([0.60, 0.80, 0.90, 0.93, 0.94, 0.94], dtype=float)
    q_max = discharge_m3_s * st.session_state.qmax_factor
    q_ratio = safe_div(discharge_m3_s, q_max)
    return float(np.interp(q_ratio, q_ratio_vals, eta_vals))


def efficiency_curve_plot(turbine: str, head_m: float, discharge_m3_s: float, eta: float) -> None:
    if turbine == "Francis":
        ns_vals = np.array([5, 10, 20, 40, 60, 80, 100], dtype=float)
        eta_vals = np.array([0.82, 0.88, 0.92, 0.93, 0.91, 0.88, 0.85], dtype=float)
        q_unit = discharge_per_unit(discharge_m3_s)
        n_q = st.session_state.runner_speed_rpm * math.sqrt(max(q_unit, 0.001)) / max(head_m, 0.001) ** 0.75
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ns_vals, y=eta_vals * 100, mode="lines+markers", line=dict(color=MONASH_BLUE), marker=dict(color=MONASH_BLUE), name="Francis n_q η"))
        fig.add_trace(go.Scatter(x=[n_q], y=[eta * 100], mode="markers", marker=dict(size=12, color=MONASH_ELECTRIC_BLUE), name="Current design"))
        fig.update_layout(template="plotly_white", height=390, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Specific speed n_q", yaxis_title="Turbine efficiency (%)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Specific speed n_q = {metric_value(n_q, '', 1)} using Q_u = {metric_value(q_unit, ' m³/s', 1)}")
        return

    q_ratio_vals = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    if turbine == "Pelton":
        eta_vals = np.array([0.70, 0.85, 0.90, 0.93, 0.94, 0.94], dtype=float)
    else:
        eta_vals = np.array([0.60, 0.80, 0.90, 0.93, 0.94, 0.94], dtype=float)
    q_ratio = 1.0 / st.session_state.qmax_factor
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_ratio_vals, y=eta_vals * 100, mode="lines+markers", line=dict(color=MONASH_BLUE), marker=dict(color=MONASH_BLUE), name=f"{turbine} η"))
    fig.add_trace(go.Scatter(x=[q_ratio], y=[eta * 100], mode="markers", marker=dict(size=12, color=MONASH_ELECTRIC_BLUE), name="Current design"))
    fig.update_layout(template="plotly_white", height=390, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Q/Qmax", yaxis_title="Turbine efficiency (%)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Q/Qmax = {metric_value(q_ratio, '', 2)} from the operating-ratio slider; it is a teaching operating point, not a recalculation of total discharge.")


def page_underground_civil(levels: dict[str, float], design_case: dict[str, float], embedded: bool = False) -> None:
    if not embedded:
        st.title("Underground Civil Structures")
    st.caption("Step 7: check pressure-tunnel confinement, lining stress, powerhouse cavern dimensions, transformer hall, IPB gallery, access, drainage, ventilation, and egress.")
    if not (head_defined() and conduit_defined()):
        st.info("Underground checks appear once the effective head (Step 4) and the conduit length and diameter (Step 5) are set: they provide the hydrostatic head and lining radius defaults.")
        return

    tab_conf, tab_lining, tab_cavern = st.tabs(["Confinement", "Lining stress", "Powerhouse cavern"])

    with tab_conf:
        st.subheader("Norwegian confinement criteria")
        c1, c2, c3 = st.columns(3)
        with c1:
            hs = st.number_input("Hydrostatic head hₛ (m)", min_value=0.0, value=num_or(fnum('teaching_effective_head_m'), 100.0), step=10.0)
            alpha = st.number_input("Tunnel inclination α (deg)", min_value=0.0, max_value=90.0, value=25.0, step=1.0)
        with c2:
            beta = st.number_input("Slope angle β (deg)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
            gamma_r = st.number_input("Rock unit weight γᵣ (kN/m³)", min_value=0.1, value=26.0, step=0.5)
        with c3:
            c_rv = st.number_input("Given vertical cover C_RV (m)", min_value=0.0, value=800.0, step=5.0)
            c_rm = st.number_input("Given minimum cover C_RM (m)", min_value=0.0, value=700.0, step=5.0)
            f_req = st.number_input("Required factor of safety", min_value=0.1, value=1.5, step=0.1)

        gamma_w = 9.81
        f_rv = safe_div(c_rv * gamma_r * math.cos(math.radians(alpha)), hs * gamma_w)
        f_rm = safe_div(c_rm * gamma_r * math.cos(math.radians(beta)), hs * gamma_w)
        c_rv_req = safe_div(f_req * hs * gamma_w, gamma_r * math.cos(math.radians(alpha)))
        c_rm_req = safe_div(f_req * hs * gamma_w, gamma_r * math.cos(math.radians(beta)))
        df = pd.DataFrame(
            {
                "Criterion": ["Vertical cover RV", "Minimum cover RM"],
                "Given cover (m)": [c_rv, c_rm],
                "Required FoS": [f_req, f_req],
                "Factor of safety": [f_rv, f_rm],
                "FoS margin": [f_rv - f_req, f_rm - f_req],
                "Required cover (m)": [c_rv_req, c_rm_req],
                "Cover deficit (m)": [max(c_rv_req - c_rv, 0.0), max(c_rm_req - c_rm, 0.0)],
                "Status": ["PASS" if f_rv >= f_req else "FAIL", "PASS" if f_rm >= f_req else "FAIL"],
            }
        )
        def style_fos_status(row: pd.Series) -> list[str]:
            if row["Status"] == "FAIL":
                return [f"color: {MONASH_ORANGE}; font-weight: 700;" if column == "Status" else "" for column in row.index]
            return [f"color: {MONASH_GREEN}; font-weight: 700;" if column == "Status" else "" for column in row.index]

        st.dataframe(
            df.style.apply(style_fos_status, axis=1).format(
                {
                    "Given cover (m)": "{:.2f}",
                    "Required FoS": "{:.2f}",
                    "Factor of safety": "{:.2f}",
                    "FoS margin": "{:.2f}",
                    "Required cover (m)": "{:.2f}",
                    "Cover deficit (m)": "{:.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ["F < 1.2", "Unsafe / not acceptable for teaching screening"],
                    ["1.2 <= F < 1.5", "Marginal; increase cover or require further geotechnical investigation"],
                    ["F >= 1.5", "Acceptable preliminary stability for this teaching check"],
                ],
                columns=["Safety factor range", "Interpretation"],
            ),
            hide_index=True,
            width="stretch",
        )
        if f_rv >= f_req and f_rm >= f_req:
            st.success("Both confinement checks satisfy the selected factor of safety.")
        else:
            failed_cover = df[df["Status"] == "FAIL"]
            failure_lines = [
                f"{row['Criterion']}: FoS {row['Factor of safety']:.2f} < required {row['Required FoS']:.2f}; "
                f"needs about {row['Required cover (m)']:.1f} m cover, deficit {row['Cover deficit (m)']:.1f} m."
                for _, row in failed_cover.iterrows()
            ]
            st.error(
                "Confinement factor of safety is not satisfied. "
                + " ".join(failure_lines)
                + " Increase cover, reduce hydraulic head, realign/deepen the tunnel, or assume a stronger lining class before accepting this zone."
            )

        st.subheader("Hydraulic jacking check and lining selection")
        st.caption(
            "The cover criteria above are geometric screens. The jacking check compares the minimum in-situ "
            "principal stress with the internal water pressure: if water pressure can exceed σ₃, it can "
            "open joints and jack the rock mass, and an unlined pressure tunnel is not acceptable."
        )
        j1, j2, j3 = st.columns(3)
        with j1:
            sigma3_basis = st.selectbox("Minimum stress σ₃ basis", ["Estimate from cover and k_min", "Measured σ₃"])
        with j2:
            if sigma3_basis == "Measured σ₃":
                sigma3_mpa = st.number_input("Measured minimum stress σ₃ (MPa)", min_value=0.0, value=5.0, step=0.5)
            else:
                k_min = st.number_input("Minimum stress ratio k_min = σ₃/σᵥ", min_value=0.1, max_value=2.0, value=0.6, step=0.05)
                sigma3_mpa = k_min * gamma_r * c_rm / 1000.0
        with j3:
            f_jack_req = st.number_input("Required jacking factor of safety", min_value=1.0, value=1.3, step=0.1)

        water_pressure_mpa = gamma_w * hs / 1000.0
        f_jack = safe_div(sigma3_mpa, water_pressure_mpa)
        jm1, jm2, jm3 = st.columns(3)
        jm1.metric("Minimum stress σ₃", metric_value(sigma3_mpa, " MPa", 2))
        jm2.metric("Internal water pressure pᵢ", metric_value(water_pressure_mpa, " MPa", 2))
        jm3.metric("Jacking FoS σ₃/pᵢ", metric_value(f_jack, "", 2))
        jacking_status = "PASS" if np.isfinite(f_jack) and f_jack >= f_jack_req else "FAIL"
        jacking_df = pd.DataFrame(
            [
                [
                    "Hydraulic jacking",
                    f_jack,
                    f_jack_req,
                    f_jack - f_jack_req if np.isfinite(f_jack) else np.nan,
                    sigma3_mpa,
                    water_pressure_mpa,
                    jacking_status,
                ]
            ],
            columns=["Check", "Factor of safety", "Required FoS", "FoS margin", "Minimum stress MPa", "Water pressure MPa", "Status"],
        )
        st.dataframe(
            jacking_df.style.apply(style_fos_status, axis=1).format(
                {
                    "Factor of safety": "{:.2f}",
                    "Required FoS": "{:.2f}",
                    "FoS margin": "{:.2f}",
                    "Minimum stress MPa": "{:.2f}",
                    "Water pressure MPa": "{:.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        if jacking_status == "FAIL":
            st.error(
                f"Hydraulic jacking factor of safety is not satisfied: FoS {f_jack:.2f} < required {f_jack_req:.2f}. "
                "Use measured in-situ stress if available, increase cover/realign the tunnel, reduce the pressure zone head, or select a pressure lining that can carry the internal pressure."
            )
        render_matplotlib_figure(confinement_jacking_schematic_figure(hs, c_rv, c_rm, sigma3_mpa, water_pressure_mpa, f_jack))

        cover_ok = f_rv >= f_req and f_rm >= f_req
        if np.isfinite(f_jack) and f_jack >= f_jack_req and cover_ok:
            st.success(
                "Confinement and jacking checks pass: an unlined or shotcrete-lined pressure tunnel is plausible "
                "(Norwegian practice). Confirm with in-situ stress measurement, joint/leakage assessment and grouting trials."
            )
        elif np.isfinite(f_jack) and f_jack >= 1.0 and min(f_rv, f_rm) >= 1.0:
            st.warning(
                "Screening margins are limited: a reinforced concrete lining with consolidation grouting is the "
                "likely requirement. Review seepage, drainage and the sensitivity of σ₃ assumptions."
            )
        else:
            st.error(
                "Confinement or jacking check fails: a steel lining, or a deeper/realigned tunnel, is likely "
                "required in this zone. Unlined and concrete-lined options are not supported by the screening checks."
            )

        st.dataframe(
            pd.DataFrame(
                [
                    ["Unlined / shotcrete", "Cover FoS and jacking FoS both at or above the required values", "Lowest cost; relies on confinement and tight rock mass; standard Norwegian solution"],
                    ["Reinforced concrete + grouting", "All screening FoS at or above 1.0 but below the required values", "Controls leakage; internal pressure still shared with the rock mass"],
                    ["Steel lining", "Jacking or cover FoS below 1.0", "Carries full internal pressure; typical for the high-pressure zone near the powerhouse"],
                ],
                columns=["Lining class", "Screening condition", "Teaching note"],
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Head increases along the waterway towards the powerhouse, so the lining class usually changes along the "
            "route: upper reaches unlined or shotcrete-lined, mid reaches concrete-lined, and the final high-pressure "
            "reach steel-lined. Re-run this check at several chainages with the local cover and hydrostatic head."
        )

    with tab_lining:
        st.subheader("Pressure tunnel lining stress")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Inner radius rᵢ (m)", min_value=0.1, step=0.1, key="lining_inner_radius_m")
            st.number_input("Concrete lining thickness (m)", min_value=0.05, step=0.05, key="lining_thickness_m")
        with c2:
            st.number_input("Concrete tensile strength fₜ (MPa)", min_value=0.1, step=0.1, key="lining_tensile_strength_mpa")
            st.number_input("External pressure pₑ (MPa)", min_value=0.0, step=0.1, key="lining_external_pressure_mpa")
        with c3:
            st.number_input("Static head at section hₛ (m)", min_value=0.0, step=10.0, key="lining_static_head_m")
            st.number_input("Transient surcharge ΔH (m)", min_value=0.0, step=5.0, key="lining_transient_surcharge_m")
            st.number_input("Cavern cover depth (m)", min_value=0.0, step=10.0, key="cover_depth_m")

        lining_profile = lining_stress_profile(levels)
        if lining_profile is None:
            st.warning("Enter a positive lining radius and thickness to calculate lining stress.")
            return
        inner_radius = float(lining_profile["inner_radius"])
        outer_radius = float(lining_profile["outer_radius"])
        tensile_strength = float(lining_profile["tensile_strength"])
        external_pressure = float(lining_profile["external_pressure"])
        static_head = float(lining_profile["static_head"])
        transient_surcharge = float(lining_profile["transient_surcharge"])
        internal_pressure = float(lining_profile["internal_pressure"])
        radius = np.asarray(lining_profile["radius"], dtype=float)
        sigma_r = np.asarray(lining_profile["sigma_r"], dtype=float)
        sigma_t = np.asarray(lining_profile["sigma_t"], dtype=float)
        inner_hoop = float(sigma_t[0])
        h_internal = safe_div(internal_pressure * 1e6, RHO * G)
        h_external = safe_div(external_pressure * 1e6, RHO * G)
        m1, m2, m3 = st.columns(3)
        m1.metric("Internal pressure pᵢ", metric_value(internal_pressure, " MPa", 2))
        m2.metric("Inner hoop stress", metric_value(inner_hoop, " MPa", 2))
        m3.metric("Outer radius", metric_value(outer_radius, " m", 2))
        m1, m2 = st.columns(2)
        m1.metric("Allowable tensile strength", metric_value(tensile_strength, " MPa", 2))
        m2.metric("External pressure pₑ", metric_value(external_pressure, " MPa", 2))
        st.dataframe(
            pd.DataFrame(
                [
                    ["Static internal head hₛ", static_head, "m", "Static/gross hydraulic head at the section; do not subtract friction loss for lining screening"],
                    ["Transient surcharge ΔH", transient_surcharge, "m", "Additional pressure head from Step 8 load rejection or pump trip check"],
                    ["Internal design head", h_internal, "m", "hₛ + ΔH converted from the displayed internal pressure"],
                    ["External groundwater head h_w", h_external, "m", "Vertical distance from crown to groundwater/piezometric level; use 0 if below crown"],
                    ["Critical section", np.nan, "-", "Choose the chainage where |hₛ - h_w| and lining stress are largest"],
                ],
                columns=["Annotation", "Value", "Unit", "Definition"],
            ),
            hide_index=True,
            width="stretch",
            column_config={"Value": st.column_config.NumberColumn(format="%.2f")},
        )
        if inner_hoop <= tensile_strength:
            st.success("Hoop stress is within the selected tensile strength.")
        else:
            st.warning("Hoop stress exceeds the selected tensile strength.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=radius, y=sigma_t, name="Hoop stress", line=dict(color=MONASH_ELECTRIC_BLUE, width=3)))
        fig.add_trace(go.Scatter(x=radius, y=sigma_r, name="Radial stress", line=dict(color=MONASH_BLUE, width=3)))
        fig.add_hline(y=tensile_strength, line_dash="dash", line_color=MONASH_GREEN, annotation_text="fₜ")
        fig.update_layout(template="plotly_white", height=390, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Radius (m)", yaxis_title="Stress (MPa)")
        st.plotly_chart(fig, width="stretch")

    with tab_cavern:
        st.subheader("Underground powerhouse cavern")
        with st.expander("How to choose cavern dimensions"):
            st.dataframe(
                pd.DataFrame(CAVERN_DIMENSION_GUIDANCE, columns=["Dimension", "How to choose it", "Teaching reference"]),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Rock mass quality does not automatically change the machine envelope. It changes the required pillar, support, cover, "
                "and verification effort; the hall dimensions should be revised only when equipment layout or stability modelling supports it."
            )
        c1, c2 = st.columns(2)
        with c1:
            shape = st.selectbox("Machine hall shape", ["Mushroom-shaped", "Horseshoe-shaped", "Elliptical"])
            hall_width = st.number_input("Machine hall width B (m)", min_value=5.0, value=25.0, step=1.0)
            unit_width = st.number_input("Unit bay width (m/unit)", min_value=5.0, value=25.0, step=1.0)
            erection_bay = st.number_input("Erection bay length (m)", min_value=0.0, value=30.0, step=5.0)
        with c2:
            if shape == "Mushroom-shaped":
                auto_height = 2.2 * hall_width
            elif shape == "Horseshoe-shaped":
                auto_height = 1.6 * hall_width
            else:
                auto_height = 2.5 * hall_width
            hall_height = st.number_input("Machine hall height H (m)", min_value=5.0, value=float(auto_height), step=1.0)
            hall_length = st.number_input("Machine hall length L (m)", min_value=5.0, value=float(st.session_state.units * unit_width + erection_bay), step=5.0)
            st.number_input("Rock unit weight γ (kN/m³)", min_value=1.0, step=0.5, key="rock_unit_weight")

        cover_depth = st.session_state.cover_depth_m
        sigma_v = st.session_state.rock_unit_weight * cover_depth / 1000.0
        k_ratio = st.slider("K-ratio σₕ/σᵥ", 0.5, 2.0, value=1.0, step=0.1)
        sigma_h = sigma_v * k_ratio
        pillar_quality = st.selectbox("Rock mass quality", ["Good", "Fair", "Poor"])
        if pillar_quality == "Good":
            pillar = max(20.0, 0.5 * hall_width)
        elif pillar_quality == "Fair":
            pillar = max(30.0, 0.8 * hall_width)
        else:
            pillar = max(40.0, 1.0 * hall_width)
        support_note = {
            "Good": "Pattern rock bolts, spot cable bolts, and shotcrete may be plausible after mapping.",
            "Fair": "Expect systematic bolts/cables, thicker shotcrete, drainage, and numerical stress checks.",
            "Poor": "Revise cavern location/span or add heavy support; 3D FEM/DEM verification becomes a must-pass item.",
        }[pillar_quality]

        unit_capacity = safe_div(fnum('design_power_mw'), st.session_state.units)
        turbine_cl = levels["lower_twl"] - fnum('draft_head_m')
        crown_elevation = turbine_cl + hall_height
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Unit capacity", metric_value(unit_capacity, " MW", 1))
        m2.metric("Turbine CL", metric_value(turbine_cl, " m", 1))
        m3.metric("Crown elevation", metric_value(crown_elevation, " m", 1))
        m4.metric("Vertical stress", metric_value(sigma_v, " MPa", 2))
        m5.metric("Horizontal stress", metric_value(sigma_h, " MPa", 2))
        m6.metric("Pillar thickness", metric_value(pillar, " m", 1))
        st.info(f"Rock mass quality selected: {pillar_quality}. {support_note}")

        st.subheader("Transformer hall and IPB gallery")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            transformer_bay = st.number_input("Transformer bay length (m/unit)", min_value=5.0, value=18.0, step=1.0)
        with t2:
            transformer_width = st.number_input("Transformer width plus access (m)", min_value=3.0, value=12.0, step=1.0)
        with t3:
            maintenance_allowance = st.number_input("Transformer maintenance bay (m)", min_value=0.0, value=20.0, step=2.0)
        with t4:
            fire_sep = st.number_input("Fire separation allowance (m)", min_value=0.0, value=8.0, step=1.0)

        ipb1, ipb2, ipb3 = st.columns(3)
        with ipb1:
            bus_diameter = st.number_input("IPB bus duct diameter (m)", min_value=0.2, value=1.0, step=0.1)
        with ipb2:
            phase_clearance = st.number_input("IPB phase/access clearance (m)", min_value=0.2, value=1.2, step=0.1)
        with ipb3:
            vertical_clearance = st.number_input("IPB vertical clearance (m)", min_value=0.2, value=1.0, step=0.1)

        transformer_hall_length = st.session_state.units * transformer_bay + maintenance_allowance + fire_sep
        transformer_hall_width = transformer_width + fire_sep
        ipb_width = 3.0 * bus_diameter + 4.0 * phase_clearance
        ipb_height = bus_diameter + 2.0 * vertical_clearance

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Transformer hall length", metric_value(transformer_hall_length, " m", 1))
        d2.metric("Transformer hall width", metric_value(transformer_hall_width, " m", 1))
        d3.metric("IPB gallery width", metric_value(ipb_width, " m", 1))
        d4.metric("IPB gallery height", metric_value(ipb_height, " m", 1))

        cavern_table = pd.DataFrame(
            [
                ["Machine hall", hall_length, hall_width, hall_height],
                ["Transformer hall", transformer_hall_length, transformer_hall_width, hall_height * 0.55],
                ["IPB gallery", max(hall_length, transformer_hall_length), ipb_width, ipb_height],
            ],
            columns=["Space", "Length m", "Width m", "Height m"],
        )
        st.dataframe(cavern_table, hide_index=True, width="stretch", column_config={col: st.column_config.NumberColumn(format="%.1f") for col in ["Length m", "Width m", "Height m"]})
        st.dataframe(
            pd.DataFrame(
                [
                    ["Hall length", f"N_units x bay width + erection bay = {st.session_state.units} x {metric_value(unit_width, ' m', 1)} + {metric_value(erection_bay, ' m', 1)}", "Equipment layout check"],
                    ["Pillar thickness", f"Quality factor x hall width -> {metric_value(pillar, ' m', 1)}", "Geotechnical stability check"],
                    ["Cover/span", f"Cover / hall width = {metric_value(safe_div(cover_depth, hall_width), '', 2)}", "Preliminary cover adequacy check"],
                    ["Transformer hall", "N_units x transformer bay + maintenance + fire separation", "Electrical/fire-safety layout check"],
                    ["IPB gallery", "3 x bus diameter + access clearances", "Electrical interface check"],
                ],
                columns=["Dimension logic", "Current reference", "Why it matters"],
            ),
            hide_index=True,
            width="stretch",
        )

        if hall_height / hall_width > 3.0:
            st.warning("Hall height/span ratio is above 3.0.")
        if cover_depth < 2.0 * hall_width:
            st.warning("Cover depth is less than twice the span.")
        st.info("This remains a preliminary cavern sizing step. Final verification should use FEA or DEM modelling for stress redistribution, pillar stability and support requirements.")


def page_surge_transient(design_case: dict[str, float], embedded: bool = False) -> None:
    if not embedded:
        st.title("Surge And Transient Check")
    st.caption("Step 8: screen transient severity, define the controlling event, and identify the connected-system model needed before surge protection can be sized.")
    if not (discharge_ready() and conduit_defined()):
        st.info("Surge and transient screening appears once the design discharge exists (Steps 1 and 4) and the conduit length and diameter (Step 5) are set.")
        return

    q_default = design_case.get("total_discharge_m3_s", float("nan"))
    if not np.isfinite(q_default) or q_default <= 0:
        q_default = selected_design_discharge(fnum('teaching_effective_head_m'))

    with st.expander("How to choose surge and water-hammer inputs"):
        st.dataframe(
            pd.DataFrame(STEP8_INPUT_GUIDANCE, columns=["Input", "Starting choice", "Teaching note"]),
            hide_index=True,
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(WAVE_SPEED_GUIDANCE, columns=["Material / waterway", "Teaching wave-speed range", "Why it matters"]),
            hide_index=True,
            width="stretch",
        )
        st.caption("If the pressure rise is high, revise closure time, add surge control, increase diameter, or move the surge tank/powerhouse before final turbine selection.")

    tab_setup, tab_hammer, tab_deliverables = st.tabs(["Transient setup", "Water hammer", "Deliverables"])

    with tab_setup:
        st.subheader("Connected-system transient setup")
        c1, c2, c3 = st.columns(3)
        with c1:
            default_connected = int(st.session_state.penstocks) if st.session_state.get("flow_area_mode") == "Per penstock" else 1
            connected_pipes = st.number_input("Connected conduits", min_value=1, max_value=20, value=max(default_connected, 1), step=1)
            pipe_d = st.number_input("Connected conduit diameter Dₚ (m)", min_value=0.2, value=num_or(fnum('penstock_diameter_m'), 5.0), step=0.1)
        with c2:
            q0 = st.number_input("Rated discharge Q₀ (m³/s)", min_value=0.1, value=max(num_or(q_default, 100.0), 0.1), step=1.0)
            net_head = st.number_input("Net head H (m)", min_value=1.0, value=max(num_or(design_case.get("net_head_m"), fnum('teaching_effective_head_m')), 1.0), step=1.0)
        with c3:
            headrace_length = st.number_input("Headrace length L (m)", min_value=1.0, value=max(num_or(fnum('penstock_length_m'), 1000.0), 1.0), step=50.0)
            event = st.selectbox("Screened event", ["Generating load rejection", "Emergency guide-vane closure", "Pumping trip", "Pump-turbine reversal", "Start-up / filling"])

        ap_single = area_circle(pipe_d)
        ap_total = connected_pipes * ap_single
        event_velocity = safe_div(q0, ap_total)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Connected area", metric_value(ap_total, " m²", 2))
        m2.metric("Event velocity", metric_value(event_velocity, " m/s", 2))
        m3.metric("Screened event", event)
        m4.metric("Surge structure", "Not sized")
        st.warning(
            "This app does not size a surge tank from a universal A_s/A_p ratio or an isolated mass-oscillation formula. "
            "Select and size surge control only from a cited stability/transient criterion and a connected-system model."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ["System boundaries", "Upper/lower reservoirs, branches, valves/guide vanes, reversible machine and any surge-control structure"],
                    ["Operating event", event],
                    ["Machine data", "Guide-vane law, rotating inertia, pump-turbine four-quadrant characteristics and runaway/reversal behaviour"],
                    ["Acceptance limits", "Maximum pressure, minimum pressure/cavitation margin, reservoir/surge levels, conduit ratings and machine limits"],
                    ["Required next model", "Method of characteristics or equivalent connected-system transient simulation"],
                ],
                columns=["Model item", "Evidence required"],
            ),
            hide_index=True,
            width="stretch",
        )

    with tab_hammer:
        st.subheader("Water-hammer sense check")
        c1, c2, c3 = st.columns(3)
        with c1:
            hammer_wave_speed = st.number_input("Wave speed a (m/s)", min_value=100.0, value=1000.0, step=50.0)
        with c2:
            closure_time = st.number_input("Valve/guide-vane closure time T_c (s)", min_value=0.1, value=30.0, step=1.0)
        with c3:
            comparison_rise = st.number_input("Comparison pressure rise (% of head)", min_value=1.0, max_value=100.0, value=30.0, step=5.0, help="Sensitivity line only; not a universal acceptance criterion.")

        initial_velocity = safe_div(q0, ap_total)
        critical_time = safe_div(2.0 * headrace_length, hammer_wave_speed)
        instantaneous_rise = safe_div(hammer_wave_speed * initial_velocity, G)
        rigid_column_rise = safe_div(2.0 * headrace_length * initial_velocity, G * closure_time)
        rapid_closure = closure_time <= critical_time
        estimated_rise = instantaneous_rise if rapid_closure else rigid_column_rise
        method_label = "Joukowsky rapid-closure bound" if rapid_closure else "Rigid-column slower-closure screen"
        comparison_rise_m = net_head * comparison_rise / 100.0
        max_transient_head = net_head + estimated_rise
        min_transient_head = net_head - estimated_rise

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Initial velocity", metric_value(initial_velocity, " m/s", 2))
        m2.metric("Round-trip wave time", metric_value(critical_time, " s", 1))
        m3.metric("Severity ΔH", metric_value(estimated_rise, " m", 1))
        m4.metric("Maximum head", metric_value(max_transient_head, " m", 1))
        m5.metric("Minimum head", metric_value(min_transient_head, " m", 1))

        st.caption(f"Selected equation: {method_label}. These values are severity screens, not predicted plant pressures.")
        if estimated_rise <= comparison_rise_m:
            st.info("The severity screen is below the selected comparison line. This does not establish transient acceptance.")
        else:
            st.warning("The severity screen exceeds the selected comparison line. Revise the event/control concept and run a connected-system model.")
        if min_transient_head <= 0:
            st.error("The simple drawdown screen reaches zero or negative head. Flag low-pressure, cavitation and column-separation risk for detailed modelling.")

        scenarios = []
        scenario_times = [("Fast closure", max(0.5 * critical_time, 0.1)), ("Selected closure", closure_time), ("Slow closure", max(2.0 * closure_time, 0.1))]
        for scenario_name, trial_closure in scenario_times:
            trial_rapid = trial_closure <= critical_time
            trial_rise = instantaneous_rise if trial_rapid else safe_div(2.0 * headrace_length * initial_velocity, G * trial_closure)
            trial_min = net_head - trial_rise
            scenarios.append(
                [
                    scenario_name,
                    trial_closure,
                    trial_rise,
                    net_head + trial_rise,
                    trial_min,
                    "Joukowsky bound" if trial_rapid else "Rigid-column screen",
                    "Low-pressure risk" if trial_min <= 0 else "Screen only",
                ]
            )
        st.dataframe(
            pd.DataFrame(scenarios, columns=["Closure case", "Closure time (s)", "Severity ΔH (m)", "Maximum head (m)", "Minimum head (m)", "Method", "Status"]),
            hide_index=True,
            width="stretch",
            column_config={
                "Closure time (s)": st.column_config.NumberColumn(format="%.1f"),
                "Severity ΔH (m)": st.column_config.NumberColumn(format="%.1f"),
                "Maximum head (m)": st.column_config.NumberColumn(format="%.1f"),
                "Minimum head (m)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    with tab_deliverables:
        st.subheader("Student deliverables")
        deliverables = pd.DataFrame(
            [
                ["Event definition", "Generating load rejection, emergency closure, pumping trip/reversal or start-up, with boundary conditions"],
                ["Connected-system basis", "Connected conduits, area, length, discharge, net/static head, reservoirs and machine/control data"],
                ["Transient severity screen", "Wave speed, event velocity, round-trip wave time, closure time, rapid/slow equation and maximum/minimum head"],
                ["Surge-control concept", "Head-/tailrace surge tank, air cushion, throttling, valve timing, or a justified no-structure concept"],
                ["Design decision", "State that no surge structure is sized here; name the detailed model, acceptance criteria and next design action"],
            ],
            columns=["Item", "Expected evidence"],
        )
        st.dataframe(deliverables, hide_index=True, width="stretch")


def lame_stress(pi_mpa: float, pe_mpa: float, ri: float, ro: float, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denom = ro**2 - ri**2
    sigma_r = (pi_mpa * ri**2 - pe_mpa * ro**2) / denom - ((pi_mpa - pe_mpa) * ri**2 * ro**2) / (denom * radius**2)
    sigma_t = (pi_mpa * ri**2 - pe_mpa * ro**2) / denom + ((pi_mpa - pe_mpa) * ri**2 * ro**2) / (denom * radius**2)
    return sigma_r, sigma_t


def lining_stress_profile(levels: dict[str, float]) -> dict[str, object] | None:
    inner_radius = max(num_or(st.session_state.get("lining_inner_radius_m"), max(num_or(fnum('penstock_diameter_m'), 5.0) / 2.0, 0.1)), 0.1)
    thickness = max(num_or(st.session_state.get("lining_thickness_m"), 0.45), 0.001)
    tensile_strength = max(num_or(st.session_state.get("lining_tensile_strength_mpa"), 3.0), 0.001)
    external_pressure = max(num_or(st.session_state.get("lining_external_pressure_mpa"), 1.0), 0.0)
    static_head = max(num_or(st.session_state.get("lining_static_head_m"), max(num_or(levels.get("gross_head"), 100.0), 0.0)), 0.0)
    transient_surcharge = max(num_or(st.session_state.get("lining_transient_surcharge_m"), 0.0), 0.0)
    outer_radius = inner_radius + thickness
    if outer_radius <= inner_radius:
        return None
    internal_pressure = RHO * G * (static_head + transient_surcharge) / 1e6
    radius = np.linspace(inner_radius * 1.001, outer_radius, 120)
    sigma_r, sigma_t = lame_stress(internal_pressure, external_pressure, inner_radius, outer_radius, radius)
    return {
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "thickness": thickness,
        "tensile_strength": tensile_strength,
        "external_pressure": external_pressure,
        "static_head": static_head,
        "transient_surcharge": transient_surcharge,
        "internal_pressure": internal_pressure,
        "radius": radius,
        "sigma_r": sigma_r,
        "sigma_t": sigma_t,
    }


def lining_stress_figure(levels: dict[str, float]) -> plt.Figure | None:
    profile = lining_stress_profile(levels)
    if profile is None:
        return None
    radius = profile["radius"]
    sigma_t = profile["sigma_t"]
    sigma_r = profile["sigma_r"]
    tensile_strength = float(profile["tensile_strength"])
    inner_hoop = float(np.asarray(sigma_t)[0])
    project_name = str(st.session_state.get("project_name") or "PHES scheme")
    check_color = MONASH_GREEN if inner_hoop <= tensile_strength else MONASH_ORANGE

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(radius, sigma_t, color=MONASH_ELECTRIC_BLUE, linewidth=2.6, label=r"Hoop stress $\sigma_\theta$")
    ax.plot(radius, sigma_r, color=MONASH_BLUE, linewidth=2.6, label=r"Radial stress $\sigma_r$")
    ax.axhline(tensile_strength, color=MONASH_GREEN, linestyle="--", linewidth=1.5, label=r"Allowable $f_t$")
    ax.scatter([float(profile["inner_radius"])], [inner_hoop], color=check_color, s=42, zorder=3, label="Inner lining check")
    ax.annotate(
        f"inner hoop = {inner_hoop:.2f} MPa",
        (float(profile["inner_radius"]), inner_hoop),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        color=check_color,
    )
    ax.set_xlabel("Radius r (m)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"{project_name} - pressure tunnel lining stress")
    ax.grid(True, ls="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def fig_png_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def turbine_application_figure(discharge_m3_s: float, head_m: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.fill([0.1, 50, 50, 0.1], [50, 50, 2000, 2000], alpha=0.22, color=MONASH_BLUEBERRY, label="Pelton")
    ax.fill([0.5, 100, 200, 10, 0.5], [20, 20, 100, 700, 700], alpha=0.25, color=MONASH_BLUE, label="Francis")
    ax.fill([10, 1000, 1000, 10], [5, 5, 100, 100], alpha=0.45, color=MONASH_HERITAGE_BLUE, label="Kaplan")
    ax.fill([50, 1000, 1000, 50], [5, 5, 20, 20], alpha=0.28, color=MONASH_GREY_2, label="Bulb")
    if np.isfinite(discharge_m3_s) and np.isfinite(head_m):
        ax.plot(discharge_m3_s, head_m, "o", color=MONASH_ELECTRIC_BLUE, markersize=8, label="Current design")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(5, 2000)
    ax.set_xlabel("Discharge Q (m$^3$/s)")
    ax.set_ylabel("Head H (m)")
    ax.set_title("Turbine application zones")
    ax.grid(True, which="both", ls="--", alpha=0.28)
    ax.legend(loc="lower left")
    return fig


def scheme_long_section_figure(levels: dict[str, float]) -> plt.Figure | None:
    length = fnum('penstock_length_m')
    upper = levels["upper_nwl"]
    lower = levels["lower_twl"]
    if not (np.isfinite(length) and length > 0 and np.isfinite(upper) and np.isfinite(lower) and upper > lower):
        return None
    draft = fnum('draft_head_m')
    draft = 0.0 if not np.isfinite(draft) else draft
    turbine_cl = lower - draft
    drop = upper - lower
    chainage = np.array([0.0, 0.194, 0.472, 0.75, 1.0]) * length
    waterway = np.array([upper, upper - 0.04 * drop, lower + 0.35 * drop, turbine_cl, lower])
    ground = np.array([upper + 0.03 * drop, upper - 0.13 * drop, upper - 0.43 * drop, lower + 0.20 * drop, lower + 0.04 * drop])

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(chainage, ground, color=MONASH_GREY_2, linestyle="--", linewidth=1.6, label="Ground surface (indicative)")
    ax.plot(chainage, waterway, color=MONASH_BLUE, linewidth=3.0, marker="o", markersize=5, label="Waterway alignment")

    upper_hwl = num_or(levels["upper_hwl"], upper + 0.02 * drop)
    upper_lwl = num_or(levels["upper_lwl"], upper - 0.02 * drop)
    lower_hwl = num_or(levels["lower_hwl"], lower + 0.02 * drop)
    ax.fill_between([-0.075 * length, 0.005 * length], upper_lwl, upper_hwl, color=MONASH_HERITAGE_BLUE, alpha=0.75, zorder=1)
    ax.fill_between([0.995 * length, 1.075 * length], lower, lower_hwl, color=MONASH_HERITAGE_BLUE, alpha=0.75, zorder=1)
    ax.annotate("Upper reservoir", (-0.035 * length, upper_hwl + 0.03 * drop), ha="center", fontsize=9, color=MONASH_BLUEBERRY)
    ax.annotate("Lower reservoir", (1.035 * length, lower_hwl + 0.03 * drop), ha="center", fontsize=9, color=MONASH_BLUEBERRY)

    ph_height = 0.05 * drop
    ax.add_patch(plt.Rectangle((0.735 * length, turbine_cl - 0.4 * ph_height), 0.03 * length, ph_height, facecolor=MONASH_GREY_1, zorder=3))
    ax.annotate("Powerhouse", (0.75 * length, turbine_cl - 0.10 * drop), ha="center", fontsize=9, color=MONASH_GREY_1)

    ax.annotate(
        "",
        xy=(0.58 * length, upper),
        xytext=(0.58 * length, lower),
        arrowprops=dict(arrowstyle="<->", color=MONASH_GREY_1, lw=1.6),
    )
    ax.annotate(f"$H_g$ = {drop:.0f} m", (0.595 * length, lower + 0.5 * drop), fontsize=10, color=MONASH_GREY_1)
    teaching_head = fnum('teaching_effective_head_m')
    if np.isfinite(teaching_head):
        ax.annotate(f"$H_e$ = {teaching_head:.0f} m", (0.595 * length, lower + 0.42 * drop), fontsize=10, color=MONASH_ELECTRIC_BLUE)

    ax.set_xlabel("Chainage (m)")
    ax.set_ylabel("Elevation (m AHD)")
    project_name = str(st.session_state.get("project_name") or "PHES scheme")
    ax.set_title(f"{project_name} - long section (screening schematic)")
    ax.set_xlim(-0.10 * length, 1.10 * length)
    ax.grid(True, ls="--", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    return fig


def reservoir_levels_figure(levels: dict[str, float]) -> plt.Figure | None:
    if not levels_complete():
        return None
    upper_hwl, upper_lwl = levels["upper_hwl"], levels["upper_lwl"]
    upper_nwl, lower_hwl, lower_twl = levels["upper_nwl"], levels["lower_hwl"], levels["lower_twl"]

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.fill_between([0.0, 1.0], upper_lwl, upper_hwl, color=MONASH_HERITAGE_BLUE, alpha=0.7)
    ax.fill_between([2.6, 3.6], lower_twl, lower_hwl, color=MONASH_HERITAGE_BLUE, alpha=0.7)

    for value, label, color, x0, x1 in [
        (upper_hwl, "HWL", MONASH_BLUE, -0.05, 1.05),
        (upper_nwl, "Representative", MONASH_ELECTRIC_BLUE, -0.05, 1.05),
        (upper_lwl, "LWL", MONASH_BLUEBERRY, -0.05, 1.05),
        (lower_hwl, "HWL", MONASH_BLUE, 2.55, 3.65),
        (lower_twl, "TWL", MONASH_BLUEBERRY, 2.55, 3.65),
    ]:
        ax.hlines(value, x0, x1, color=color, linewidth=1.8)
        ax.annotate(f"{label} {value:.1f} m", (x1 + 0.03, value), va="center", fontsize=9, color=color)

    crest_level = levels["upper_hwl"] + fnum('freeboard_m') + fnum('wave_allowance_m') + fnum('settlement_allowance_m')
    if np.isfinite(crest_level):
        ax.hlines(crest_level, -0.15, 1.15, color=MONASH_GREY_1, linestyle="--", linewidth=1.6)
        ax.annotate(f"Crest {crest_level:.1f} m", (-0.18, crest_level), va="center", ha="right", fontsize=9, color=MONASH_GREY_1)

    ax.annotate(
        "",
        xy=(1.8, upper_nwl),
        xytext=(1.8, lower_twl),
        arrowprops=dict(arrowstyle="<->", color=MONASH_GREY_1, lw=1.6),
    )
    ax.annotate(f"$H_{{g,rep}}$ = {levels['gross_head']:.1f} m", (1.86, 0.5 * (upper_nwl + lower_twl)), fontsize=10, color=MONASH_GREY_1)
    ax.annotate("Upper reservoir", (0.5, upper_lwl - 0.06 * (upper_hwl - lower_twl)), ha="center", fontsize=10)
    ax.annotate("Lower reservoir", (3.1, lower_twl - 0.06 * (upper_hwl - lower_twl)), ha="center", fontsize=10)

    ax.set_xlim(-1.3, 4.9)
    ax.set_ylabel("Elevation (m AHD)")
    ax.set_xticks([])
    project_name = str(st.session_state.get("project_name") or "PHES scheme")
    ax.set_title(f"{project_name} - reservoir operating levels")
    ax.grid(True, axis="y", ls="--", alpha=0.25)
    return fig


def system_curves_figure(levels: dict[str, float], design_case: dict[str, float]) -> plt.Figure | None:
    q_design = design_case["total_discharge_m3_s"]
    if not (np.isfinite(q_design) and q_design > 0 and hydraulics_ready()):
        return None
    area = area_circle(fnum('penstock_diameter_m'))
    nu = water_nu_kinematic_m2_s(float(st.session_state.temperature_c))
    k_sum = selected_k_sum()
    selected_head = fnum('teaching_effective_head_m')
    other_loss = fnum('other_head_loss_m')
    other_loss = 0.0 if not np.isfinite(other_loss) else other_loss
    eta_gen = st.session_state.eta_turbine * st.session_state.eta_generator * st.session_state.eta_transformer
    q_range = np.linspace(0.0, q_design * 1.6, 120)
    head_net, power = [], []
    for q_total in q_range:
        q_per = safe_div(q_total, int(st.session_state.penstocks))
        flow = q_total if st.session_state.flow_area_mode == "Shared conduit" else q_per
        velocity = safe_div(flow, area)
        reynolds = safe_div(velocity * fnum('penstock_diameter_m'), nu)
        friction = f_swamee_jain(reynolds, safe_div(st.session_state.roughness_m, fnum('penstock_diameter_m')))
        loss = head_loss(friction, fnum('penstock_length_m'), fnum('penstock_diameter_m'), velocity, k_sum)
        h_net = max(selected_head - other_loss - loss, 0.0) if np.isfinite(loss) else float("nan")
        head_net.append(h_net)
        power.append(power_from_q_head(q_total, h_net, eta_gen) if np.isfinite(h_net) else float("nan"))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(q_range, head_net, color=MONASH_BLUE, linewidth=2.6, label="Net head")
    ax.set_xlabel("Total discharge Q (m$^3$/s)")
    ax.set_ylabel("Net head (m)", color=MONASH_BLUE)
    ax.tick_params(axis="y", labelcolor=MONASH_BLUE)
    ax2 = ax.twinx()
    ax2.plot(q_range, power, color=MONASH_ELECTRIC_BLUE, linewidth=2.6, label="Generated power")
    ax2.set_ylabel("Generated power (MW)", color=MONASH_ELECTRIC_BLUE)
    ax2.tick_params(axis="y", labelcolor=MONASH_ELECTRIC_BLUE)
    ax.axvline(q_design, color=MONASH_BLUEBERRY, linestyle="--", linewidth=1.6)
    ax.annotate(f"Design Q = {q_design:.0f} m$^3$/s", (q_design, ax.get_ylim()[0]), xytext=(6, 12), textcoords="offset points", fontsize=9, color=MONASH_BLUEBERRY)
    project_name = str(st.session_state.get("project_name") or "PHES scheme")
    ax.set_title(f"{project_name} - system head and power curves")
    ax.grid(True, ls="--", alpha=0.25)
    return fig


def report_figure_assets(levels: dict[str, float], design_case: dict[str, float]) -> dict[str, bytes]:
    """PNG bytes for every report figure that the current inputs can produce.
    The filenames are referenced by the extended LaTeX report template."""
    assets: dict[str, bytes] = {}
    fig = scheme_long_section_figure(levels)
    if fig is not None:
        assets["fig_long_section.png"] = fig_png_bytes(fig)
    fig = reservoir_levels_figure(levels)
    if fig is not None:
        assets["fig_reservoir_levels.png"] = fig_png_bytes(fig)
    fig = system_curves_figure(levels, design_case)
    if fig is not None:
        assets["fig_system_curves.png"] = fig_png_bytes(fig)
    fig = lining_stress_figure(levels)
    if fig is not None:
        assets["fig_lining_stress.png"] = fig_png_bytes(fig)
    teaching_head = fnum('teaching_effective_head_m')
    if np.isfinite(design_case["total_discharge_m3_s"]) and np.isfinite(teaching_head):
        assets["fig_turbine_chart.png"] = fig_png_bytes(turbine_application_figure(design_case["total_discharge_m3_s"], teaching_head))
    return assets


def render_report_figures(levels: dict[str, float], design_case: dict[str, float], project_stem: str) -> None:
    st.subheader("Report figures (PNG downloads)")
    st.caption(
        "Report-ready schematic drawings generated from the current design values. Insert them into the "
        "relevant report step; each caption should state that they are screening-level schematics, not survey drawings."
    )
    teaching_head = fnum('teaching_effective_head_m')
    turbine_fig = None
    if np.isfinite(design_case["total_discharge_m3_s"]) and np.isfinite(teaching_head):
        turbine_fig = turbine_application_figure(design_case["total_discharge_m3_s"], teaching_head)
    figures = [
        ("PHES scheme long section", f"{project_stem}_long_section.png", scheme_long_section_figure(levels), "Steps 2 and 4 evidence"),
        ("Reservoir operating levels", f"{project_stem}_reservoir_levels.png", reservoir_levels_figure(levels), "Step 3 evidence"),
        ("System head and power curves", f"{project_stem}_system_curves.png", system_curves_figure(levels, design_case), "Step 5 evidence"),
        ("Pressure tunnel lining stress", f"{project_stem}_lining_stress.png", lining_stress_figure(levels), "Step 7 evidence"),
        ("Turbine application chart", f"{project_stem}_turbine_chart.png", turbine_fig, "Step 9 evidence"),
    ]
    for title, filename, fig, used_for in figures:
        if fig is None:
            st.caption(f"{title}: complete the required inputs to generate this figure ({used_for}).")
            continue
        with st.expander(f"{title} ({used_for})"):
            st.pyplot(fig, clear_figure=False)
            st.download_button(
                f"Download {title} (PNG)",
                data=fig_png_bytes(fig),
                file_name=filename,
                mime="image/png",
                width="stretch",
                key=f"figure_download_{filename}",
            )


def render_report_assembly(levels: dict[str, float], design_case: dict[str, float]) -> None:
    st.subheader("Report pack")
    st.caption(
        "Use this page after completing Steps 1-10. The report should explain the design logic, equations, "
        "data sources and evidence behind each selected value, not only copy the app outputs."
    )
    if not (step1_complete() and head_defined()):
        st.info("Report exports appear once the project is defined: complete Step 1 (design power and duration) and set the effective head (Step 4). The exports are pre-populated with your current design values.")
        st.dataframe(report_logic_dataframe(), hide_index=True, width="stretch")
        return

    st.subheader("Design logic by workflow step")
    st.dataframe(report_logic_dataframe(), hide_index=True, width="stretch")

    st.session_state.setdefault("market_role", "Daily energy shifting")
    market_role = st.session_state.market_role
    project_stem = safe_file_stem(st.session_state.get("project_name", "phes_design"))

    st.subheader("Short design summary pack")
    st.caption("This export is intentionally compact, usually about 2 pages. Use it for a quick design snapshot, not for the final project report.")
    summary_source = build_latex_summary_source(levels, design_case, market_role)
    summary_stem = f"{project_stem}_summary"
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download short summary LaTeX (.tex)",
            data=summary_source,
            file_name=f"{summary_stem}.tex",
            mime="text/x-tex",
            width="stretch",
        )
    with c2:
        if st.button("Generate short summary PDF", width="stretch"):
            pdf_bytes, pdf_error = compile_latex_to_pdf(summary_source, summary_stem)
            st.session_state["summary_pdf_bytes"] = pdf_bytes
            st.session_state["summary_pdf_error"] = pdf_error
            st.session_state["summary_pdf_name"] = f"{summary_stem}.pdf"

        if st.session_state.get("summary_pdf_bytes"):
            st.download_button(
                "Download short summary PDF",
                data=st.session_state["summary_pdf_bytes"],
                file_name=st.session_state.get("summary_pdf_name", f"{summary_stem}.pdf"),
                mime="application/pdf",
                width="stretch",
            )
        elif st.session_state.get("summary_pdf_error"):
            st.warning(st.session_state["summary_pdf_error"])

    with st.expander("Preview summary LaTeX source"):
        st.code(summary_source, language="latex")

    st.subheader("Extended final design report")
    st.info(
        "This is the extended report template for final project work. It expands Steps 1-10 into purpose, "
        "method, equations, current outputs, evidence requirements, and discussion prompts, and embeds the "
        "report figures below at Steps 3, 4, 5, 7, and 9. Students should replace placeholders with maps, "
        "spreadsheet checks, sensitivity cases and design evidence."
    )
    figure_assets = report_figure_assets(levels, design_case)
    if figure_assets:
        st.caption(f"Figures embedded in this export: {', '.join(sorted(figure_assets))}.")
    else:
        st.caption("No figures are available yet; the report compiles without them and picks them up automatically once the relevant inputs exist.")
    report_source = build_latex_report_source(levels, design_case, market_role, set(figure_assets))
    report_stem = f"{project_stem}_final_report"
    report_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(report_zip_buffer, "w", zipfile.ZIP_DEFLATED) as report_zip:
        report_zip.writestr(f"{report_stem}.tex", report_source)
        for asset_name, asset_bytes in figure_assets.items():
            report_zip.writestr(asset_name, asset_bytes)
    c3, c4 = st.columns(2)
    with c3:
        st.download_button(
            "Download extended report pack (LaTeX + figures, .zip)",
            data=report_zip_buffer.getvalue(),
            file_name=f"{report_stem}.zip",
            mime="application/zip",
            width="stretch",
        )
        st.caption("The .zip contains the .tex source plus the embedded PNG figures, so it compiles standalone.")
    with c4:
        if st.button("Generate extended report PDF", width="stretch"):
            pdf_bytes, pdf_error = compile_latex_to_pdf(report_source, report_stem, figure_assets)
            st.session_state["full_report_pdf_bytes"] = pdf_bytes
            st.session_state["full_report_pdf_error"] = pdf_error
            st.session_state["full_report_pdf_name"] = f"{report_stem}.pdf"

        if st.session_state.get("full_report_pdf_bytes"):
            st.download_button(
                "Download extended report PDF",
                data=st.session_state["full_report_pdf_bytes"],
                file_name=st.session_state.get("full_report_pdf_name", f"{report_stem}.pdf"),
                mime="application/pdf",
                width="stretch",
            )
        elif st.session_state.get("full_report_pdf_error"):
            st.warning(st.session_state["full_report_pdf_error"])

    with st.expander("Preview full report LaTeX source"):
        st.code(report_source, language="latex")

    render_report_figures(levels, design_case, project_stem)


def page_equations_references(levels: dict[str, float], design_case: dict[str, float]) -> None:
    st.title("Report / Refs / Equations")

    tab_report, tab_h, tab_t, tab_c, tab_refs = st.tabs(["Report pack", "Hydraulics", "Turbines", "Civil checks", "Reference tables"])
    with tab_report:
        render_report_assembly(levels, design_case)
    with tab_h:
        st.latex(r"Q = \frac{P \times 10^6}{\rho g H \eta}")
        st.latex(r"H_{g,rep}=z_{u,rep}-z_{l,rep}")
        st.latex(r"CL = TWL_{lower} - h_{set}")
        st.latex(r"A = \frac{\pi D^2}{4}")
        st.latex(r"v = \frac{Q_v}{A}")
        st.latex(r"h_f = \left(f\frac{L}{D} + \Sigma K\right)\frac{v^2}{2g}")
        st.latex(r"H_e = H_{sel} - h_{major} - h_{minor} - h_{other}")
        st.latex(r"Re = \frac{vD}{\nu}")
        st.latex(r"H_{g,max}=HWL_u-LWL_l,\qquad H_{g,min}=LWL_u-HWL_l")
    with tab_t:
        st.latex(r"P_{hydraulic} = \rho g Q H")
        st.latex(r"P_{gen} = \rho g Q H \eta_{total}/10^6")
        st.latex(r"\eta_{total} = \eta_t\eta_g\eta_{tr}")
        st.latex(r"n_q = \frac{N\sqrt{Q_u}}{H^{3/4}}")
        st.latex(r"N_s = \frac{N\sqrt{P_u}}{H^{5/4}}")
        st.latex(r"P_{pump} = \frac{\rho g Q H}{\eta_p10^6}")
    with tab_c:
        st.latex(r"RL_{crest} = HWL + F_b + H_w + S_a")
        st.latex(r"H_{dam} = RL_{crest} - RL_{foundation}")
        st.latex(r"A_{active} \approx \frac{V_{active}}{HWL-LWL}")
        st.latex(r"A_{section} \approx H_{dam}\left[b_c + \frac{(m_u+m_d)H_{dam}}{2}\right]")
        st.latex(r"V_{dam} \approx A_{section}L_{crest}")
        st.latex(r"F_{RV} = \frac{C_{RV}\gamma_r\cos\alpha}{h_s\gamma_w}")
        st.latex(r"F_{RM} = \frac{C_{RM}\gamma_r\cos\beta}{h_s\gamma_w}")
        st.latex(r"\sigma_r(r) = \frac{p_i r_i^2 - p_e r_o^2}{r_o^2-r_i^2} - \frac{(p_i-p_e)r_i^2r_o^2}{(r_o^2-r_i^2)r^2}")
        st.latex(r"\sigma_\theta(r) = \frac{p_i r_i^2 - p_e r_o^2}{r_o^2-r_i^2} + \frac{(p_i-p_e)r_i^2r_o^2}{(r_o^2-r_i^2)r^2}")
        st.latex(r"A_{conn}=n\frac{\pi D^2}{4},\quad v_0=\frac{Q_0}{A_{conn}},\quad T_w=\frac{2L}{a}")
        st.latex(r"\Delta H_J=\frac{av_0}{g}\ (T_c\le T_w),\quad \Delta H_{RC}=\frac{2Lv_0}{gT_c}\ (T_c>T_w)")
    with tab_refs:
        friction = pd.DataFrame(
            {
                "Material": [key for key, value in ROUGHNESS.items() if value is not None],
                "Absolute roughness m": [value for value in ROUGHNESS.values() if value is not None],
            }
        )
        losses = pd.DataFrame({"Component": list(LOSS_COMPONENTS.keys()), "K value": list(LOSS_COMPONENTS.values())})
        st.subheader("Absolute roughness")
        st.dataframe(friction, hide_index=True, width="stretch", column_config={"Absolute roughness m": st.column_config.NumberColumn(format="%.6f")})
        st.subheader("Local loss coefficients")
        st.dataframe(losses, hide_index=True, width="stretch")
        st.subheader("Scientific and practice screening evidence")
        render_reference_links(EVIDENCE_SOURCE_LINKS)
        st.subheader("Open-source GIS tools and adoptable repositories")
        render_reference_links(OPEN_SOURCE_GIS_LINKS)
        st.subheader("Evidence-based screening fields")
        st.dataframe(
            pd.DataFrame(SCREENING_CRITERIA, columns=["Screen", "Scientific / practice basis", "Design use"]),
            hide_index=True,
            width="stretch",
        )
        st.subheader("QGIS screening workflow")
        st.dataframe(
            pd.DataFrame(QGIS_SCREENING_WORKFLOW, columns=["Order", "QGIS task", "How to do the check", "Output for this app"]),
            hide_index=True,
            width="stretch",
        )
        st.subheader("QGIS tool cookbook")
        st.dataframe(
            pd.DataFrame(QGIS_TOOL_COOKBOOK, columns=["Task", "QGIS / plugin tools", "Inputs", "Outputs", "Teaching note"]),
            hide_index=True,
            width="stretch",
        )
        st.subheader("Tutorial demo options")
        st.dataframe(
            pd.DataFrame(QGIS_TUTORIAL_DEMOS, columns=["Demo", "Starting data", "Tool sequence", "Student output", "How it connects to this app"]),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download QGIS candidate CSV template",
            data=pd.DataFrame(columns=QGIS_CANDIDATE_TEMPLATE_COLUMNS).to_csv(index=False),
            file_name="phes_qgis_candidate_template.csv",
            mime="text/csv",
            width="stretch",
        )
        st.subheader("Relative cost-rank reference")
        st.dataframe(
            pd.DataFrame(SCREENING_COST_CLASS_GUIDANCE, columns=["Rank", "Screening meaning", "How to use"]),
            hide_index=True,
            width="stretch",
        )
        render_screening_limitation()
        st.subheader("Teaching references")
        st.markdown(
            """
            - Peer-reviewed GIS/MCDA and global closed-loop PHES resource-screening studies.
            - Practice guidance on topography, geology, access, grid integration, environmental sustainability, and social benefit.
            - Open-source QGIS, GRASS GIS, WhiteboxTools, and PHES-specific DEM-search repositories, subject to licence review.
            - USBR, Design of Small Dams, 3rd edition.
            - USACE EM 1110-2-1602, Hydraulic Design of Reservoir Outlet Works.
            - ICOLD bulletins on pressure tunnels, surge tanks and hydropower structures.
            - Chaudhry, Applied Hydraulic Transients.
            - Gordon, Hydraulics of Hydroelectric Power.
            - Hoek and Brown, Practical estimates of rock mass properties.
            """
        )
        st.caption("This app supports classroom learning and scoping studies. It is not a detailed design tool.")


def page_report_risks(levels: dict[str, float], design_case: dict[str, float], embedded: bool = False) -> None:
    if not embedded:
        st.title("NEM Integration And Risk")
    st.caption("Step 10: connect the technical design to energy storage need, NEM/grid operation, project risks, economics, environmental monitoring, and the final recommendation.")
    if not (discharge_ready() and has_values("reservoir_volume_m3")):
        st.info("NEM integration and economics appear once the design power and duration (Step 1), active storage (Step 3), and effective head (Step 4) are set.")
        return

    eta_gen = st.session_state.eta_turbine * st.session_state.eta_generator * st.session_state.eta_transformer
    eta_cycle = eta_gen * st.session_state.eta_pump
    energy_head = design_case["net_head_m"] if np.isfinite(design_case["net_head_m"]) else fnum('teaching_effective_head_m')
    e_deliverable_gwh = RHO * G * fnum('reservoir_volume_m3') * max(energy_head, 0.0) * eta_gen / 3.6e12
    pump_energy_gwh = safe_div(e_deliverable_gwh, eta_cycle)
    duration_h = safe_div(e_deliverable_gwh * 1000.0, fnum('design_power_mw'))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Deliverable storage", metric_value(e_deliverable_gwh, " GWh", 1))
    c2.metric("Full-load duration", metric_value(duration_h, " h", 1))
    c3.metric("Cycle efficiency", metric_value(eta_cycle * 100.0, "%", 1))
    c4.metric("Pumping energy", metric_value(pump_energy_gwh, " GWh", 1))
    c5.metric("Grid power", metric_value(fnum('design_power_mw'), " MW", 0))

    st.session_state.setdefault("market_role", "Daily energy shifting")
    ensure_option_state("summary_screening_cost_rank", SCREENING_COST_CLASSES)
    tab_market, tab_schedule, tab_economics, tab_risk = st.tabs(["NEM/storage role", "Operation schedule", "Economics & traceability", "Risk register"])

    with tab_market:
        st.subheader("Energy storage and market role")
        role = st.selectbox(
            "Primary operating role",
            ["Daily energy shifting", "Renewable firming", "Peaking capacity", "Reserve/reliability", "Black start/system restoration", "Long-duration storage"],
            key="market_role",
        )
        st.dataframe(
            pd.DataFrame(MARKET_ROLE_GUIDANCE, columns=["Role", "What must be quantified", "Where it changes the report"]),
            hide_index=True,
            width="stretch",
        )
        dispatch_rows = pd.DataFrame(
            [
                ["Charging/pumping window", "Low price, high renewable output, or system minimum demand periods"],
                ["Generation window", "Peak price, low renewable output, reliability events, or reserve activation"],
                ["Transmission/grid connection", "Connection capacity, voltage level, substation distance, congestion, losses, nearby renewable energy zone"],
                ["System services", "FCAS, inertia/system strength, black start, reserve, voltage support"],
                ["Operating constraints", "Reservoir levels, environmental flow, pumping power, round-trip efficiency"],
            ],
            columns=["NEM item", "Student discussion"],
        )
        st.dataframe(dispatch_rows, hide_index=True, width="stretch")
        st.info(
            f"Selected role: {role}. This changes the discussion, evidence requirements, and risk register. "
            "It changes the economic result only when students convert the role into an annual benefit/revenue proxy in the Economics tab."
        )

    with tab_schedule:
        st.subheader("Generation, pumping, and daily cycling")
        st.text_input(
            "Price-series source, date range and price basis",
            key="schedule_price_source",
            placeholder="For example: AEMO/NEM interval prices, region, date range, nominal/real basis",
        )
        c1, c2 = st.columns(2)
        with c1:
            generation_hours_day = st.number_input("Generation hours per day", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
        with c2:
            pumping_hours_day = st.number_input("Pumping hours per day", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
        p1, p2, p3 = st.columns(3)
        with p1:
            generation_price = st.number_input("Generation price proxy ($/MWh)", min_value=0.0, value=0.0, step=10.0)
        with p2:
            pumping_price = st.number_input("Pumping price proxy ($/MWh)", min_value=0.0, value=0.0, step=10.0)
        with p3:
            operating_days_year = st.number_input("Operating days per year", min_value=0, max_value=366, value=0, step=10)
        q_operating = design_case["total_discharge_m3_s"] if np.isfinite(design_case["total_discharge_m3_s"]) else selected_design_discharge(fnum('teaching_effective_head_m'))
        operating_head = design_case["net_head_m"] if np.isfinite(design_case["net_head_m"]) else fnum('teaching_effective_head_m')
        p_gen_mw = power_from_q_head(q_operating, operating_head, eta_gen)
        p_pump_mw = safe_div(RHO * G * q_operating * operating_head, st.session_state.eta_pump * 1e6)
        daily_generation_mwh = p_gen_mw * generation_hours_day
        daily_pumping_mwh = p_pump_mw * pumping_hours_day
        daily_cycle_eff = safe_div(daily_generation_mwh, daily_pumping_mwh)
        daily_net_value_m = (daily_generation_mwh * generation_price - daily_pumping_mwh * pumping_price) / 1e6
        annual_net_value_m = daily_net_value_m * operating_days_year
        st.session_state["schedule_annual_benefit_m"] = annual_net_value_m
        schedule_rows = pd.DataFrame(
            [
                ["Full-capacity storage duration", duration_h, "h", "Use for maximum theoretical discharge period"],
                ["Generation power", p_gen_mw, "MW", "Turbine mode, reduced by turbine/generator/transformer losses"],
                ["Pumping demand", p_pump_mw, "MW", "Pump mode, electrical demand increases because pump efficiency is less than 1"],
                ["Daily generation", daily_generation_mwh, "MWh/day", "Generation window, e.g. evening peak"],
                ["Daily pumping", daily_pumping_mwh, "MWh/day", "Charging/pumping window, e.g. low-price or high-renewable periods"],
                ["Daily cycle efficiency", daily_cycle_eff * 100.0, "%", "Energy generated divided by energy used for pumping"],
                ["Daily net value proxy", daily_net_value_m, "$M/day", "Generation value minus pumping energy cost using the price proxies above"],
                ["Annual benefit proxy", annual_net_value_m, "$M/y", "Can be copied into Economics as a simple benefit input"],
            ],
            columns=["Quantity", "Value", "Unit", "Interpretation"],
        )
        st.dataframe(schedule_rows, hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.2f")})
        st.caption("The schedule uses the same representative operating head in both directions. Replace it with separate generation and pumping head cases when level-dependent and direction-dependent losses are available.")
        st.caption("Zero values mean no market-value result is claimed. Use a dated price series and consistent dispatch interval; do not add energy-arbitrage value again inside a bundled services benefit.")

    with tab_economics:
        st.subheader("Construction-cost screening and design traceability")
        schedule_proxy = st.session_state.get("schedule_annual_benefit_m", float("nan"))
        if np.isfinite(schedule_proxy) and schedule_proxy > 0 and str(st.session_state.get("schedule_price_source", "")).strip():
            if st.button("Use operation schedule benefit proxy", width="stretch"):
                st.session_state["summary_annual_benefit_m"] = float(schedule_proxy)
                st.session_state["summary_benefit_source"] = str(st.session_state["schedule_price_source"])
                st.success(f"Copied {metric_value(float(schedule_proxy), ' $M/y', 1)} and its source into the annual benefit proxy.")
        st.caption(
            "NEM role and operation schedule do not silently change the economic comparison. "
            "They become inputs only when converted into a sourced annual proxy below. Keep arbitrage, capacity and service benefits mutually exclusive unless the dispatch logic proves they can coexist."
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            capital_cost_m = st.number_input("Screening capital cost ($M)", min_value=0.0, value=0.0, step=100.0, key="summary_capital_cost_m")
        with c2:
            annual_om_percent = st.number_input("Annual O&M (% of capital)", min_value=0.0, max_value=10.0, value=1.5, step=0.1, key="summary_annual_om_percent")
        with c3:
            annual_benefit_m = st.number_input("Annual benefit/revenue proxy ($M/y)", min_value=0.0, value=0.0, step=10.0, key="summary_annual_benefit_m")
        with c4:
            screening_life_y = st.number_input("Screening life (years)", min_value=1, max_value=100, value=40, step=1, key="summary_screening_life_y")
        with c5:
            screening_cost_rank = st.selectbox("Desktop cost rank", SCREENING_COST_CLASSES, key="summary_screening_cost_rank")

        s1, s2 = st.columns(2)
        with s1:
            cost_source = st.text_input("Cost source, scope, base date and price basis", key="summary_cost_source")
        with s2:
            benefit_source = st.text_input("Benefit source, dispatch basis and date range", key="summary_benefit_source")

        cost_per_kw = safe_div(capital_cost_m * 1e6, fnum('design_power_mw') * 1000.0)
        cost_per_kwh = safe_div(capital_cost_m * 1e6, e_deliverable_gwh * 1e6)
        annual_om_m = capital_cost_m * annual_om_percent / 100.0
        simple_bc = undiscounted_benefit_cost_proxy(
            capital_cost_m,
            annual_om_m,
            annual_benefit_m,
            screening_life_y,
            cost_source,
            benefit_source,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Cost per kW", metric_value(cost_per_kw, " $/kW", 0))
        m2.metric("Cost per kWh", metric_value(cost_per_kwh, " $/kWh", 0))
        m3.metric("Annual O&M", metric_value(annual_om_m, " $M/y", 1))
        m4.metric("Undiscounted ratio", metric_value(simple_bc, "", 2))
        m5.metric("Cost rank", screening_cost_rank)
        st.caption("The ratio is shown only when both source fields and a positive capital cost are supplied. It is undiscounted and is not an NPV or bankable benefit-cost ratio; add discount rate, escalation, residual value and timing in a proper economic model.")
        st.subheader("Relative cost-rank benchmark")
        st.dataframe(
            pd.DataFrame(SCREENING_COST_CLASS_GUIDANCE, columns=["Rank", "Screening meaning", "How to use"]),
            hide_index=True,
            width="stretch",
        )
        st.caption("Use the rank as a desktop screening benchmark only; convert it to project cost with site-specific quantities, market rates, risk allowances, and contingency.")

        st.dataframe(
            pd.DataFrame(
                [
                    ["Capital cost build-up", "Civil works, tunnels/shafts, dam/reservoir works, electro-mechanical plant, grid connection, owner/environmental costs, contingency"],
                    ["Economic comparison", "Cost per kW, cost per kWh, O&M and sourced discounted economics or a labelled undiscounted proxy; avoid benefit double counting"],
                    ["Environmental monitoring", "Water quality, aquatic ecology, environmental flows, erosion/sediment, biodiversity offsets, noise/dust, cultural heritage commitments"],
                    ["Final design traceability", "Every final MW, GWh, Q, head, diameter, turbine, cost and risk statement traces to a step, equation, source, assumption or sensitivity case"],
                ],
                columns=["Step 10 item", "Student evidence"],
            ),
            hide_index=True,
            width="stretch",
        )
        st.subheader("Traceability matrix")
        trace_df = pd.DataFrame(
            [
                ["Power and energy target", "Step 1", "Project brief / NEM role", "Design power, storage duration"],
                ["Reservoir pair and levels", "Steps 2-3", "QGIS contours / storage curve / operating rule", "Upper/lower HWL and LWL, representative levels, head envelope, active volume"],
                ["Maximum plant discharge", "Steps 4-6", "Power equation, losses, velocity checks", "Q_{total}, Q_u, D, D_u"],
                ["GIS screening evidence", "Steps 1-3, 10", "Source category, cost rank, constraint flag, confidence notes", "Head-to-distance, water-to-rock ratio, storage class, cost rank"],
                ["Cost and economic comparison", "Step 10", "Dated cost/benefit sources with consistent scope", "$/kW, $/kWh, discounted BCR or labelled undiscounted proxy"],
                ["Environmental commitments", "Steps 1-2, 10", "Constraint maps and approvals assumptions", "Monitoring and mitigation plan"],
            ],
            columns=["Final report claim", "Source step", "Evidence source", "Traceable outputs"],
        )
        st.data_editor(trace_df, num_rows="dynamic", width="stretch", key="traceability_matrix_editor")

    with tab_risk:
        st.subheader("Risk register")
        st.caption(
            "Select likelihood and consequence from the dropdowns; the rating is calculated from the 5x5 "
            "matrix below. Every rating must be backed by the evidence column - a rating without evidence "
            "is an opinion, not an assessment. These are not economic weightings; they prioritise mitigation "
            "and final recommendation conditions."
        )
        risk_df = pd.DataFrame(
            [
                ["Hydrology and refill", "Possible", "Major", "Regional rainfall/evaporation records; no site water balance yet", "Water balance, evaporation, seepage and licence study"],
                ["Geotechnical uncertainty", "Likely", "Major", "Desktop geology only; no drilling, mapping or in-situ stress data", "Site investigation, drilling, lab testing, stress measurement"],
                ["Environmental approval", "Possible", "Major", "Constraint mapping only; no field surveys or agency engagement", "EIA, protected species, waterway and offset assessment"],
                ["Desktop-screening limitation", "Possible", "Major", "GIS/desktop candidate only; no field validation yet", "Confirm geology, hydrology, land tenure, heritage, protected areas, and constructability before recommendation"],
                ["NEM dispatch/revenue", "Possible", "Moderate", "Historical price traces; no dispatch or storage-value modelling", "Market modelling, price traces, storage value study"],
                ["Transmission connection", "Possible", "Major", "GIS distance to nearest substation; no connection enquiry lodged", "Connection enquiry, grid studies, network constraint review"],
                ["Construction cost escalation", "Possible", "Major", "Benchmark $/kW ranges only; no quantities or market pricing", "Benchmark cost, quantities, contingencies and procurement strategy"],
                ["Environmental monitoring", "Possible", "Moderate", "Standard commitments assumed; no baseline monitoring data", "Monitoring plan, trigger levels, reporting and adaptive management"],
            ],
            columns=["Risk", "Likelihood", "Consequence", "Evidence basis", "Mitigation / next study"],
        )
        edited_risk = st.data_editor(
            risk_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Likelihood": st.column_config.SelectboxColumn("Likelihood", options=LIKELIHOOD_LEVELS, required=True),
                "Consequence": st.column_config.SelectboxColumn("Consequence", options=CONSEQUENCE_LEVELS, required=True),
                "Evidence basis": st.column_config.TextColumn("Evidence basis", help="What data or study supports the selected likelihood and consequence?"),
            },
        )
        rated = edited_risk.copy()
        rated["Rating"] = [risk_rating(likelihood, consequence) for likelihood, consequence in zip(rated["Likelihood"], rated["Consequence"])]
        st.dataframe(
            rated[["Risk", "Likelihood", "Consequence", "Rating", "Evidence basis", "Mitigation / next study"]],
            hide_index=True,
            width="stretch",
        )
        st.caption("Each group should add project-specific risks and identify the discipline responsible for the next investigation.")

        with st.expander("How to select likelihood and consequence (descriptors and matrix)"):
            st.caption(
                "Teaching adaptation of the qualitative risk-matrix practice used in ISO 31000-style "
                "infrastructure risk management. The descriptors below calibrate the scales for a PHES "
                "development at concept/feasibility stage; state your own descriptors in the report if "
                "you adopt different ones."
            )
            st.markdown("**Likelihood descriptors** (chance the risk materialises over the project development horizon)")
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Rare", "< 5%", "Would require an exceptional combination of circumstances"],
                        ["Unlikely", "5-25%", "Known mechanism but strong evidence it is controlled"],
                        ["Possible", "25-50%", "Plausible with current evidence gaps; typical concept-stage default"],
                        ["Likely", "50-80%", "Evidence gaps directly on the project's critical path (e.g. no drilling yet)"],
                        ["Almost certain", "> 80%", "Mechanism already observed on site or on directly comparable projects"],
                    ],
                    columns=["Level", "Indicative probability", "Interpretation"],
                ),
                hide_index=True,
                width="stretch",
            )
            st.markdown("**Consequence descriptors** (impact if the risk materialises)")
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Insignificant", "< 1% of capital cost", "< 1 month", "No approval implications"],
                        ["Minor", "1-3%", "1-3 months", "Manageable conditions"],
                        ["Moderate", "3-10%", "3-12 months", "Additional studies or redesign of one component"],
                        ["Major", "10-25%", "1-2 years", "Major redesign, approval conditions, or re-consultation"],
                        ["Severe", "> 25%", "> 2 years", "Project viability threatened"],
                    ],
                    columns=["Level", "Indicative cost impact", "Indicative schedule impact", "Approvals / design impact"],
                ),
                hide_index=True,
                width="stretch",
            )
            st.markdown("**Rating matrix** (likelihood x consequence; score bands 1-4 Low, 5-9 Medium, 10-16 High, 17-25 Extreme)")
            matrix = pd.DataFrame(
                [[risk_rating(likelihood, consequence) for consequence in CONSEQUENCE_LEVELS] for likelihood in LIKELIHOOD_LEVELS],
                index=LIKELIHOOD_LEVELS,
                columns=CONSEQUENCE_LEVELS,
            )
            st.dataframe(matrix, width="stretch")
            st.caption(
                "Evidence expectations by rating: High and Extreme risks need a named next investigation with "
                "an owner and timing; Medium risks need monitoring triggers; Low risks need a one-line justification. "
                "As site data arrives (drilling, hydrology, market studies), likelihoods should move - a register "
                "that never changes between submissions signals that no evidence was collected."
            )


def main() -> None:
    st.set_page_config(page_title="Hydropower Teaching App", layout="wide")
    init_state()
    app_style()
    page = sidebar()

    levels = reservoir_levels()
    design_case = hydraulic_snapshot(fnum('design_power_mw'), levels["gross_head"])

    if page == OVERVIEW_PAGE:
        render_overview_page(levels, design_case)
    elif page.startswith("Step "):
        page_workflow_step(page, levels, design_case)
    else:
        page_equations_references(levels, design_case)


if __name__ == "__main__":
    main()
