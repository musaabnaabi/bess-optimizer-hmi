import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import math

st.set_page_config(page_title="Net Demand + BESS Optimizer (48h)", layout="wide")
st.title("Net Demand (Load - RES) + BESS Optimization (48h)")
st.caption("Auto-identifies valley (solar/high RES, low net demand) and peak (low RES, high net demand). Uses 4-hour block schedule (no switching inside block).")

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.header("Time & Data")
dt_hours = st.sidebar.number_input("Time step (hours)", value=1.0, min_value=0.05, step=0.25)
expected_points = st.sidebar.number_input("Expected number of points (48h)", value=48, min_value=2, step=1)

st.sidebar.header("BESS Parameters")
E_mwh = st.sidebar.number_input("Energy capacity E (MWh)", value=100.0, min_value=1.0)
Pmax_mw = st.sidebar.number_input("Power limit Pmax (MW)", value=50.0, min_value=0.1)
soc0 = st.sidebar.slider("Initial SOC (%)", 0, 100, 50) / 100.0
soc_min = st.sidebar.slider("SOC min (%)", 0, 100, 10) / 100.0
soc_max = st.sidebar.slider("SOC max (%)", 0, 100, 90) / 100.0
eta_ch = st.sidebar.slider("Charge efficiency", 50, 100, 95) / 100.0
eta_dis = st.sidebar.slider("Discharge efficiency", 50, 100, 95) / 100.0

st.sidebar.header("No Switching Constraint")
block_hours = st.sidebar.number_input("Minimum constant duration (hours)", value=4.0, min_value=0.5, step=0.5)

st.sidebar.header("Auto Valley / Peak Identification")
# Gate by RES to represent "solar penetration" vs "non-solar"
res_solar_gate = st.sidebar.number_input("Solar penetration gate (RES > MW)", value=400.0, min_value=0.0)

# Percentile rules on net demand
valley_percentile = st.sidebar.slider("Valley net-demand percentile (low)", 1, 50, 25)  # e.g. 25th percentile
peak_percentile = st.sidebar.slider("Peak net-demand percentile (high)", 50, 99, 80)    # e.g. 80th percentile

st.sidebar.header("Optimization Tuning")
lam_smooth = st.sidebar.slider("Smoothness weight (lambda)", 0.0, 10.0, 0.2, 0.1)
w_range = st.sidebar.slider("Peak-Valley weight", 0.0, 10.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.write("Rule summary:")
st.sidebar.write("- Valley blocks: RES high + net demand low -> CHARGE allowed")
st.sidebar.write("- Peak blocks: RES low + net demand high -> DISCHARGE allowed")
st.sidebar.write("- 4-hour blocks: constant power inside each block")

# -----------------------
# Upload data
# -----------------------
colA, colB = st.columns(2)
with colA:
    load_file = st.file_uploader("Upload Load profile CSV (column: MW)", type=["csv"])
with colB:
    res_file = st.file_uploader("Upload RES profile CSV (column: MW)", type=["csv"])

def read_profile(file, name):
    df = pd.read_csv(file)
    if "MW" not in df.columns:
        raise ValueError(f"{name} CSV must include a column named 'MW'")
    mw = df["MW"].astype(float).to_numpy()
    t = df["time"].astype(str).to_numpy() if "time" in df.columns else np.array([f"t{i}" for i in range(len(mw))])
    return mw, t

if (load_file is None) or (res_file is None):
    st.info("Upload both Load and RES CSV files to continue.")
    st.stop()

try:
    load_mw, t_labels = read_profile(load_file, "Load")
    res_mw, _ = read_profile(res_file, "RES")
except Exception as e:
    st.error(str(e))
    st.stop()

if len(load_mw) != len(res_mw):
    st.error(f"Load and RES must have the same number of rows. Load={len(load_mw)}, RES={len(res_mw)}")
    st.stop()

N = len(load_mw)
if expected_points and N != int(expected_points):
    st.warning(f"Expected {int(expected_points)} points, but uploaded N={N}. Continuing anyway.")

net_before = load_mw - res_mw

# -----------------------
# Identify valley & peak hours (per timestep)
# -----------------------
valley_thr = np.percentile(net_before, valley_percentile)
peak_thr = np.percentile(net_before, peak_percentile)

# “Solar penetration” indicator
is_solar = res_mw > res_solar_gate
is_non_solar = ~is_solar

# Valley hours: solar penetration + net demand in low tail
valley_mask = is_solar & (net_before <= valley_thr)

# Peak hours: non-solar + net demand in high tail
peak_mask = is_non_solar & (net_before >= peak_thr)

# -----------------------
# Build blocks (4-hour blocks)
# -----------------------
steps_per_block = max(1, int(round(block_hours / dt_hours)))
B = int(math.ceil(N / steps_per_block))
N_pad = B * steps_per_b*_
