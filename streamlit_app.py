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
N_pad = B * steps_per_block
pad_len = N_pad - N

def pad_last(x):
    return np.concatenate([x, np.repeat(x[-1], pad_len)]) if pad_len > 0 else x

load_pad = pad_last(load_mw)
res_pad = pad_last(res_mw)
net_before_pad = pad_last(net_before)
valley_pad = pad_last(valley_mask.astype(int)).astype(bool)
peak_pad = pad_last(peak_mask.astype(int)).astype(bool)

# Decide valley/peak per block:
# Operationally safer to require "most of the block" qualifies.
min_share = 0.6  # 60% of timesteps in the block must be valley/peak
charge_block = np.zeros(B, dtype=bool)
disch_block = np.zeros(B, dtype=bool)

for b in range(B):
    s = b * steps_per_block
    e = s + steps_per_block
    charge_block[b] = (np.mean(valley_pad[s:e]) >= min_share)
    disch_block[b] = (np.mean(peak_pad[s:e]) >= min_share)

# If a block is neither valley nor peak, allow both to be 0 (idle). We don’t force action.

# -----------------------
# Optimization variables (per block)
# -----------------------
p_ch_b = cp.Variable(B, nonneg=True)   # MW charge constant per block
p_dis_b = cp.Variable(B, nonneg=True)  # MW discharge constant per block
soc = cp.Variable(N_pad + 1)

# Expand block power to per-step vectors
p_ch = cp.hstack([cp.hstack([p_ch_b[b]] * steps_per_block) for b in range(B)])
p_dis = cp.hstack([cp.hstack([p_dis_b[b]] * steps_per_block) for b in range(B)])

net_after = net_before_pad + p_ch - p_dis

peak = cp.Variable()
valley = cp.Variable()

constraints = []

# Power limits
constraints += [p_ch_b <= Pmax_mw]
constraints += [p_dis_b <= Pmax_mw]

# SOC bounds + initial
constraints += [soc[0] == soc0]
constraints += [soc >= soc_min, soc <= soc_max]

# SOC dynamics
for k in range(N_pad):
    constraints += [
        soc[k+1] == soc[k] + (eta_ch * p_ch[k] - (1.0 / eta_dis) * p_dis[k]) * (dt_hours / E_mwh)
    ]

# Peak/valley envelope
constraints += [net_after <= peak]
constraints += [net_after >= valley]

# Enforce valley/peak operational logic at block level:
# - If block is NOT valley -> no charging
# - If block is NOT peak   -> no discharging
for b in range(B):
    if not charge_block[b]:
        constraints += [p_ch_b[b] == 0]
    if not disch_block[b]:
        constraints += [p_dis_b[b] == 0]

# Smoothness (helps reduce sharp boundary jumps)
smooth = cp.sum_squares(net_after[1:] - net_after[:-1])
objective = cp.Minimize(w_range * (peak - valley) + lam_smooth * smooth)

problem = cp.Problem(objective, constraints)

with st.spinner("Running optimization (valley/peak blocks + 4-hour no-switching)..."):
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        problem.solve(solver=cp.ECOS, verbose=False)

if problem.status not in ["optimal", "optimal_inaccurate"]:
    st.error(f"Optimization failed. Status: {problem.status}")
    st.stop()

# Extract and unpad
p_ch_v = np.array(p_ch.value).flatten()[:N]
p_dis_v = np.array(p_dis.value).flatten()[:N]
soc_v = np.array(soc.value).flatten()[:N+1]
net_after_v = np.array(net_after.value).flatten()[:N]

p_bess = p_dis_v - p_ch_v  # +discharge, -charge

# -----------------------
# KPIs
# -----------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Peak before (MW)", f"{np.max(net_before):.1f}")
kpi2.metric("Peak after (MW)", f"{np.max(net_after_v):.1f}")
kpi3.metric("Valley before (MW)", f"{np.min(net_before):.1f}")
kpi4.metric("Valley after (MW)", f"{np.min(net_after_v):.1f}")

st.info(
    f"Valley threshold (net demand) = {valley_thr:.1f} MW (<= {valley_percentile}th percentile) | "
    f"Peak threshold (net demand) = {peak_thr:.1f} MW (>= {peak_percentile}th percentile) | "
    f"Solar gate: RES > {res_solar_gate:.1f} MW"
)

# -----------------------
# Charts
# -----------------------
st.subheader("Charts")

c1, c2 = st.columns(2)

with c1:
    fig = plt.figure()
    plt.plot(net_before, label="Net demand BEFORE")
    plt.plot(net_after_v, label="Net demand AFTER (with BESS)")
    plt.scatter(np.where(valley_mask)[0], net_before[valley_mask], s=12, label="Valley hours", marker="o")
    plt.scatter(np.where(peak_mask)[0], net_before[peak_mask], s=12, label="Peak hours", marker="x")
    plt.xlabel("Time step")
    plt.ylabel("MW")
    plt.title("Net Demand + Identified Valley/Peak Hours")
    plt.legend()
    st.pyplot(fig)

with c2:
    fig = plt.figure()
    plt.plot(p_bess, label="BESS Power (+discharge, -charge)")
    plt.axhline(Pmax_mw, linestyle="--")
    plt.axhline(-Pmax_mw, linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("MW")
    plt.title("BESS Schedule (4-hour blocks)")
    plt.legend()
    st.pyplot(fig)

st.subheader("SOC Profile")
fig = plt.figure()
plt.plot(soc_v[:-1] * 100.0, label="SOC (%)")
plt.axhline(soc_min * 100.0, linestyle="--")
plt.axhline(soc_max * 100.0, linestyle="--")
plt.xlabel("Time step")
plt.ylabel("%")
plt.title("State of Charge")
plt.legend()
st.pyplot(fig)

# -----------------------
# Results table + download
# -----------------------
st.subheader("Results Table (Download)")

out = pd.DataFrame({
    "time": t_labels if len(t_labels) == N else [f"t{i}" for i in range(N)],
    "load_MW": load_mw,
    "res_MW": res_mw,
    "net_before_MW": net_before,
    "is_valley": valley_mask.astype(int),
    "is_peak": peak_mask.astype(int),
    "bess_charge_MW": p_ch_v,
    "bess_discharge_MW": p_dis_v,
    "bess_MW_(+dis,-ch)": p_bess,
    "net_after_MW": net_after_v,
    "soc_%": soc_v[:-1] * 100.0
})

st.dataframe(out, use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv_bytes, file_name="bess_results.csv", mime="text/csv")
