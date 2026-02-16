import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import math

st.set_page_config(page_title="Net Demand + BESS Optimizer (48h)", layout="wide")
st.title("Net Demand (Load - RES) + BESS Optimization (48h)")
st.caption("Features: Upload Load/RES, Net demand before/after, Peak shave + Valley fill, Charge only when RES>threshold, 4-hour block schedule (no switching within 4 hours).")

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

st.sidebar.header("Charging Rule")
res_charge_threshold = st.sidebar.number_input("Charge only if RES > (MW)", value=400.0, min_value=0.0)

st.sidebar.header("No Switching Constraint")
block_hours = st.sidebar.number_input("Minimum constant duration (hours)", value=4.0, min_value=0.5, step=0.5)

st.sidebar.header("Optimization Tuning")
lam_smooth = st.sidebar.slider("Smoothness weight (lambda)", 0.0, 10.0, 0.2, 0.1)
w_range = st.sidebar.slider("Peak-Valley weight", 0.0, 10.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.write("Sign convention:")
st.sidebar.write("- Discharge reduces net demand")
st.sidebar.write("- Charge increases net demand")
st.sidebar.write("Block schedule: power is constant inside each block.")

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
# Build 4-hour blocks (or user-defined)
# -----------------------
steps_per_block = max(1, int(round(block_hours / dt_hours)))
B = int(math.ceil(N / steps_per_block))
N_pad = B * steps_per_block
pad_len = N_pad - N

# Pad last values (repeat last) so blocks fit exactly
if pad_len > 0:
    load_pad = np.concatenate([load_mw, np.repeat(load_mw[-1], pad_len)])
    res_pad = np.concatenate([res_mw, np.repeat(res_mw[-1], pad_len)])
    net_before_pad = load_pad - res_pad
    t_pad = np.concatenate([t_labels, np.array([f"pad{i+1}" for i in range(pad_len)])])
else:
    load_pad, res_pad, net_before_pad, t_pad = load_mw, res_mw, net_before, t_labels

# For the RES charging rule, we restrict charging if ANY step in the block is <= threshold
# (more strict, operationally safer)
res_block_allow = np.ones(B, dtype=bool)
for b in range(B):
    s = b * steps_per_block
    e = s + steps_per_block
    res_block_allow[b] = np.all(res_pad[s:e] > res_charge_threshold)

# -----------------------
# Optimization variables (per block)
# -----------------------
p_ch_b = cp.Variable(B, nonneg=True)   # MW, constant within block
p_dis_b = cp.Variable(B, nonneg=True)  # MW, constant within block
soc = cp.Variable(N_pad + 1)

# Expand block power to per-step arrays
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

# Charging rule at BLOCK level
for b in range(B):
    if not res_block_allow[b]:
        constraints += [p_ch_b[b] == 0]

# Smoothness (still helps reduce oscillation at boundaries)
smooth = cp.sum_squares(net_after[1:] - net_after[:-1])

objective = cp.Minimize(w_range * (peak - valley) + lam_smooth * smooth)
problem = cp.Problem(objective, constraints)

with st.spinner("Running BESS optimization with 4-hour block constraint..."):
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        problem.solve(solver=cp.ECOS, verbose=False)

if problem.status not in ["optimal", "optimal_inaccurate"]:
    st.error(f"Optimization failed. Status: {problem.status}")
    st.stop()

# Extract and unpad back to N
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

st.info(f"Block size = {steps_per_block} steps ≈ {steps_per_block*dt_hours:.2f} hours (changes only at block boundaries).")

# Verify “no switching within block” (diagnostic)
block_ok = True
for b in range(B):
    s = b * steps_per_block
    e = min(s + steps_per_block, N)
    if e - s <= 1:
        continue
    if (np.max(p_bess[s:e]) - np.min(p_bess[s:e])) > 1e-6:
        block_ok = False
        break

if block_ok:
    st.success("No-switching constraint satisfied: BESS power is constant inside each block.")
else:
    st.warning("Unexpected variation detected inside a block (should be constant).")

# -----------------------
# Plots
# -----------------------
st.subheader("Charts")

c1, c2 = st.columns(2)

with c1:
    fig = plt.figure()
    plt.plot(net_before, label="Net demand BEFORE (Load-RES)")
    plt.plot(net_after_v, label="Net demand AFTER (with BESS)")
    plt.xlabel("Time step")
    plt.ylabel("MW")
    plt.title("Net Demand Before vs After BESS")
    plt.legend()
    st.pyplot(fig)

with c2:
    fig = plt.figure()
    plt.plot(p_bess, label="BESS Power (MW) (+discharge, -charge)")
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
# Table + Download
# -----------------------
st.subheader("Results Table (Download)")

out = pd.DataFrame({
    "time": t_labels if len(t_labels) == N else [f"t{i}" for i in range(N)],
    "load_MW": load_mw,
    "res_MW": res_mw,
    "net_before_MW": net_before,
    "bess_charge_MW": p_ch_v,
    "bess_discharge_MW": p_dis_v,
    "bess_MW_(+dis,-ch)": p_bess,
    "net_after_MW": net_after_v,
    "soc_%": soc_v[:-1] * 100.0
})

st.dataframe(out, use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv_bytes, file_name="bess_results.csv", mime="text/csv")
