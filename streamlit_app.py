import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

st.set_page_config(page_title="Valley/Peak Detector + BESS Optimizer", layout="wide")
st.title("Valley/Peak Detector + BESS Optimizer (from Load & RES)")
st.caption("Identifies valley/peak from Net Demand = Load - RES, then charges in valley and discharges in peak.")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Time Step")
dt_hours = st.sidebar.number_input("Time step (hours)", value=1.0, min_value=0.25, step=0.25)

st.sidebar.header("Valley/Peak Identification")
valley_pct = st.sidebar.slider("Valley percentile (low net demand)", 5, 50, 25)
peak_pct = st.sidebar.slider("Peak percentile (high net demand)", 50, 95, 80)

res_gate_mode = st.sidebar.selectbox("RES gate mode", ["Median gate", "Fixed threshold"])
if res_gate_mode == "Fixed threshold":
    res_gate_value = st.sidebar.number_input("RES threshold (MW)", value=400.0, min_value=0.0)
else:
    res_gate_value = None

st.sidebar.header("BESS Parameters")
E_mwh = st.sidebar.number_input("Energy capacity E (MWh)", value=100.0, min_value=1.0)
Pmax_mw = st.sidebar.number_input("Power limit Pmax (MW)", value=50.0, min_value=0.1)
soc0 = st.sidebar.slider("Initial SOC (%)", 0, 100, 50) / 100.0
soc_min = st.sidebar.slider("SOC min (%)", 0, 100, 10) / 100.0
soc_max = st.sidebar.slider("SOC max (%)", 0, 100, 90) / 100.0
eta_ch = st.sidebar.slider("Charge efficiency", 50, 100, 95) / 100.0
eta_dis = st.sidebar.slider("Discharge efficiency", 50, 100, 95) / 100.0

st.sidebar.header("Optimization")
w_range = st.sidebar.slider("Flattening weight (peak-valley)", 0.1, 10.0, 1.0, 0.1)
lam_smooth = st.sidebar.slider("Smoothness weight", 0.0, 10.0, 0.2, 0.1)

# -----------------------
# Upload
# -----------------------
col1, col2 = st.columns(2)
with col1:
    load_file = st.file_uploader("Upload Load CSV (column: MW)", type=["csv"])
with col2:
    res_file = st.file_uploader("Upload RES CSV (column: MW)", type=["csv"])

def read_profile(file, name):
    df = pd.read_csv(file)
    if "MW" not in df.columns:
        raise ValueError(f"{name} CSV must include a column named 'MW'")
    mw = df["MW"].astype(float).to_numpy()
    t = df["time"].astype(str).to_numpy() if "time" in df.columns else np.array([f"t{i}" for i in range(len(mw))])
    return mw, t

if load_file is None or res_file is None:
    st.info("Upload both Load and RES to continue.")
    st.stop()

load_mw, t = read_profile(load_file, "Load")
res_mw, _ = read_profile(res_file, "RES")

if len(load_mw) != len(res_mw):
    st.error("Load and RES must have the same number of rows.")
    st.stop()

N = len(load_mw)
net = load_mw - res_mw

# -----------------------
# Identify valley / peak
# -----------------------
valley_thr = np.percentile(net, valley_pct)
peak_thr = np.percentile(net, peak_pct)

if res_gate_mode == "Median gate":
    res_hi = np.median(res_mw)
    res_lo = np.median(res_mw)
    valley_mask = (net <= valley_thr) & (res_mw >= res_hi)
    peak_mask = (net >= peak_thr) & (res_mw <= res_lo)
    res_gate_text = f"Median RES gate: high>= {res_hi:.1f}, low<= {res_lo:.1f}"
else:
    valley_mask = (net <= valley_thr) & (res_mw > res_gate_value)
    peak_mask = (net >= peak_thr) & (res_mw <= res_gate_value)
    res_gate_text = f"Fixed RES gate: valley RES>{res_gate_value:.1f}, peak RES<={res_gate_value:.1f}"

# Ensure we have at least some valley/peak points
if np.sum(valley_mask) == 0:
    st.warning("No valley hours detected with current settings. Try increasing valley percentile or lowering RES gate.")
if np.sum(peak_mask) == 0:
    st.warning("No peak hours detected with current settings. Try decreasing peak percentile or increasing RES gate.")

# -----------------------
# BESS optimization
# -----------------------
p_ch = cp.Variable(N, nonneg=True)
p_dis = cp.Variable(N, nonneg=True)
soc = cp.Variable(N + 1)

net_after = net + p_ch - p_dis

peak_var = cp.Variable()
valley_var = cp.Variable()

constraints = []
constraints += [p_ch <= Pmax_mw, p_dis <= Pmax_mw]
constraints += [soc[0] == soc0]
constraints += [soc >= soc_min, soc <= soc_max]

for k in range(N):
    constraints += [
        soc[k+1] == soc[k] + (eta_ch * p_ch[k] - (1.0 / eta_dis) * p_dis[k]) * (dt_hours / E_mwh)
    ]

# Allow charge only in valley, discharge only in peak
for k in range(N):
    if not valley_mask[k]:
        constraints += [p_ch[k] == 0]
    if not peak_mask[k]:
        constraints += [p_dis[k] == 0]

constraints += [net_after <= peak_var, net_after >= valley_var]

smooth = cp.sum_squares(net_after[1:] - net_after[:-1])
objective = cp.Minimize(w_range * (peak_var - valley_var) + lam_smooth * smooth)

problem = cp.Problem(objective, constraints)
try:
    problem.solve(solver=cp.OSQP, verbose=False)
except Exception:
    problem.solve(solver=cp.ECOS, verbose=False)

if problem.status not in ["optimal", "optimal_inaccurate"]:
    st.error(f"Optimization failed: {problem.status}")
    st.stop()

p_ch_v = np.array(p_ch.value).flatten()
p_dis_v = np.array(p_dis.value).flatten()
soc_v = np.array(soc.value).flatten()
net_after_v = np.array(net_after.value).flatten()
p_bess = p_dis_v - p_ch_v

# -----------------------
# KPIs
# -----------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Peak before (MW)", f"{np.max(net):.1f}")
k2.metric("Peak after (MW)", f"{np.max(net_after_v):.1f}")
k3.metric("Valley before (MW)", f"{np.min(net):.1f}")
k4.metric("Valley after (MW)", f"{np.min(net_after_v):.1f}")

st.info(f"Net valley threshold = {valley_thr:.1f} (P{valley_pct}) | Net peak threshold = {peak_thr:.1f} (P{peak_pct}) | {res_gate_text}")

# -----------------------
# Plots
# -----------------------
st.subheader("Charts")
c1, c2 = st.columns(2)

with c1:
    fig = plt.figure()
    plt.plot(net, label="Net before (Load-RES)")
    plt.plot(net_after_v, label="Net after (with BESS)")
    plt.scatter(np.where(valley_mask)[0], net[valley_mask], s=15, label="Valley hours", marker="o")
    plt.scatter(np.where(peak_mask)[0], net[peak_mask], s=15, label="Peak hours", marker="x")
    plt.xlabel("Time step")
    plt.ylabel("MW")
    plt.title("Net Demand + Identified Valley/Peak")
    plt.legend()
    st.pyplot(fig)

with c2:
    fig = plt.figure()
    plt.plot(p_bess, label="BESS MW (+discharge, -charge)")
    plt.axhline(Pmax_mw, linestyle="--")
    plt.axhline(-Pmax_mw, linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("MW")
    plt.title("BESS Schedule")
    plt.legend()
    st.pyplot(fig)

st.subheader("SOC (%)")
fig = plt.figure()
plt.plot(soc_v[:-1] * 100.0, label="SOC")
plt.axhline(soc_min * 100.0, linestyle="--")
plt.axhline(soc_max * 100.0, linestyle="--")
plt.xlabel("Time step")
plt.ylabel("%")
plt.title("SOC Profile")
plt.legend()
st.pyplot(fig)

# -----------------------
# Output table
# -----------------------
st.subheader("Results Table")
out = pd.DataFrame({
    "time": t,
    "load_MW": load_mw,
    "res_MW": res_mw,
    "net_before_MW": net,
    "is_valley": valley_mask.astype(int),
    "is_peak": peak_mask.astype(int),
    "bess_charge_MW": p_ch_v,
    "bess_discharge_MW": p_dis_v,
    "bess_MW_(+dis,-ch)": p_bess,
    "net_after_MW": net_after_v,
    "soc_%": soc_v[:-1] * 100.0
})
st.dataframe(out, use_container_width=True)

st.download_button(
    "Download results CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="bess_valley_peak_results.csv",
    mime="text/csv",
)
