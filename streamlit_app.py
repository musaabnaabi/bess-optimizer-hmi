import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

st.set_page_config(page_title="Net Demand + BESS Optimizer (48h)", layout="wide")
st.title("Net Demand (Load - RES) + BESS Optimization (Valley Fill & Peak Shave)")
st.caption("Rule added: Charging is allowed only when RES > threshold (default 400 MW).")

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

st.sidebar.header("Optimization Tuning")
lam_smooth = st.sidebar.slider("Smoothness weight (lambda)", 0.0, 10.0, 0.5, 0.1)
w_range = st.sidebar.slider("Peak-Valley weight", 0.0, 10.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.write("Sign convention:")
st.sidebar.write("- BESS discharge reduces net demand")
st.sidebar.write("- BESS charge increases net demand")

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

    if "time" in df.columns:
        t = df["time"].astype(str).to_numpy()
    else:
        t = np.array([f"t{i}" for i in range(len(mw))])

    return mw, t

if (load_file is None) or (res_file is None):
    st.info("Upload both Load and RES CSV files to continue.")
    st.stop()

try:
    load_mw, t_labels = read_profile(load_file, "Load")
    res_mw, t_labels2 = read_profile(res_file, "RES")
except Exception as e:
    st.error(str(e))
    st.stop()

if len(load_mw) != len(res_mw):
    st.error(f"Load and RES must have the same number of rows. Load={len(load_mw)}, RES={len(res_mw)}")
    st.stop()

N = len(load_mw)
if expected_points and N != int(expected_points):
    st.warning(f"You set expected points = {int(expected_points)}, but uploaded data has N = {N}. Continuing anyway.")

# Net demand before BESS
net_before = load_mw - res_mw

# -----------------------
# Optimization (convex):
# minimize (peak - valley) + lambda*smoothness
# with SOC and power constraints
# and rule: charge only if RES > threshold
# -----------------------
p_ch = cp.Variable(N, nonneg=True)   # MW charging (adds to demand)
p_dis = cp.Variable(N, nonneg=True)  # MW discharging (reduces demand)
soc = cp.Variable(N + 1)             # SOC fraction [0..1]

net_after = net_before + p_ch - p_dis

peak = cp.Variable()
valley = cp.Variable()

constraints = []

# Power limits
constraints += [p_ch <= Pmax_mw]
constraints += [p_dis <= Pmax_mw]

# SOC limits + initial SOC
constraints += [soc[0] == soc0]
constraints += [soc >= soc_min, soc <= soc_max]

# SOC dynamics:
# SOC[k+1] = SOC[k] + (eta_ch*p_ch - (1/eta_dis)*p_dis) * dt / E
for k in range(N):
    constraints += [
        soc[k+1] == soc[k] + (eta_ch * p_ch[k] - (1.0 / eta_dis) * p_dis[k]) * (dt_hours / E_mwh)
    ]

# Peak-valley envelope
constraints += [net_after <= peak]
constraints += [net_after >= valley]

# ---- Charging rule: charging only allowed when RES > threshold ----
high_res_mask = (res_mw > res_charge_threshold)  # boolean array
for k in range(N):
    if not high_res_mask[k]:
        constraints += [p_ch[k] == 0]

# Smoothness term (avoid switching)
smooth = cp.sum_squares(net_after[1:] - net_after[:-1])

# Objective: flatten net demand (range) + smoothness
objective = cp.Minimize(w_range * (peak - valley) + lam_smooth * smooth)

problem = cp.Problem(objective, constraints)

with st.spinner("Running BESS optimization..."):
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        problem.solve(solver=cp.ECOS, verbose=False)

if problem.status not in ["optimal", "optimal_inaccurate"]:
    st.error(f"Optimization failed. Status: {problem.status}")
    st.stop()

p_ch_v = np.array(p_ch.value).flatten()
p_dis_v = np.array(p_dis.value).flatten()
soc_v = np.array(soc.value).flatten()
net_after_v = np.array(net_after.value).flatten()

# Signed BESS power (positive discharge, negative charge)
p_bess = p_dis_v - p_ch_v

# -----------------------
# KPIs
# -----------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Peak before (MW)", f"{np.max(net_before):.1f}")
kpi2.metric("Peak after (MW)", f"{np.max(net_after_v):.1f}")
kpi3.metric("Valley before (MW)", f"{np.min(net_before):.1f}")
kpi4.metric("Valley after (MW)", f"{np.min(net_after_v):.1f}")

# Quick check of rule
viol = np.sum((res_mw <= res_charge_threshold) & (p_ch_v > 1e-6))
if viol == 0:
    st.success(f"Charging rule satisfied: p_ch=0 whenever RES ≤ {res_charge_threshold:.1f} MW.")
else:
    st.warning(f"Charging rule violations found: {viol} timesteps (should be 0).")

# -----------------------
# Charts
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
    plt.title("BESS Schedule")
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
# Data table + download
# -----------------------
st.subheader("Data Table (Download)")

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
