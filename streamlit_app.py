import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

st.set_page_config(page_title="BESS Band-Constrained Net Demand Smoothing", layout="wide")
st.title("BESS Optimizer: Keep Net Demand Within Thermal + Reserve Band")
st.caption("NetAfter must satisfy:  Gmin + DownSR  ≤  NetAfter  ≤  Gmax - UpSR  , then smooth as much as possible.")

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.header("Time Step")
dt_hours = st.sidebar.number_input("Time step (hours)", value=1.0, min_value=0.25, step=0.25)

st.sidebar.header("Thermal Limits + Reserves (MW)")
use_time_varying_limits = st.sidebar.checkbox("Upload time-varying thermal limits/reserves (optional)", value=False)

Gmax_const = st.sidebar.number_input("Gmax (Max thermal generation)", value=4800.0, min_value=0.0)
Gmin_const = st.sidebar.number_input("Gmin (Min thermal generation)", value=1860.0, min_value=0.0)
SR_up_const = st.sidebar.number_input("Upward spinning reserve SR_up", value=427.0, min_value=0.0)
SR_dn_const = st.sidebar.number_input("Downward spinning reserve SR_down", value=170.0, min_value=0.0)

st.sidebar.header("BESS Parameters")
E_mwh = st.sidebar.number_input("Energy capacity E (MWh)", value=100.0, min_value=1.0)
Pmax_mw = st.sidebar.number_input("Power limit Pmax (MW)", value=50.0, min_value=0.1)
soc0 = st.sidebar.slider("Initial SOC (%)", 0, 100, 50) / 100.0
soc_min = st.sidebar.slider("SOC min (%)", 0, 100, 10) / 100.0
soc_max = st.sidebar.slider("SOC max (%)", 0, 100, 90) / 100.0
eta_ch = st.sidebar.slider("Charge efficiency", 50, 100, 95) / 100.0
eta_dis = st.sidebar.slider("Discharge efficiency", 50, 100, 95) / 100.0

st.sidebar.header("Smoothing Strength")
w_flat = st.sidebar.slider("Flattening weight", 0.1, 100.0, 20.0, 0.1)
w_smooth = st.sidebar.slider("Smoothness (ramping) weight", 0.0, 200.0, 40.0, 0.5)

st.sidebar.header("Optional")
enforce_end_soc = st.sidebar.checkbox("Force end SOC = start SOC", value=False)

# -----------------------
# Upload Load/RES
# -----------------------
c1, c2 = st.columns(2)
with c1:
    load_file = st.file_uploader("Upload Load CSV (column: MW)", type=["csv"])
with c2:
    res_file = st.file_uploader("Upload RES CSV (column: MW)", type=["csv"])

limits_file = None
if use_time_varying_limits:
    limits_file = st.file_uploader(
        "Upload limits CSV (columns: Gmax, Gmin, SR_up, SR_down) optional time column",
        type=["csv"]
    )

def read_profile(file, name):
    df = pd.read_csv(file)
    if "MW" not in df.columns:
        raise ValueError(f"{name} CSV must include a column named 'MW'")
    mw = df["MW"].astype(float).to_numpy()
    t = df["time"].astype(str).to_numpy() if "time" in df.columns else np.array([f"t{i}" for i in range(len(mw))])
    return mw, t

if load_file is None or res_file is None:
    st.info("Upload Load and RES CSVs to continue.")
    st.stop()

load_mw, t = read_profile(load_file, "Load")
res_mw, _ = read_profile(res_file, "RES")

if len(load_mw) != len(res_mw):
    st.error("Load and RES must have the same number of rows.")
    st.stop()

N = len(load_mw)
net_before = load_mw - res_mw

# -----------------------
# Build thermal band arrays (constant or time-varying)
# -----------------------
if limits_file is None:
    Gmax = np.full(N, Gmax_const, dtype=float)
    Gmin = np.full(N, Gmin_const, dtype=float)
    SR_up = np.full(N, SR_up_const, dtype=float)
    SR_dn = np.full(N, SR_dn_const, dtype=float)
else:
    dfL = pd.read_csv(limits_file)
    for col in ["Gmax", "Gmin", "SR_up", "SR_down"]:
        if col not in dfL.columns:
            st.error("Limits CSV must include columns: Gmax, Gmin, SR_up, SR_down")
            st.stop()
    if len(dfL) != N:
        st.error(f"Limits CSV rows must match Load/RES rows. Limits={len(dfL)}, data={N}")
        st.stop()
    Gmax = dfL["Gmax"].astype(float).to_numpy()
    Gmin = dfL["Gmin"].astype(float).to_numpy()
    SR_up = dfL["SR_up"].astype(float).to_numpy()
    SR_dn = dfL["SR_down"].astype(float).to_numpy()

upper_band = Gmax - SR_up
lower_band = Gmin + SR_dn

# Feasibility check
if np.any(lower_band > upper_band):
    st.error("Infeasible band: some hours have (Gmin+DownSR) > (Gmax-SR_up). Fix inputs.")
    st.stop()

# -----------------------
# Optimization
# -----------------------
p_ch = cp.Variable(N, nonneg=True)
p_dis = cp.Variable(N, nonneg=True)
soc = cp.Variable(N + 1)

net_after = net_before + p_ch - p_dis

constraints = []
constraints += [p_ch <= Pmax_mw, p_dis <= Pmax_mw]
constraints += [soc[0] == soc0]
constraints += [soc >= soc_min, soc <= soc_max]

for k in range(N):
    constraints += [
        soc[k+1] == soc[k] + (eta_ch * p_ch[k] - (1.0 / eta_dis) * p_dis[k]) * (dt_hours / E_mwh)
    ]

if enforce_end_soc:
    constraints += [soc[N] == soc0]

# HARD security band constraints
constraints += [net_after <= upper_band]
constraints += [net_after >= lower_band]

# "Smooth as much as possible"
m = cp.Variable()  # chosen flat level
flat_term = cp.sum_squares(net_after - m)
smooth_term = cp.sum_squares(net_after[1:] - net_after[:-1])

objective = cp.Minimize(w_flat * flat_term + w_smooth * smooth_term)
problem = cp.Problem(objective, constraints)

with st.spinner("Optimizing..."):
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
k1.metric("Net peak BEFORE (MW)", f"{np.max(net_before):.1f}")
k2.metric("Net peak AFTER (MW)", f"{np.max(net_after_v):.1f}")
k3.metric("Net min BEFORE (MW)", f"{np.min(net_before):.1f}")
k4.metric("Net min AFTER (MW)", f"{np.min(net_after_v):.1f}")

st.info(f"Band enforced: lower = Gmin+DownSR, upper = Gmax-SR_up")

# -----------------------
# Plot: Net + Band
# -----------------------
st.subheader("Net Demand Before/After with Thermal+Reserve Band")

fig = plt.figure()
plt.plot(net_before, label="Net before (Load-RES)")
plt.plot(net_after_v, label="Net after (with BESS)")
plt.plot(upper_band, label="Upper limit = Gmax - SR_up")
plt.plot(lower_band, label="Lower limit = Gmin + SR_down")
plt.xlabel("Time step")
plt.ylabel("MW")
plt.title("Band-Constrained Net Smoothing")
plt.legend()
st.pyplot(fig)

st.subheader("BESS Power (+discharge, -charge)")
fig = plt.figure()
plt.plot(p_bess, label="BESS MW (+dis, -ch)")
plt.axhline(Pmax_mw, linestyle="--")
plt.axhline(-Pmax_mw, linestyle="--")
plt.xlabel("Time step")
plt.ylabel("MW")
plt.legend()
st.pyplot(fig)

st.subheader("SOC (%)")
fig = plt.figure()
plt.plot(soc_v[:-1] * 100.0, label="SOC")
plt.axhline(soc_min * 100.0, linestyle="--")
plt.axhline(soc_max * 100.0, linestyle="--")
plt.xlabel("Time step")
plt.ylabel("%")
plt.legend()
st.pyplot(fig)

# -----------------------
# Table + download
# -----------------------
st.subheader("Results Table")
out = pd.DataFrame({
    "time": t,
    "load_MW": load_mw,
    "res_MW": res_mw,
    "net_before_MW": net_before,
    "upper_limit_Gmax_minus_SRup": upper_band,
    "lower_limit_Gmin_plus_SRdn": lower_band,
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
    file_name="bess_band_smoothing_results.csv",
    mime="text/csv",
)
