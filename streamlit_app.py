import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

st.set_page_config(page_title="Valley/Peak Window + BESS Smoothing Optimizer", layout="wide")
st.title("Valley/Peak Window Identification + BESS Optimizer")
st.caption("Finds ONE continuous valley window and ONE continuous peak window from Load & RES. Then smooths net demand as much as possible.")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Time step")
dt_hours = st.sidebar.number_input("Time step (hours)", value=1.0, min_value=0.25, step=0.25)

st.sidebar.header("Smoothing for detection")
roll_n = st.sidebar.slider("Rolling window (points)", 1, 9, 3)

st.sidebar.header("Solar / Non-solar split")
res_gate = st.sidebar.number_input("Solar penetration gate (RES > MW)", value=400.0, min_value=0.0)

st.sidebar.header("Window selection")
# These pick candidate points BEFORE we choose the best contiguous window
valley_pct = st.sidebar.slider("Valley net percentile (low)", 5, 50, 30)
peak_pct = st.sidebar.slider("Peak net percentile (high)", 50, 95, 80)
min_window_pts = st.sidebar.slider("Min window length (points)", 2, 24, 6)

st.sidebar.header("BESS")
E_mwh = st.sidebar.number_input("Energy capacity E (MWh)", value=100.0, min_value=1.0)
Pmax_mw = st.sidebar.number_input("Power limit Pmax (MW)", value=50.0, min_value=0.1)
soc0 = st.sidebar.slider("Initial SOC (%)", 0, 100, 50) / 100.0
soc_min = st.sidebar.slider("SOC min (%)", 0, 100, 10) / 100.0
soc_max = st.sidebar.slider("SOC max (%)", 0, 100, 90) / 100.0
eta_ch = st.sidebar.slider("Charge efficiency", 50, 100, 95) / 100.0
eta_dis = st.sidebar.slider("Discharge efficiency", 50, 100, 95) / 100.0

st.sidebar.header("Optimization (maximize smoothness)")
w_flat = st.sidebar.slider("Flattening weight", 0.1, 50.0, 10.0, 0.1)
w_smooth = st.sidebar.slider("Smoothness weight", 0.0, 50.0, 10.0, 0.1)

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
    st.info("Upload both Load and RES files.")
    st.stop()

load_mw, t = read_profile(load_file, "Load")
res_mw, _ = read_profile(res_file, "RES")

if len(load_mw) != len(res_mw):
    st.error("Load and RES must have the same number of rows.")
    st.stop()

N = len(load_mw)
net = load_mw - res_mw

# -----------------------
# Helper: rolling mean (for detection only)
# -----------------------
def rolling_mean(x, n):
    if n <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(n, center=True, min_periods=1).mean().to_numpy()

net_s = rolling_mean(net, roll_n)
res_s = rolling_mean(res_mw, roll_n)

is_solar = res_s > res_gate
is_nonsolar = ~is_solar

# Candidate points
valley_thr = np.percentile(net_s, valley_pct)
peak_thr = np.percentile(net_s, peak_pct)

valley_candidates = is_solar & (net_s <= valley_thr)
peak_candidates = is_nonsolar & (net_s >= peak_thr)

# -----------------------
# Helper: find best contiguous window
# We pick the "best" window by score:
# - valley: lowest average net_s (and longest)
# - peak: highest average net_s (and longest)
# -----------------------
def contiguous_windows(mask):
    """Return list of (start, end_exclusive) for True runs."""
    windows = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            windows.append((i, j))
            i = j
        else:
            i += 1
    return windows

def best_valley_window(mask, series, min_len):
    wins = [(s, e) for (s, e) in contiguous_windows(mask) if (e - s) >= min_len]
    if not wins:
        return None
    # score: lower mean is better; tie-breaker longer
    best = None
    best_score = None
    for s, e in wins:
        mean_val = float(np.mean(series[s:e]))
        score = (mean_val, -(e - s))  # minimize mean, maximize length
        if best is None or score < best_score:
            best = (s, e)
            best_score = score
    return best

def best_peak_window(mask, series, min_len):
    wins = [(s, e) for (s, e) in contiguous_windows(mask) if (e - s) >= min_len]
    if not wins:
        return None
    # score: higher mean is better; tie-breaker longer
    best = None
    best_score = None
    for s, e in wins:
        mean_val = float(np.mean(series[s:e]))
        score = (-mean_val, -(e - s))  # maximize mean => minimize negative mean
        if best is None or score < best_score:
            best = (s, e)
            best_score = score
    return best

valley_win = best_valley_window(valley_candidates, net_s, min_window_pts)
peak_win = best_peak_window(peak_candidates, net_s, min_window_pts)

valley_mask = np.zeros(N, dtype=bool)
peak_mask = np.zeros(N, dtype=bool)

if valley_win is not None:
    valley_mask[valley_win[0]:valley_win[1]] = True
if peak_win is not None:
    peak_mask[peak_win[0]:peak_win[1]] = True

if valley_win is None:
    st.warning("No VALLEY window found. Try increasing valley percentile or reducing min window length.")
if peak_win is None:
    st.warning("No PEAK window found. Try decreasing peak percentile or reducing min window length.")

# -----------------------
# Optimization: flatten net_after as much as possible
# net_after = net + charge - discharge
# Objective:
#   w_flat * sum((net_after - m)^2)  +  w_smooth * sum((Δnet_after)^2)
# This makes net demand very smooth and flat.
# -----------------------
p_ch = cp.Variable(N, nonneg=True)
p_dis = cp.Variable(N, nonneg=True)
soc = cp.Variable(N + 1)

net_after = net + p_ch - p_dis

m = cp.Variable()  # "flat target" level (chosen by optimizer)

constraints = []
constraints += [p_ch <= Pmax_mw, p_dis <= Pmax_mw]
constraints += [soc[0] == soc0]
constraints += [soc >= soc_min, soc <= soc_max]

for k in range(N):
    constraints += [
        soc[k+1] == soc[k] + (eta_ch * p_ch[k] - (1.0 / eta_dis) * p_dis[k]) * (dt_hours / E_mwh)
    ]

# Charge only in valley window
for k in range(N):
    if not valley_mask[k]:
        constraints += [p_ch[k] == 0]
# Discharge only in peak window
for k in range(N):
    if not peak_mask[k]:
        constraints += [p_dis[k] == 0]

# Objective terms
flat_term = cp.sum_squares(net_after - m)
smooth_term = cp.sum_squares(net_after[1:] - net_after[:-1])

objective = cp.Minimize(w_flat * flat_term + w_smooth * smooth_term)

problem = cp.Problem(objective, constraints)

with st.spinner("Optimizing (flatten + smooth)..."):
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

if valley_win is not None:
    st.success(f"VALLEY window: {valley_win[0]} → {valley_win[1]-1}  (length {valley_win[1]-valley_win[0]} points)")
if peak_win is not None:
    st.success(f"PEAK window: {peak_win[0]} → {peak_win[1]-1}  (length {peak_win[1]-peak_win[0]} points)")

# -----------------------
# Plot
# -----------------------
st.subheader("Net Demand + Identified Valley/Peak Windows")

fig = plt.figure()
plt.plot(net, label="Net before (Load-RES)")
plt.plot(net_after_v, label="Net after (with BESS)")

# Shade valley and peak windows (continuous)
if valley_win is not None:
    plt.axvspan(valley_win[0], valley_win[1]-1, alpha=0.2, label="Valley window")
if peak_win is not None:
    plt.axvspan(peak_win[0], peak_win[1]-1, alpha=0.2, label="Peak window")

plt.xlabel("Time step")
plt.ylabel("MW")
plt.title("Net Demand (Before/After) + Windows")
plt.legend()
st.pyplot(fig)

st.subheader("BESS Schedule")
fig = plt.figure()
plt.plot(p_bess, label="BESS MW (+discharge, -charge)")
plt.axhline(Pmax_mw, linestyle="--")
plt.axhline(-Pmax_mw, linestyle="--")
plt.xlabel("Time step")
plt.ylabel("MW")
plt.title("BESS Power")
plt.legend()
st.pyplot(fig)

st.subheader("SOC (%)")
fig = plt.figure()
plt.plot(soc_v[:-1] * 100.0, label="SOC")
plt.axhline(soc_min * 100.0, linestyle="--")
plt.axhline(soc_max * 100.0, linestyle="--")
plt.xlabel("Time step")
plt.ylabel("%")
plt.title("State of Charge")
plt.legend()
st.pyplot(fig)

# -----------------------
# Table
# -----------------------
st.subheader("Results Table")
out = pd.DataFrame({
    "time": t,
    "load_MW": load_mw,
    "res_MW": res_mw,
    "net_before_MW": net,
    "valley_window": valley_mask.astype(int),
    "peak_window": peak_mask.astype(int),
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
    file_name="bess_valley_peak_windows_results.csv",
    mime="text/csv",
)
