import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Scheduler + Graphs", layout="wide")

st.title("Input → Schedule + Graphs (Single Page App)")

# --- Sidebar inputs ---
st.sidebar.header("Inputs")

n_steps = st.sidebar.number_input("Number of time steps", min_value=24, max_value=1440, value=48, step=1)
dt_hours = st.sidebar.number_input("Step duration (hours)", min_value=0.01, max_value=1.0, value=0.5)

energy_mwh = st.sidebar.number_input("Battery Energy (MWh)", min_value=1.0, value=100.0)
pmax_mw = st.sidebar.number_input("Max Power (MW)", min_value=0.1, value=50.0)
soc0 = st.sidebar.slider("Initial SOC (%)", 0, 100, 50)
soc_min = st.sidebar.slider("Min SOC (%)", 0, 100, 20)
soc_max = st.sidebar.slider("Max SOC (%)", 0, 100, 90)

st.sidebar.subheader("Price signal (optional)")
use_random_price = st.sidebar.checkbox("Use sample price profile", value=True)

# --- Data preparation ---
t = pd.date_range("2026-02-15 00:00", periods=int(n_steps), freq=f"{int(dt_hours*60)}min")

if use_random_price:
    # sample price curve
    base = 30 + 10*np.sin(np.linspace(0, 2*np.pi, len(t)))
    noise = np.random.normal(0, 2, len(t))
    price = np.clip(base + noise, 5, None)
else:
    price = np.ones(len(t)) * 30

df = pd.DataFrame({"time": t, "price": price})

# --- Button to generate schedule ---
st.write("### Generate")
run = st.button("Generate Schedule + Graphs")

if run:
    # Simple heuristic scheduler (replace later with real optimizer)
    # Rule: charge when price is low, discharge when price is high
    p = np.zeros(len(df))
    low_thr = np.quantile(df["price"], 0.30)
    high_thr = np.quantile(df["price"], 0.70)

    soc = np.zeros(len(df))
    soc[0] = soc0/100 * energy_mwh

    soc_min_mwh = soc_min/100 * energy_mwh
    soc_max_mwh = soc_max/100 * energy_mwh

    for i in range(1, len(df)):
        if df["price"].iloc[i] <= low_thr:
            p[i] = -pmax_mw  # charge (negative)
        elif df["price"].iloc[i] >= high_thr:
            p[i] = pmax_mw   # discharge (positive)
        else:
            p[i] = 0

        # Update SOC: soc(t+1)=soc(t) - P*dt (discharge reduces SOC)
        soc[i] = soc[i-1] - p[i]*dt_hours

        # Enforce SOC limits by clipping power if needed
        if soc[i] > soc_max_mwh:
            soc[i] = soc_max_mwh
            p[i] = (soc[i-1] - soc[i]) / dt_hours * (-1)  # adjust
        if soc[i] < soc_min_mwh:
            soc[i] = soc_min_mwh
            p[i] = (soc[i-1] - soc[i]) / dt_hours * (-1)

    df["power_mw"] = p
    df["soc_mwh"] = soc
    df["soc_%"] = 100*df["soc_mwh"]/energy_mwh
    df["energy_mwh_step"] = df["power_mw"]*dt_hours
    df["cashflow"] = df["power_mw"]*dt_hours*df["price"]  # simple (sign shows buy/sell)
    total_value = df["cashflow"].sum()

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Schedule Table")
        st.dataframe(df[["time","price","power_mw","soc_%","cashflow"]], use_container_width=True)

    with c2:
        st.subheader("KPIs")
        st.metric("Total Value (simple)", f"{total_value:,.2f}")
        st.metric("Min SOC %", f"{df['soc_%'].min():.1f}")
        st.metric("Max SOC %", f"{df['soc_%'].max():.1f}")

    st.subheader("Graphs")

    fig1, ax1 = plt.subplots()
    ax1.plot(df["time"], df["price"])
    ax1.set_title("Price")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(df["time"], df["power_mw"])
    ax2.set_title("Battery Power (MW)  (+ discharge, - charge)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("MW")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(df["time"], df["soc_%"])
    ax3.set_title("SOC (%)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("%")
    st.pyplot(fig3)
