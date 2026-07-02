
import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint

# 1. WEB PAGE GRAPHICAL CONFIGURATION
st.set_page_config(page_title="Global Neuro-Climate Simulator", layout="wide")

st.title(" Universal In-Silico Neuroinflammation Simulator")
st.markdown("""
**Author: Tasnim** | Developed as a scalable predictive framework linking satellite-derived climate stress to blood-brain barrier degradation and microglial kinetics.
""")

st.sidebar.header(" Global Coordinates & Timeline")

#  2. USER CONTROLS (SLIDERS & DROP-DOWNS)
lat = st.sidebar.number_input("Latitude", value=50.8503, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=4.3517, format="%.4f")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2025-07-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-07-20"))

st.sidebar.header(" Biological Sensitivity Tuners")
bbb_gain = st.sidebar.slider("BBB Degradation Rate (k_gain)", 0.05, 0.30, 0.12)
m1_gain = st.sidebar.slider("Microglia Activation Rate (k_gain)", 0.05, 0.40, 0.20)

#  3. THE COMPUTATIONAL BACKEND ENGINE
@st.cache_data # Smart feature: caches data so it doesn't slow down the server
def run_simulation(latitude, longitude, s_date, e_date, b_gain, m_gain):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={s_date}&end_date={e_date}&daily=temperature_2m_max,relative_humidity_2m_max&timezone=auto"
    res = requests.get(url).json()
    
    if 'daily' not in res:
        return None
        
    daily_data = res['daily']
    df = pd.DataFrame({
        'Date': pd.to_datetime(daily_data['time']),
        'Max_Temp': daily_data['temperature_2m_max'],
        'Max_Humidity': daily_data['relative_humidity_2m_max']
    })
    
    df['Heat_Stress_Index'] = df['Max_Temp'] + (0.55 * (df['Max_Humidity']/100) * (df['Max_Temp'] - 14.5))
    df['Anomaly'] = np.clip(df['Heat_Stress_Index'] - 25, 0, None)
    days_timeline = np.arange(len(df))
    
    def internal_ode(y, t, anomaly_data):
        idx = int(np.clip(t, 0, len(anomaly_data) - 1))
        dt_anomaly = anomaly_data[idx]
        BBB_perm, M1_activation = y[0], y[1]
        
        k_bbb_recovery = 0.25 if dt_anomaly == 0 else 0.05
        k_m1_recovery = 0.20 if dt_anomaly == 0 else 0.08
            
        d_BBB_dt = (b_gain * dt_anomaly * (1.0 - BBB_perm)) - (k_bbb_recovery * BBB_perm)
        d_M1_dt = (m_gain * BBB_perm * (1.0 - M1_activation)) - (k_m1_recovery * M1_activation)
        return [d_BBB_dt, d_M1_dt]

    initial_states = [0.05, 0.01]
    solution = odeint(internal_ode, initial_states, days_timeline, args=(df['Anomaly'].values,))
    df['BBB_Leakage'] = solution[:, 0]
    df['Microglia_M1'] = solution[:, 1]
    return df

#  4. DYNAMIC VISUALIZATION INTERFACE
if st.button("Compute Live Global Model"):
    with st.spinner("Accessing Copernicus Satellite Proxies..."):
        data = run_simulation(lat, lon, start_date, end_date, bbb_gain, m1_gain)
        
        if data is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Pathokinetic Response Curve")
                sns.set_theme(style="whitegrid")
                fig, ax1 = plt.subplots(figsize=(10, 5))
                
                ax1.plot(data['Date'], data['Anomaly'], color='#e74c3c', linewidth=2, label='Heat Anomaly')
                ax1.set_ylabel('Climate Stress Anomaly (°C)', color='#e74c3c')
                plt.xticks(rotation=45)
                
                ax2 = ax1.twinx()
                ax2.plot(data['Date'], data['BBB_Leakage'], color='#2980b9', linewidth=2.5, linestyle='--', label='BBB Leakage')
                ax2.plot(data['Date'], data['Microglia_M1'], color='#2c3e50', linewidth=3, label='M1 Activation')
                ax2.set_ylabel('Cellular State (0-1)', color='#2c3e50')
                
                fig.tight_layout()
                st.pyplot(fig)
                
            with col2:
                st.subheader(" Raw Simulated Dataset")
                st.dataframe(data[['Date', 'Anomaly', 'BBB_Leakage', 'Microglia_M1']].style.format(precision=4))
        else:
            st.error("Data Fetching Error. Check coordinates or global connection parameters.")
