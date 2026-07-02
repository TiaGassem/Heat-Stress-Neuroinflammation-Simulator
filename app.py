import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint

#  1. ACADEMIC GRAPHICAL CONFIGURATION
st.set_page_config(page_title="Global Neuro-Climate Simulator (Pro)", layout="wide")

st.title("Advanced In-Silico Pathokinetic Dashboard")
st.markdown("""
**Principal Architect: Tasnim (TiaGassem)** | *Translational Bio-Climate Predictive Framework*
This platform integrates satellite-derived climate stress data with systems biology kinetic models to simulate blood-brain barrier degradation and microglial activation profiles.
""")

# --- SIDEBAR SYSTEMS ---
st.sidebar.header(" Simulation Engine Mode")
app_mode = st.sidebar.radio("Select Analytics View:", ["Single City Deep-Dive", "Multi-City Contrast Analytics"])

st.sidebar.header(" Timeline Parameters")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2025-07-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-07-20"))

st.sidebar.header(" Biological Kinetic Tuners")
bbb_gain = st.sidebar.slider("BBB Degradation Speed (k_gain)", 0.05, 0.30, 0.12)
m1_gain = st.sidebar.slider("Microglia Activation Speed (k_gain)", 0.05, 0.40, 0.20)

# --- CORE MATHEMATICAL & INGESTION BACKEND ---
@st.cache_data
def fetch_and_model(latitude, longitude, s_date, e_date, b_gain, m_gain):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={s_date}&end_date={e_date}&daily=temperature_2m_max,relative_humidity_2m_max&timezone=auto"
    try:
        res = requests.get(url).json()
        if 'daily' not in res: return None
    except:
        return None
        
    daily_data = res['daily']
    df = pd.DataFrame({
        'Date': pd.to_datetime(daily_data['time']),
        'Max_Temp': daily_data['temperature_2m_max'],
        'Max_Humidity': daily_data['relative_humidity_2m_max']
    })
    
    # Calculate Heat Stress Index
    df['Heat_Stress_Index'] = df['Max_Temp'] + (0.55 * (df['Max_Humidity']/100) * (df['Max_Temp'] - 14.5))
    df['Anomaly'] = np.clip(df['Heat_Stress_Index'] - 25, 0, None)
    days_timeline = np.arange(len(df))
    
    def internal_ode(y, t, anomaly_data, bg, mg):
        idx = int(np.clip(t, 0, len(anomaly_data) - 1))
        dt_anomaly = anomaly_data[idx]
        BBB_perm, M1_activation = y[0], y[1]
        
        k_bbb_recovery = 0.25 if dt_anomaly == 0 else 0.05
        k_m1_recovery = 0.20 if dt_anomaly == 0 else 0.08
            
        d_BBB_dt = (bg * dt_anomaly * (1.0 - BBB_perm)) - (k_bbb_recovery * BBB_perm)
        d_M1_dt = (mg * BBB_perm * (1.0 - M1_activation)) - (k_m1_recovery * M1_activation)
        return [d_BBB_dt, d_M1_dt]

    initial_states = [0.05, 0.01]
    
    # Run 3 Scenarios for Shaded Sensitivity Bounds (±20% variance)
    sol_std = odeint(internal_ode, initial_states, days_timeline, args=(df['Anomaly'].values, b_gain, m_gain))
    sol_low = odeint(internal_ode, initial_states, days_timeline, args=(df['Anomaly'].values, b_gain*0.8, m_gain*0.8))
    sol_high = odeint(internal_ode, initial_states, days_timeline, args=(df['Anomaly'].values, b_gain*1.2, m_gain*1.2))
    
    df['BBB_Leakage'] = sol_std[:, 0]
    df['Microglia_M1'] = sol_std[:, 1]
    df['BBB_Low'], df['BBB_High'] = sol_low[:, 0], sol_high[:, 0]
    df['M1_Low'], df['M1_High'] = sol_low[:, 1], sol_high[:, 1]
    
    return df

# --- FRONTEND ROUTING ---
if app_mode == "Single City Deep-Dive":
    st.header(" Single Location Analysis")
    col_input1, col_input2 = st.columns(2)
    with col_input1: lat = st.number_input("Target Latitude", value=50.8503, format="%.4f")
    with col_input2: lon = st.number_input("Target Longitude", value=4.3517, format="%.4f")
    
    if st.button(" Run Analytical Simulation"):
        with st.spinner("Processing Reanalysis Satellite Archives..."):
            data = fetch_and_model(lat, lon, start_date, end_date, bbb_gain, m1_gain)
            if data is not None:
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.subheader("Pathokinetic Response Curve (with Sensitivity Bounds)")
                    sns.set_theme(style="whitegrid")
                    fig, ax1 = plt.subplots(figsize=(10, 4.5))
                    
                    ax1.plot(data['Date'], data['Anomaly'], color='#e74c3c', linewidth=2, label='Climate Anomaly')
                    ax1.set_ylabel('Climate Stress Anomaly (°C)', color='#e74c3c')
                    plt.xticks(rotation=45)
                    
                    ax2 = ax1.twinx()
                    ax2.plot(data['Date'], data['BBB_Leakage'], color='#2980b9', linewidth=2.5, linestyle='--', label='BBB Leakage')
                    ax2.plot(data['Date'], data['Microglia_M1'], color='#2c3e50', linewidth=3, label='M1 Profile')
                    
                    # Beautiful Shaded Sensitivity Analysis Zones
                    ax2.fill_between(data['Date'], data['BBB_Low'], data['BBB_High'], color='#2980b9', alpha=0.15)
                    ax2.fill_between(data['Date'], data['M1_Low'], data['M1_High'], color='#2c3e50', alpha=0.15)
                    ax2.set_ylabel('Cellular State (0-1 Spectrum)', color='#2c3e50')
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                with c2:
                    st.subheader("Core Data Summary")
                    st.dataframe(data[['Date', 'Anomaly', 'BBB_Leakage', 'Microglia_M1']].style.format(precision=3))
                
                # --- CLINICAL EXPORT SYSTEM ---
                st.markdown("---")
                st.subheader(" Clinical Report Export Generation")
                
                max_stress = data['Anomaly'].max()
                max_bbb = data['BBB_Leakage'].max()
                max_m1 = data['Microglia_M1'].max()
                
                report_content = f"""
                # CLINICAL TRANSLATIONAL LAB REPORT
                **Generated By:** Tasnim's In-Silico Pathokinetic Platform  
                **Geographic Target:** Lat {lat}, Lon {lon} | **Evaluation Window:** {start_date} to {end_date}
                
                ###  Simulated Kinetic Milestones
                * **Peak Environmental Climate Stress:** {max_stress:.2f} °C above baseline threshold.
                * **Maximum Simulated Blood-Brain Barrier Permeability:** {max_bbb*100:.1f}% leakage velocity.
                * **Peak Microglial Transgression Index (M1 Phenotype):** {max_m1*100:.1f}% state activation.
                
                ###  Model Verification & Limitations Disclosure
                This abstract simulation presents an open-source proof-of-concept. Biological kinetic rates ($k_{{gain}}$) are derived from baseline mathematical frameworks. Actual pathokinetic progression varies based on individualized demographic and genetic patient baselines.
                """
                st.info("Review the medical summary layout below. Click your browser options to print/save this sector directly to PDF format.")
                st.markdown(report_content)
            else:
                st.error("Data error encountered. Please check connection variables or target dates.")

else:
    st.header(" Multi-City Contrast Analytics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(" Location A (e.g., Tunis)")
        lat_a = st.number_input("Lat A", value=36.8065, format="%.4f")
        lon_a = st.number_input("Lon A", value=10.1815, format="%.4f")
    with col_b:
        st.subheader("Location B (e.g., Brussels)")
        lat_b = st.number_input("Lat B", value=50.8503, format="%.4f")
        lon_b = st.number_input("Lon B", value=4.3517, format="%.4f")
        
    if st.button("Execute Cross-Comparison Engine"):
        with st.spinner("Fetching parallel global reanalysis arrays..."):
            df_a = fetch_and_model(lat_a, lon_a, start_date, end_date, bbb_gain, m1_gain)
            df_b = fetch_and_model(lat_b, lon_b, start_date, end_date, bbb_gain, m1_gain)
            
            if df_a is not None and df_b is not None:
                st.subheader(" Comparative Neuroinflammatory Breakdown")
                fig_comp, (ax_bbb, ax_m1) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot BBB comparison
                ax_bbb.plot(df_a['Date'], df_a['BBB_Leakage'], label=f"Loc A (Lat {lat_a})", color="#e67e22", linewidth=2.5)
                ax_bbb.plot(df_b['Date'], df_b['BBB_Leakage'], label=f"Loc B (Lat {lat_b})", color="#9b59b6", linewidth=2.5, linestyle="--")
                ax_bbb.set_title("Blood-Brain Barrier Permeability Overlap")
                ax_bbb.set_ylabel("Leakage Index (0-1)")
                ax_bbb.legend()
                ax_bbb.tick_params(axis='x', rotation=45)
                
                # Plot Microglia comparison
                ax_m1.plot(df_a['Date'], df_a['Microglia_M1'], label=f"Loc A (Lat {lat_a})", color="#e67e22", linewidth=2.5)
                ax_m1.plot(df_b['Date'], df_b['Microglia_M1'], label=f"Loc B (Lat {lat_b})", color="#9b59b6", linewidth=2.5, linestyle="--")
                ax_m1.set_title("Microglial M1 Activation Profiles")
                ax_m1.set_ylabel("Activation Spectrum (0-1)")
                ax_m1.legend()
                ax_m1.tick_params(axis='x', rotation=45)
                
                fig_comp.tight_layout()
                st.pyplot(fig_comp)
