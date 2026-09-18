import sys
import os

# Ensure project root is in sys.path for Streamlit Cloud & local execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

for path in [ROOT_DIR, CURRENT_DIR, os.getcwd()]:
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

from src.data.loader import load_raw_data, load_config
from src.data.preprocess import preprocess_data
from src.data.features import engineer_features
from src.models.predict import AQIPredictor
from src.models.explainability import compute_partial_dependence

st.set_page_config(
    page_title="Air Quality & Meteorological Impact Dashboard | Gaya, Bihar",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    .station-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: 1px solid #BFDBFE;
        display: inline-block;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prep_data():
    config = load_config()
    df_raw = load_raw_data(config=config)
    df_clean = preprocess_data(df_raw, config=config)
    df_feat = engineer_features(df_clean, include_lags=True, config=config)
    return config, df_raw, df_clean, df_feat

@st.cache_resource
def load_predictor_instance():
    try:
        config = load_config()
        if not os.path.exists(config["best_model_path"]):
            from src.models.train import train_and_evaluate_models
            train_and_evaluate_models(config=config)
        return AQIPredictor()
    except Exception as e:
        st.warning(f"Note: Predictor auto-training notice: {e}")
        return None

def main():
    config, df_raw, df_clean, df_feat = load_and_prep_data()
    predictor = load_predictor_instance()
    
    st.markdown('<div class="main-header">🌫️ Air Quality & Meteorological Impact Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-End Machine Learning System demonstrating meteorological influence on atmospheric pollutants</div>', unsafe_allow_html=True)
    
    # Station Metadata Banner
    st.markdown("""
    <div class="station-badge">
        📍 <b>Data Source:</b> Central Pollution Control Board (CPCB) | <b>Station:</b> Kareemganj, Gaya - BSPCB, Bihar | <b>Period:</b> 2025 (1H Hourly Readings)
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📌 Navigation & Settings")
    st.sidebar.markdown("""
    **Monitoring Details**:
    - **State**: Bihar
    - **City**: Gaya
    - **Station**: Kareemganj (BSPCB)
    - **Authority**: CPCB
    """)
    st.sidebar.divider()
    
    page = st.sidebar.radio(
        "Select Tab",
        [
            "📊 Data & Trends Overview",
            "🔬 Meteorological Driver & Explainability (XAI)",
            "🎛️ Interactive Weather Simulator",
            "📈 Model Performance & Metrics"
        ]
    )
    
    # AQI Category Colors
    aqi_color_map = {
        "Good": "#10B981",
        "Satisfactory": "#84CC16",
        "Moderate": "#F59E0B",
        "Poor": "#F97316",
        "Very Poor": "#EF4444",
        "Severe": "#7F1D1D"
    }

    # ==========================================
    # TAB 1: DATA OVERVIEW
    # ==========================================
    if page == "📊 Data & Trends Overview":
        st.subheader("Gaya Station Summary & Key Environmental Metrics")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Hourly Readings", f"{len(df_clean):,}")
        c2.metric("Mean PM2.5", f"{df_clean['PM2.5'].mean():.1f} µg/m³")
        c3.metric("Max PM2.5", f"{df_clean['PM2.5'].max():.1f} µg/m³")
        c4.metric("Mean Temp (AT)", f"{df_clean['AT'].mean():.1f} °C")
        c5.metric("Mean Humidity (RH)", f"{df_clean['RH'].mean():.1f} %")
        
        st.divider()
        
        st.subheader("Hourly Pollutant & Weather Time Series (Kareemganj, Gaya)")
        pollutant_choice = st.multiselect("Select Variables to Plot", options=["PM2.5", "PM10", "NO2", "AT", "RH"], default=["PM2.5", "AT", "RH"])
        
        if pollutant_choice and "datetime" in df_clean.columns:
            fig_ts = px.line(
                df_clean.sample(min(2000, len(df_clean))).sort_values("datetime"),
                x="datetime",
                y=pollutant_choice,
                title="1-Year Temporal Environmental Variations (Gaya)",
                labels={"datetime": "Date & Time"},
                template="plotly_white"
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Pollutant & Weather Correlation Matrix")
            corr_cols = ["PM2.5", "PM10", "NO2", "CO", "OZONE", "AT", "RH", "WD", "RF"]
            corr_df = df_clean[corr_cols].corr()
            fig_corr = px.imshow(
                corr_df,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Linear Correlation Matrix (CPCB Gaya Data)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_right:
            st.subheader("PM2.5 Distribution across AQI Categories")
            fig_box = px.box(
                df_clean,
                x="AQI_Category",
                y="PM2.5",
                color="AQI_Category",
                color_discrete_map=aqi_color_map,
                category_orders={"AQI_Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]},
                title="PM2.5 Concentration by AQI Category"
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # ==========================================
    # TAB 2: METEOROLOGICAL IMPACT & EXPLAINABILITY
    # ==========================================
    elif page == "🔬 Meteorological Driver & Explainability (XAI)":
        st.subheader("How Meteorological Factors Drive PM2.5 in Gaya, Bihar")
        st.info("💡 **Scientific Context**: Gaya experiences seasonal microclimatic variations. Winter inversion traps fine PM2.5, whereas summer wind direction and humidity dictate dispersion.")
        
        # Load feature importances
        imp_path = os.path.join(config["model_dir"], "feature_importances.csv")
        if os.path.exists(imp_path):
            df_imp = pd.read_csv(imp_path)
            
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown("#### Top Model Feature Importance Ranking")
                fig_imp = px.bar(
                    df_imp.head(15),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Viridis",
                    title="Feature Importances in PM2.5 Prediction Model"
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)
                
            with c_right:
                st.markdown("#### Partial Dependence Analysis (Weather Drivers)")
                met_feature = st.selectbox(
                    "Select Meteorological Variable to Inspect Impact:",
                    options=["AT", "RH", "WD", "RF", "THI"]
                )
                if predictor and predictor.reg_model:
                    try:
                        feature_names = predictor.feature_names
                        X_sample = df_feat[feature_names].sample(min(500, len(df_feat)), random_state=42)
                        pd_res = compute_partial_dependence(predictor.reg_model, X_sample, met_feature)
                        
                        fig_pd = px.line(
                            x=pd_res["values"],
                            y=pd_res["predicted_pm25"],
                            markers=True,
                            title=f"Marginal Effect of {met_feature} on Predicted PM2.5 (µg/m³)",
                            labels={"x": f"{met_feature} Value", "y": "Predicted PM2.5 (µg/m³)"},
                            template="plotly_white"
                        )
                        fig_pd.add_hline(y=df_clean["PM2.5"].mean(), line_dash="dash", annotation_text="Mean PM2.5 Level")
                        st.plotly_chart(fig_pd, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not compute Partial Dependence: {e}")

        st.divider()
        st.markdown("#### Wind Direction vs. PM2.5 Concentration (Gaya Polar Radar)")
        if "WD" in df_clean.columns and "PM2.5" in df_clean.columns:
            wd_bins = np.linspace(0, 360, 17)
            wd_labels = [f"{int(wd_bins[i])}°" for i in range(16)]
            df_wind = df_clean.copy()
            df_wind["WD_sector"] = pd.cut(df_wind["WD"], bins=wd_bins, labels=wd_labels, include_lowest=True)
            wind_summary = df_wind.groupby("WD_sector", observed=False)["PM2.5"].mean().reset_index()
            
            fig_polar = px.line_polar(
                wind_summary,
                r="PM2.5",
                theta="WD_sector",
                line_close=True,
                title="Average PM2.5 Concentration (µg/m³) by Wind Angle Sector in Gaya",
                template="plotly_white"
            )
            fig_polar.update_traces(fill='toself')
            st.plotly_chart(fig_polar, use_container_width=True)

    # ==========================================
    # TAB 3: INTERACTIVE WEATHER SIMULATOR
    # ==========================================
    elif page == "🎛️ Interactive Weather Simulator":
        st.subheader("Real-Time Atmospheric Scenario Simulator")
        st.markdown("Simulate weather scenarios in Gaya to see how changes in Temperature, Humidity, Wind Direction, and Rainfall impact predicted PM2.5 and CPCB AQI categories.")
        
        sim_col1, sim_col2 = st.columns([1, 1])
        
        with sim_col1:
            st.markdown("### 🌡️ Meteorological Inputs")
            at_input = st.slider("Ambient Temperature AT (°C)", min_value=5.0, max_value=45.0, value=25.0, step=0.5)
            rh_input = st.slider("Relative Humidity RH (%)", min_value=10.0, max_value=100.0, value=65.0, step=1.0)
            wd_input = st.slider("Wind Direction WD (Degrees)", min_value=0.0, max_value=360.0, value=180.0, step=5.0)
            rf_input = st.slider("Rainfall RF (mm)", min_value=0.0, max_value=25.0, value=0.0, step=0.5)
            
            st.markdown("### 🏭 Atmospheric Pollutant Baselines")
            pm10_input = st.number_input("PM10 (µg/m³)", value=120.0, step=5.0)
            no2_input = st.number_input("NO2 (µg/m³)", value=35.0, step=2.0)
            co_input = st.number_input("CO (mg/m³)", value=0.8, step=0.1)
            ozone_input = st.number_input("OZONE (µg/m³)", value=25.0, step=2.0)
            
            hour_input = st.selectbox("Hour of Day", options=list(range(24)), index=14)
            month_input = st.selectbox("Month of Year", options=list(range(1, 13)), index=4)

        with sim_col2:
            st.markdown("### 🎯 Model Inferred Predictions")
            
            if predictor:
                sample_dict = {
                    "AT": at_input,
                    "RH": rh_input,
                    "WD": wd_input,
                    "RF": rf_input,
                    "PM10": pm10_input,
                    "NO2": no2_input,
                    "CO": co_input,
                    "OZONE": ozone_input,
                    "hour": hour_input,
                    "month": month_input,
                    "PM2.5": 50.0
                }
                
                res = predictor.predict_sample(sample_dict)
                pred_pm25 = res["predicted_PM2.5"]
                pred_aqi = res["predicted_AQI_category"]
                
                aqi_color = aqi_color_map.get(pred_aqi, "#3B82F6")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_pm25,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Predicted PM2.5 (µg/m³)<br><span style='color:{aqi_color};font-size:1.4em;'>AQI: {pred_aqi}</span>"},
                    gauge={
                        'axis': {'range': [None, 300]},
                        'bar': {'color': aqi_color},
                        'steps': [
                            {'range': [0, 30], 'color': "#10B981"},
                            {'range': [30, 60], 'color': "#84CC16"},
                            {'range': [60, 90], 'color': "#F59E0B"},
                            {'range': [90, 120], 'color': "#F97316"},
                            {'range': [120, 250], 'color': "#EF4444"},
                            {'range': [250, 300], 'color': "#7F1D1D"}
                        ]
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown(f"""
                <div style="background-color:{aqi_color}15; border-left:6px solid {aqi_color}; padding:1rem; border-radius:0.5rem;">
                    <h4 style="margin:0; color:{aqi_color};">AQI Hazard Status: {pred_aqi}</h4>
                    <p style="margin-top:0.5rem; margin-bottom:0;">
                    Under simulated conditions in Gaya (Temperature={at_input}°C, Humidity={rh_input}%), 
                    the estimated PM2.5 level is <b>{pred_pm25} µg/m³</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Model predictor not available.")

    # ==========================================
    # TAB 4: MODEL PERFORMANCE & METRICS
    # ==========================================
    elif page == "📈 Model Performance & Metrics":
        st.subheader("Machine Learning Model Evaluation & Benchmark Results")
        
        summary_path = os.path.join(config["model_dir"], "training_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
                
            st.success(f"Best Selected Model Architecture: **{summary.get('best_regression_model', 'N/A')}**")
            
            st.markdown("#### Regression Benchmark Metrics")
            reg_df = pd.DataFrame(summary["regression_results"]).T
            st.dataframe(reg_df.style.highlight_min(subset=["RMSE", "MAE"], color="#D1FAE5").highlight_max(subset=["R2"], color="#D1FAE5"))
            
            st.markdown("#### AQI Hazard Classification Accuracy & F1 Scores")
            cls_res = summary["classification_results"]
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Classifier Accuracy", f"{cls_res['Accuracy']*100:.2f}%")
            c_m2.metric("Macro F1-Score", f"{cls_res['Macro_F1']:.4f}")
            c_m3.metric("Weighted F1-Score", f"{cls_res['Weighted_F1']:.4f}")

if __name__ == "__main__":
    main()
