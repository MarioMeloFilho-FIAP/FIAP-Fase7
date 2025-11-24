#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
farmtech_consolidated_dashboard.py
Author: Mário (DevOps/SRE) & AI Assistant
Version: 1.0
Date: 2025-11-23

Consolidated Streamlit dashboard for FarmTech Solutions Phase 7.
Displays sensor data, ML predictions, and LSTM time series forecasts.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.system_config import (
    DATA_DIR, MODELS_DIR, SENSOR_COLUMNS, DASHBOARD_CONFIG
)

# Page configuration
st.set_page_config(
    page_title="FarmTech Solutions",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
    }
    .status-good {
        color: #2E7D32;
        font-weight: bold;
    }
    .status-warning {
        color: #F57C00;
        font-weight: bold;
    }
    .status-critical {
        color: #C62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_sensor_data():
    """Load sensor data from file."""
    data_file = DATA_DIR / "sensor_data.csv"
    
    if data_file.exists():
        df = pd.read_csv(data_file, parse_dates=['timestamp'])
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['timestamp'] + SENSOR_COLUMNS)


def load_lstm_predictions(target_column: str):
    """Load LSTM model predictions if available."""
    model_file = MODELS_DIR / f"lstm_{target_column}.h5"
    
    if model_file.exists():
        # In a real scenario, we would load the model and make predictions
        # For now, return placeholder data
        return None
    else:
        return None


def plot_sensor_timeseries(df: pd.DataFrame, sensor: str):
    """Create time series plot for a sensor."""
    if df.empty or sensor not in df.columns:
        st.warning(f"No data available for {sensor}")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[sensor],
        mode='lines',
        name=sensor,
        line=dict(color='#2E7D32', width=2)
    ))
    
    fig.update_layout(
        title=f"{sensor.replace('_', ' ').title()} Over Time",
        xaxis_title="Time",
        yaxis_title=sensor.replace('_', ' ').title(),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(df: pd.DataFrame):
    """Create correlation heatmap for sensors."""
    if df.empty:
        st.warning("No data available for correlation analysis")
        return
    
    # Calculate correlation
    corr = df[SENSOR_COLUMNS].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdYlGn',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Sensor Correlation Matrix",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<div class="main-header">🌱 FarmTech Solutions Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/2E7D32/FFFFFF?text=FarmTech", 
                 use_container_width=True)
        st.markdown("## Navigation")
        
        page = st.radio(
            "Select View",
            ["Overview", "Sensor Data", "Time Series Forecast", "System Status"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Settings")
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=5,
                max_value=60,
                value=DASHBOARD_CONFIG['refresh_interval']
            )
            st.rerun()
    
    # Load data
    df = load_sensor_data()
    
    # Page routing
    if page == "Overview":
        show_overview(df)
    elif page == "Sensor Data":
        show_sensor_data(df)
    elif page == "Time Series Forecast":
        show_forecast(df)
    elif page == "System Status":
        show_system_status(df)


def show_overview(df: pd.DataFrame):
    """Show overview page."""
    st.header("📊 System Overview")
    
    if df.empty:
        st.info("No sensor data available. Please run data collection or training to generate sample data.")
        return
    
    # Latest readings
    st.subheader("Latest Sensor Readings")
    
    latest = df.iloc[-1] if not df.empty else None
    
    if latest is not None:
        cols = st.columns(len(SENSOR_COLUMNS))
        
        for i, sensor in enumerate(SENSOR_COLUMNS):
            with cols[i]:
                value = latest[sensor]
                st.metric(
                    label=sensor.replace('_', ' ').title(),
                    value=f"{value:.2f}",
                    delta=None
                )
    
    st.markdown("---")
    
    # Recent trends
    st.subheader("Recent Trends (Last 24 Hours)")
    
    if len(df) > 24:
        recent_df = df.tail(24)
    else:
        recent_df = df
    
    # Create subplot for all sensors
    for sensor in SENSOR_COLUMNS[:3]:  # Show first 3 sensors
        plot_sensor_timeseries(recent_df, sensor)


def show_sensor_data(df: pd.DataFrame):
    """Show detailed sensor data page."""
    st.header("📈 Sensor Data Analysis")
    
    if df.empty:
        st.info("No sensor data available.")
        return
    
    # Sensor selection
    selected_sensor = st.selectbox(
        "Select Sensor",
        SENSOR_COLUMNS,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Time range selection
    col1, col2 = st.columns(2)
    
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week", "All Time"]
        )
    
    # Filter data based on time range
    if time_range != "All Time" and not df.empty:
        hours_map = {
            "Last Hour": 1,
            "Last 6 Hours": 6,
            "Last 24 Hours": 24,
            "Last Week": 168
        }
        hours = hours_map.get(time_range, 24)
        filtered_df = df.tail(hours)
    else:
        filtered_df = df
    
    # Plot
    plot_sensor_timeseries(filtered_df, selected_sensor)
    
    # Statistics
    st.subheader("Statistics")
    
    if not filtered_df.empty and selected_sensor in filtered_df.columns:
        col1, col2, col3, col4 = st.columns(4)
        
        sensor_data = filtered_df[selected_sensor]
        
        with col1:
            st.metric("Mean", f"{sensor_data.mean():.2f}")
        with col2:
            st.metric("Min", f"{sensor_data.min():.2f}")
        with col3:
            st.metric("Max", f"{sensor_data.max():.2f}")
        with col4:
            st.metric("Std Dev", f"{sensor_data.std():.2f}")
    
    # Correlation analysis
    st.markdown("---")
    st.subheader("Sensor Correlations")
    plot_correlation_heatmap(filtered_df)


def show_forecast(df: pd.DataFrame):
    """Show LSTM forecast page."""
    st.header("🔮 Time Series Forecast")
    
    if df.empty:
        st.info("No data available for forecasting. Please run LSTM training first.")
        return
    
    st.markdown("""
    This section displays LSTM-based time series forecasts for agricultural sensors.
    The model predicts future values based on historical patterns.
    """)
    
    # Check for trained models
    model_files = list(MODELS_DIR.glob("lstm_*.h5"))
    
    if not model_files:
        st.warning("No trained LSTM models found. Please train a model using option 5 in the main menu.")
        return
    
    # Model selection
    model_names = [f.stem.replace('lstm_', '') for f in model_files]
    selected_model = st.selectbox(
        "Select Forecast Model",
        model_names,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    st.info(f"Model: lstm_{selected_model}.h5")
    st.markdown("*Note: Live predictions require loading the trained model. This is a placeholder view.*")
    
    # Show recent data and placeholder forecast
    if selected_model in df.columns:
        recent_df = df.tail(100)
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=recent_df['timestamp'],
            y=recent_df[selected_model],
            mode='lines',
            name='Historical',
            line=dict(color='#2E7D32', width=2)
        ))
        
        # Placeholder forecast
        last_timestamp = recent_df['timestamp'].iloc[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=6,
            freq='H'
        )
        
        # Simple forecast (last value with small variation)
        last_value = recent_df[selected_model].iloc[-1]
        forecast_values = last_value + np.random.normal(0, 1, 6)
        
        fig.add_trace(go.Scatter(
            x=future_timestamps,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#F57C00', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{selected_model.replace('_', ' ').title()} - Historical and Forecast",
            xaxis_title="Time",
            yaxis_title=selected_model.replace('_', ' ').title(),
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_system_status(df: pd.DataFrame):
    """Show system status page."""
    st.header("⚙️ System Status")
    
    # Data status
    st.subheader("Data Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        if not df.empty:
            latest_time = df['timestamp'].max()
            st.metric("Latest Reading", latest_time.strftime("%Y-%m-%d %H:%M"))
        else:
            st.metric("Latest Reading", "N/A")
    with col3:
        st.metric("Sensors", len(SENSOR_COLUMNS))
    
    # Model status
    st.markdown("---")
    st.subheader("Trained Models")
    
    model_files = list(MODELS_DIR.glob("*.h5")) + list(MODELS_DIR.glob("*.keras"))
    
    if model_files:
        for model_file in model_files:
            st.success(f"✓ {model_file.name}")
    else:
        st.info("No trained models found.")
    
    # System info
    st.markdown("---")
    st.subheader("System Information")
    
    st.code(f"""
Data Directory: {DATA_DIR}
Models Directory: {MODELS_DIR}
Dashboard Version: 1.0
Phase: 7 (Consolidated System)
    """)


if __name__ == "__main__":
    main()
