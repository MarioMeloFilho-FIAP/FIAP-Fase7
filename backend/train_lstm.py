#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm.py
Author: Mário (DevOps/SRE) & AI Assistant
Version: 1.0
Date: 2025-11-23

Backend script for training LSTM time series models on agricultural data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.lstm_predictor import LSTMPredictor
from models.time_series_preprocessor import TimeSeriesPreprocessor
from config.system_config import (
    LSTM_CONFIG, MODELS_DIR, DATA_DIR, SENSOR_COLUMNS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample agricultural sensor data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic sensor data
    """
    logger.info(f"Generating {n_samples} samples of synthetic data...")
    
    # Generate time index
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    # Generate synthetic data with patterns
    np.random.seed(42)
    
    # Temperature: daily cycle + noise
    hours = np.arange(n_samples) % 24
    temperature = 20 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, n_samples)
    
    # Humidity: inverse of temperature + noise
    humidity = 70 - 15 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, n_samples)
    
    # Soil moisture: slow decay with occasional spikes (irrigation)
    soil_moisture = 50 + np.cumsum(np.random.normal(-0.1, 0.5, n_samples))
    irrigation_events = np.random.choice(n_samples, size=20, replace=False)
    soil_moisture[irrigation_events] += 20
    soil_moisture = np.clip(soil_moisture, 20, 80)
    
    # Light intensity: day/night cycle
    light_intensity = np.maximum(0, 800 * np.sin(2 * np.pi * hours / 24)) + np.random.normal(0, 50, n_samples)
    
    # pH level: relatively stable with small variations
    ph_level = 6.5 + np.random.normal(0, 0.3, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature,
        'humidity': humidity,
        'soil_moisture': soil_moisture,
        'light_intensity': light_intensity,
        'ph_level': ph_level
    })
    
    return df


def train_lstm_model():
    """Train LSTM model on agricultural time series data."""
    print("\n" + "="*65)
    print(" LSTM Time Series Model Training ".center(65))
    print("="*65)
    
    # Check if we have real data, otherwise generate sample data
    data_file = DATA_DIR / "sensor_data.csv"
    
    if data_file.exists():
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file, parse_dates=['timestamp'])
        print(f"\n  ✓ Loaded {len(df)} records from {data_file.name}")
    else:
        logger.info("No existing data found. Generating sample data...")
        df = generate_sample_data(n_samples=2000)
        
        # Save sample data
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_file, index=False)
        print(f"\n  ✓ Generated {len(df)} sample records")
        print(f"  ✓ Saved to {data_file}")
    
    # Display data info
    print(f"\n  Data shape: {df.shape}")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"\n  First few rows:")
    print(df.head().to_string(index=False))
    
    # Select target variable
    print("\n" + "-"*65)
    print("  Available sensors for prediction:")
    for i, col in enumerate(SENSOR_COLUMNS, 1):
        print(f"    {i}. {col}")
    
    try:
        choice = input(f"\n  Select sensor to predict [1-{len(SENSOR_COLUMNS)}] (default: 1): ").strip()
        target_idx = int(choice) - 1 if choice else 0
        target_column = SENSOR_COLUMNS[target_idx]
    except (ValueError, IndexError):
        target_column = SENSOR_COLUMNS[0]
    
    print(f"\n  Target variable: {target_column}")
    
    # Prepare data
    print("\n" + "-"*65)
    print("  Preparing data for LSTM...")
    
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=LSTM_CONFIG['sequence_length'],
        prediction_horizon=LSTM_CONFIG['prediction_horizon']
    )
    
    X_train, y_train, X_test, y_test = preprocessor.prepare_data(
        df=df,
        feature_columns=SENSOR_COLUMNS,
        target_column=target_column,
        train_split=0.8
    )
    
    print(f"  ✓ Training set: {X_train.shape[0]} sequences")
    print(f"  ✓ Test set: {X_test.shape[0]} sequences")
    print(f"  ✓ Input shape: {X_train.shape[1:]} (sequence_length, features)")
    print(f"  ✓ Output shape: {y_train.shape[1:]} (prediction_horizon,)")
    
    # Build and train model
    print("\n" + "-"*65)
    print("  Building LSTM model...")
    
    predictor = LSTMPredictor(
        sequence_length=LSTM_CONFIG['sequence_length'],
        prediction_horizon=LSTM_CONFIG['prediction_horizon'],
        lstm_units=LSTM_CONFIG['lstm_units'],
        dropout_rate=LSTM_CONFIG['dropout_rate'],
        learning_rate=LSTM_CONFIG['learning_rate']
    )
    
    predictor.build_model(n_features=X_train.shape[2])
    
    print("\n  Training model...")
    print(f"  Epochs: {LSTM_CONFIG['epochs']}")
    print(f"  Batch size: {LSTM_CONFIG['batch_size']}")
    print(f"  Validation split: {LSTM_CONFIG['validation_split']}")
    
    # Train
    model_path = MODELS_DIR / f"lstm_{target_column}.h5"
    history = predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=LSTM_CONFIG['epochs'],
        batch_size=LSTM_CONFIG['batch_size'],
        validation_split=0.0,  # Using explicit validation set
        model_save_path=model_path
    )
    
    # Evaluate
    print("\n" + "-"*65)
    print("  Evaluating model on test set...")
    
    loss, mae = predictor.evaluate(X_test, y_test)
    
    print(f"\n  Test Results:")
    print(f"    Loss (MSE): {loss:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    # Save training plot
    plot_path = MODELS_DIR / f"lstm_{target_column}_training.png"
    predictor.plot_training_history(save_path=plot_path)
    
    # Make sample predictions
    print("\n" + "-"*65)
    print("  Sample Predictions:")
    
    predictions = predictor.predict(X_test[:5])
    
    for i in range(min(5, len(predictions))):
        actual = y_test[i]
        pred = predictions[i]
        
        print(f"\n  Sample {i+1}:")
        print(f"    Actual:    {actual}")
        print(f"    Predicted: {pred}")
        print(f"    Error:     {np.abs(actual - pred)}")
    
    print("\n" + "="*65)
    print(f"  ✓ Model saved to: {model_path}")
    print(f"  ✓ Training plot saved to: {plot_path}")
    print("="*65)


if __name__ == "__main__":
    train_lstm_model()
