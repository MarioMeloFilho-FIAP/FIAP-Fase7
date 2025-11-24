#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
system_config.py
Author: Mário (DevOps/SRE) & AI Assistant
Version: 1.0
Date: 2025-11-23

Centralized configuration for FarmTech Solutions Phase 7 consolidated system.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
FIAP_BASE = BASE_DIR.parent

# Previous phase paths
FASE2_PATH = FIAP_BASE / "Fase2"
FASE3_PATH = FIAP_BASE / "Fase3" / "fase3-maquina-agricola"
FASE4_PATH = FIAP_BASE / "Fase4" / "fase4-maquina-agricola"
FASE6_PATH = FIAP_BASE / "Fase6" / "Cap1-Grupo"

# Data paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database configuration
DB_PATH = DATA_DIR / "farmtech_consolidated.db"

# LSTM Model configuration
LSTM_CONFIG = {
    "sequence_length": 24,  # 24 hours of data for prediction
    "prediction_horizon": 6,  # Predict 6 hours ahead
    "batch_size": 32,
    "epochs": 50,
    "validation_split": 0.2,
    "learning_rate": 0.001,
    "lstm_units": [64, 32],  # Two LSTM layers
    "dropout_rate": 0.2,
}

# Sensor configuration
SENSOR_COLUMNS = [
    "temperature",
    "humidity",
    "soil_moisture",
    "light_intensity",
    "ph_level",
]

# Dashboard configuration
DASHBOARD_CONFIG = {
    "title": "FarmTech Solutions - Consolidated Dashboard",
    "refresh_interval": 5,  # seconds
    "max_data_points": 1000,
}

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
