# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="assets/logo-fiap.jpg" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# Enterprise Challenge

## Hephaestus

## 👨‍🎓 Integrantes

- <a href="[#](https://www.linkedin.com/in/mariomelofilho)">Carlos Mario Vieira de Melo</a>
- <a href="#">Matheus Cardoso Oliveira Lima</a>
- <a href="https://www.linkedin.com/in/silasfr">Silas Fernandes de Souza Fonseca</a>
- <a href="#">Stephanie Dias dos Santos</a>

## 👩‍🏫 Professores

### Tutor(a)

- <a href="https://www.linkedin.com/company/inova-fusca">Leonardo Ruiz Orabona</a>

### Coordenador(a)

- <a href="https://www.linkedin.com/company/inova-fusca">ANDRÉ GODOI CHIOVATO</a>


# FarmTech Solutions - Phase 7: Consolidated System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

**IA como Fertilizante Digital - Um Novo Agronegócio do Amanhã**

Phase 7 consolidates all previous phases (1-6) of the FarmTech Solutions project into a unified agricultural intelligence system with advanced LSTM-based time series prediction capabilities.

## 🌟 Features

- **Consolidated System**: Unified interface for all FarmTech subsystems
- **LSTM Time Series Prediction**: Advanced forecasting for agricultural sensor data
- **Interactive Dashboard**: Real-time visualization with Streamlit
- **Multi-Sensor Support**: Temperature, humidity, soil moisture, light intensity, pH monitoring
- **Modular Architecture**: Easy integration with previous phase implementations
- **Sample Data Generation**: Built-in synthetic data for testing and demonstration

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow 2.13+
- Streamlit 1.28+
- See `requirements.txt` for complete list

## 🚀 Quick Start

### 1. Setup Virtual Environment

```bash
cd /Users/mario/Dropbox/FIAP/Fase7
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the System

```bash
python farmtech_main.py
```

## 📖 Usage Guide

### Main Menu Options

1. **Check System Status** - Verify availability of all subsystems
2. **Generate Sample Data** - Create synthetic agricultural data (Fase2)
3. **Start IoT Data Collection** - Begin sensor data collection (Fase3/4)
4. **Train ML Models** - Train traditional machine learning models (Fase4)
5. **Train LSTM Model** - Train time series prediction model (NEW)
6. **Launch Dashboard** - Open consolidated Streamlit dashboard
7. **Computer Vision** - Run crop analysis (Fase6)
8. **System Information** - Display system details
9. **Exit** - Close the application

### Training LSTM Models

```bash
# From main menu, select option 5
# Or run directly:
python backend/train_lstm.py
```

The training process will:
- Generate sample data if none exists
- Prepare sequences for LSTM input
- Train the model with early stopping
- Save the trained model to `models/saved_models/`
- Generate training history plots

### Launching the Dashboard

```bash
# From main menu, select option 6
# Or run directly:
streamlit run dashboard/farmtech_consolidated_dashboard.py
```

Dashboard features:
- **Overview**: Latest sensor readings and recent trends
- **Sensor Data**: Detailed analysis with time range selection
- **Time Series Forecast**: LSTM-based predictions
- **System Status**: Model and data availability

## 📁 Project Structure

```
Fase7/
├── farmtech_main.py              # Main entry point
├── requirements.txt              # Python dependencies
├── config/
│   └── system_config.py          # Centralized configuration
├── models/
│   ├── lstm_predictor.py         # LSTM model class
│   ├── time_series_preprocessor.py  # Data preprocessing
│   └── saved_models/             # Trained models (created at runtime)
├── backend/
│   └── train_lstm.py             # LSTM training script
├── dashboard/
│   └── farmtech_consolidated_dashboard.py  # Streamlit dashboard
├── utils/
│   └── integration_helpers.py    # Integration utilities
├── data/                         # Data storage (created at runtime)
├── logs/                         # Log files (created at runtime)
└── tests/                        # Unit tests (to be implemented)
```

## 🔧 Configuration

Edit `config/system_config.py` to customize:

- **Paths**: Locations of previous phase implementations
- **LSTM Parameters**: Sequence length, prediction horizon, model architecture
- **Sensor Configuration**: Available sensor types
- **Dashboard Settings**: Refresh intervals, display options

## 🧪 LSTM Model Details

### Architecture

- **Input**: Sequences of sensor readings (default: 24 time steps)
- **LSTM Layers**: Configurable (default: [64, 32] units)
- **Dropout**: Regularization to prevent overfitting (default: 0.2)
- **Output**: Multi-step ahead predictions (default: 6 time steps)

### Training Configuration

```python
LSTM_CONFIG = {
    "sequence_length": 24,      # Hours of history to use
    "prediction_horizon": 6,    # Hours to predict ahead
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "lstm_units": [64, 32],
    "dropout_rate": 0.2,
}
```

## 🔗 Integration with Previous Phases

### Fase 2: Data Generation and Statistics
- Agricultural data generation
- Statistical analysis with R
- Excel report generation

### Fase 3: IoT Data Collection
- ESP32/Arduino sensor integration
- Real-time data collection
- Basic dashboard visualization

### Fase 4: Machine Learning
- Traditional ML model training
- Streamlit dashboard
- Model evaluation and predictions

### Fase 6: Computer Vision
- Crop image analysis
- Object detection for agricultural monitoring

## 📊 Sample Data

The system includes synthetic data generation for demonstration:

- **Temperature**: Daily cycle with realistic variations
- **Humidity**: Inverse correlation with temperature
- **Soil Moisture**: Decay with irrigation events
- **Light Intensity**: Day/night cycle
- **pH Level**: Stable with small variations

## 🐛 Troubleshooting

### Import Errors

If you encounter import errors, ensure:
1. Virtual environment is activated
2. All dependencies are installed: `pip install -r requirements.txt`
3. You're running from the Fase7 directory

### TensorFlow Issues

For M1/M2 Mac users:
```bash
pip install tensorflow-macos tensorflow-metal
```

For GPU support on other systems, see [TensorFlow installation guide](https://www.tensorflow.org/install).

### Dashboard Not Loading

Ensure Streamlit is installed:
```bash
pip install streamlit --upgrade
streamlit --version
```

## 📝 Development

### Adding New Sensors

1. Update `SENSOR_COLUMNS` in `config/system_config.py`
2. Modify data generation in `backend/train_lstm.py`
3. Update dashboard visualizations

### Extending LSTM Models

1. Modify `LSTM_CONFIG` in `config/system_config.py`
2. Adjust model architecture in `models/lstm_predictor.py`
3. Update preprocessing in `models/time_series_preprocessor.py`

## 📚 References

Based on Phase 7 course materials:
- Chapter 1: System Consolidation
- Chapter 2: RNN and LSTM Networks
- Chapter 3: Voice Recognition and Synthesis
- Chapter 4: Genetic Algorithms
- Chapters 5-7: AWS Services and AI
- Chapter 8: ESP32 OOP Programming
- Chapter 9: Cybersecurity

## 👥 Authors

- **Mário** (DevOps/SRE)
- **AI Assistant** (Implementation Support)

## 📄 License

This project is part of the FIAP academic program.

---

**FarmTech Solutions** - Transforming agriculture through artificial intelligence 🌱
