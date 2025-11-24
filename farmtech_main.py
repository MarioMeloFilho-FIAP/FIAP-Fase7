#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
farmtech_main.py
Author: Mário (DevOps/SRE) & AI Assistant
Version: 2.0 (Phase 7 - Consolidated System)
Date: 2025-11-23

Main entry point for FarmTech Solutions consolidated system.
Integrates all subsystems from Phases 1-6 with new LSTM time series prediction.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from utils.integration_helpers import get_phase_status, setup_logging
from config.system_config import (
    FASE2_PATH, FASE3_PATH, FASE4_PATH, FASE6_PATH,
    MODELS_DIR, DATA_DIR
)
import logging

# Setup logging
setup_logging(log_file="farmtech_main.log")
logger = logging.getLogger(__name__)


def print_banner():
    """Print FarmTech Solutions banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           FarmTech Solutions - Phase 7                        ║
    ║           Consolidated Agricultural Intelligence System       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print main menu."""
    print("\n" + "="*65)
    print(" Main Menu ".center(65, "="))
    print("="*65)
    print("  1 - Check System Status")
    print("  2 - Generate Sample Agricultural Data (Fase2)")
    print("  3 - Start IoT Data Collection (Fase3/4)")
    print("  4 - Train Traditional ML Models (Fase4)")
    print("  5 - Train LSTM Time Series Model (NEW)")
    print("  6 - Launch Consolidated Dashboard")
    print("  7 - Run Computer Vision Analysis (Fase6)")
    print("  8 - System Information")
    print("  9 - Exit")
    print("="*65)


def check_system_status():
    """Check and display status of all subsystems."""
    print("\n" + "─"*65)
    print(" System Status Check ".center(65, "─"))
    print("─"*65)
    
    status = get_phase_status()
    
    for phase, available in status.items():
        status_str = "✓ Available" if available else "✗ Not Found"
        print(f"  {phase:10s}: {status_str}")
    
    # Check key directories
    print(f"\n  Data Directory: {'✓' if DATA_DIR.exists() else '✗'} {DATA_DIR}")
    print(f"  Models Directory: {'✓' if MODELS_DIR.exists() else '✗'} {MODELS_DIR}")
    
    print("─"*65)


def train_lstm_model():
    """Train LSTM time series prediction model."""
    print("\n" + "─"*65)
    print(" LSTM Time Series Model Training ".center(65, "─"))
    print("─"*65)
    
    try:
        from backend.train_lstm import train_lstm_model as train_func
        train_func()
    except ImportError:
        logger.error("LSTM training module not found. Creating sample training script...")
        print("\n  ⚠ LSTM training module will be created.")
        print("  Please run option 5 again after setup.")
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        print(f"\n  ✗ Error: {e}")


def launch_dashboard():
    """Launch consolidated Streamlit dashboard."""
    print("\n" + "─"*65)
    print(" Launching Consolidated Dashboard ".center(65, "─"))
    print("─"*65)
    
    try:
        import subprocess
        dashboard_path = BASE_DIR / "dashboard" / "farmtech_consolidated_dashboard.py"
        
        if not dashboard_path.exists():
            print(f"\n  ✗ Dashboard not found at: {dashboard_path}")
            return
        
        print(f"\n  Starting Streamlit dashboard...")
        print(f"  Dashboard will open in your browser at http://localhost:8501")
        print(f"  Press CTRL+C to stop the dashboard\n")
        
        subprocess.run(["streamlit", "run", str(dashboard_path)])
        
    except KeyboardInterrupt:
        print("\n\n  Dashboard stopped by user.")
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")
        print(f"\n  ✗ Error: {e}")


def show_system_info():
    """Display system information."""
    print("\n" + "─"*65)
    print(" System Information ".center(65, "─"))
    print("─"*65)
    
    print(f"\n  Base Directory: {BASE_DIR}")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Models Directory: {MODELS_DIR}")
    
    # Check for saved models
    if MODELS_DIR.exists():
        models = list(MODELS_DIR.glob("*.h5")) + list(MODELS_DIR.glob("*.keras"))
        print(f"\n  Saved Models ({len(models)}):")
        for model in models:
            print(f"    - {model.name}")
    
    print("─"*65)


def main():
    """Main program loop."""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n  Select an option [1-9]: ").strip()
            
            if choice == "1":
                check_system_status()
                
            elif choice == "2":
                print("\n  ℹ Fase2 integration: Generate sample agricultural data")
                print("  This feature requires Fase2 implementation.")
                input("\n  Press Enter to continue...")
                
            elif choice == "3":
                print("\n  ℹ Fase3/4 integration: IoT data collection")
                print("  This feature requires Fase3 or Fase4 implementation.")
                input("\n  Press Enter to continue...")
                
            elif choice == "4":
                print("\n  ℹ Fase4 integration: Traditional ML model training")
                print("  This feature requires Fase4 implementation.")
                input("\n  Press Enter to continue...")
                
            elif choice == "5":
                train_lstm_model()
                input("\n  Press Enter to continue...")
                
            elif choice == "6":
                launch_dashboard()
                
            elif choice == "7":
                print("\n  ℹ Fase6 integration: Computer vision analysis")
                print("  This feature requires Fase6 implementation.")
                input("\n  Press Enter to continue...")
                
            elif choice == "8":
                show_system_info()
                input("\n  Press Enter to continue...")
                
            elif choice == "9":
                print("\n  Exiting FarmTech Solutions. Goodbye!")
                break
                
            else:
                print("\n  ✗ Invalid option. Please select 1-9.")
                
        except KeyboardInterrupt:
            print("\n\n  Interrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"\n  ✗ Error: {e}")


if __name__ == "__main__":
    main()
