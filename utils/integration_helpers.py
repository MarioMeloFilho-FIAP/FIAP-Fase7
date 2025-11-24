#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integration_helpers.py
Author: Mário (DevOps/SRE) & AI Assistant
Version: 1.0
Date: 2025-11-23

Helper functions for integrating subsystems from previous phases.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import importlib.util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_phase_to_path(phase_path: Path) -> bool:
    """
    Add a phase directory to Python path.
    
    Args:
        phase_path: Path to phase directory
        
    Returns:
        True if successful, False otherwise
    """
    if not phase_path.exists():
        logger.warning(f"Phase path does not exist: {phase_path}")
        return False
    
    phase_str = str(phase_path)
    if phase_str not in sys.path:
        sys.path.insert(0, phase_str)
        logger.info(f"Added to Python path: {phase_path}")
    
    return True


def import_phase_module(module_name: str, module_path: Path) -> Optional[object]:
    """
    Dynamically import a module from a phase.
    
    Args:
        module_name: Name for the module
        module_path: Path to the Python file
        
    Returns:
        Imported module or None if failed
    """
    if not module_path.exists():
        logger.warning(f"Module file does not exist: {module_path}")
        return None
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logger.info(f"Successfully imported: {module_name}")
            return module
    except Exception as e:
        logger.error(f"Failed to import {module_name}: {e}")
    
    return None


def check_phase_availability(phase_path: Path, phase_name: str) -> bool:
    """
    Check if a phase directory exists and is accessible.
    
    Args:
        phase_path: Path to phase directory
        phase_name: Name of the phase for logging
        
    Returns:
        True if available, False otherwise
    """
    if not phase_path.exists():
        logger.warning(f"{phase_name} not found at: {phase_path}")
        return False
    
    logger.info(f"{phase_name} available at: {phase_path}")
    return True


def get_phase_status() -> dict:
    """
    Check availability of all previous phases.
    
    Returns:
        Dictionary with phase availability status
    """
    from config.system_config import FASE2_PATH, FASE3_PATH, FASE4_PATH, FASE6_PATH
    
    status = {
        "Fase2": check_phase_availability(FASE2_PATH, "Fase2"),
        "Fase3": check_phase_availability(FASE3_PATH, "Fase3"),
        "Fase4": check_phase_availability(FASE4_PATH, "Fase4"),
        "Fase6": check_phase_availability(FASE6_PATH, "Fase6"),
    }
    
    return status


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO"):
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
    """
    from config.system_config import LOG_FORMAT, LOGS_DIR
    
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = LOGS_DIR / log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True
    )
