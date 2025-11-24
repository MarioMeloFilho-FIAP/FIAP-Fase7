#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
time_series_preprocessor.py
Author: Mário (DevOps/SRE) & AI Assistant
Version: 1.0
Date: 2025-11-23

Data preprocessing utilities for time series analysis with LSTM models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Preprocessor for agricultural time series data."""
    
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 6):
        """
        Initialize the preprocessor.
        
        Args:
            sequence_length: Number of time steps to use for prediction
            prediction_horizon: Number of time steps to predict ahead
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        
    def create_sequences(
        self, 
        data: np.ndarray, 
        target_column_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Normalized data array
            target_column_idx: Index of the column to predict
            
        Returns:
            X: Input sequences (samples, sequence_length, features)
            y: Target values (samples, prediction_horizon)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            X.append(data[i:i + self.sequence_length])
            
            # Target: next prediction_horizon values of target column
            y.append(data[
                i + self.sequence_length:i + self.sequence_length + self.prediction_horizon,
                target_column_idx
            ])
        
        return np.array(X), np.array(y)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            df: DataFrame with time series data
            feature_columns: List of feature column names
            target_column: Name of target column to predict
            train_split: Fraction of data to use for training
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        self.feature_columns = feature_columns
        
        # Extract features
        data = df[feature_columns].values
        
        # Normalize data
        data_normalized = self.scaler.fit_transform(data)
        
        # Get target column index
        target_idx = feature_columns.index(target_column)
        
        # Create sequences
        X, y = self.create_sequences(data_normalized, target_idx)
        
        # Split into train and test
        train_size = int(len(X) * train_split)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        logger.info(f"Data prepared: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def inverse_transform_predictions(self, predictions: np.ndarray, target_column: str) -> np.ndarray:
        """
        Inverse transform predictions back to original scale.
        
        Args:
            predictions: Normalized predictions
            target_column: Name of target column
            
        Returns:
            Predictions in original scale
        """
        # Create dummy array with same shape as original features
        n_features = len(self.feature_columns)
        target_idx = self.feature_columns.index(target_column)
        
        # For each prediction horizon
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Create array to inverse transform
        dummy = np.zeros((predictions.shape[0], n_features))
        dummy[:, target_idx] = predictions[:, 0] if predictions.shape[1] == 1 else predictions[:, -1]
        
        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy)
        
        return inversed[:, target_idx]
    
    def prepare_univariate_data(
        self,
        series: pd.Series,
        train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare univariate time series data.
        
        Args:
            series: Pandas Series with time series data
            train_split: Fraction of data to use for training
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        # Convert to numpy array and reshape
        data = series.values.reshape(-1, 1)
        
        # Normalize
        data_normalized = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(data_normalized, target_column_idx=0)
        
        # Split
        train_size = int(len(X) * train_split)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        logger.info(f"Univariate data prepared: X_train shape: {X_train.shape}")
        
        return X_train, y_train, X_test, y_test
