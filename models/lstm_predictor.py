#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lstm_predictor.py
Author: Mário (DevOps/SRE) & AI Assistant
Version: 1.0
Date: 2025-11-23

LSTM-based time series prediction for agricultural data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM model for time series prediction."""
    
    def __init__(
        self,
        sequence_length: int = 24,
        prediction_horizon: int = 6,
        lstm_units: list = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            sequence_length: Number of time steps in input sequence
            prediction_horizon: Number of time steps to predict
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self, n_features: int) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            n_features: Number of input features
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=(self.sequence_length, n_features)
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_sequences = i < len(self.lstm_units) - 2
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.prediction_horizon))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        logger.info(f"Model built with {n_features} features")
        logger.info(f"Model summary:\n{model.summary()}")
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        model_save_path: Optional[Path] = None
    ) -> dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            X_val: Validation input sequences (optional)
            y_val: Validation target values (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
            model_save_path: Path to save best model
            
        Returns:
            Training history
        """
        if self.model is None:
            n_features = X_train.shape[2]
            self.build_model(n_features)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        if model_save_path:
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=str(model_save_path),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        
        # Train model
        logger.info("Starting model training...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test input sequences
            y_test: Test target values
            
        Returns:
            loss, mae
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
        
        return loss, mae
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path: Path):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load model from file."""
        self.model = keras.models.load_model(str(path))
        logger.info(f"Model loaded from {path}")
