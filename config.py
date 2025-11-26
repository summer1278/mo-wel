"""
Configuration settings for the ensemble training pipeline.
"""

import os
from pathlib import Path
from typing import List


class Config:
    """Configuration class for training and evaluation parameters."""
    
    # Path configurations
    BASE_PATH = os.getcwd()
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    MODEL_PATH = os.path.join(BASE_PATH, 'trained_models')
    LOG_PATH = os.path.join(BASE_PATH, 'train_logs')
    PREDICTION_PATH = os.path.join(BASE_PATH, 'predictions')
    
    # Model configurations
    DEFAULT_MODEL = 'google-bert/bert-base-multilingual-uncased'
    DATASETS = ['ArMIS', 'ConvAbuse', 'MD-Agreement', 'HS-Brexit']
    MAX_LENGTH = 350
    
    # Training configurations
    DEFAULT_EPOCHS = 20
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LEARNING_RATE = 2e-5
    DEFAULT_WEIGHT_DECAY = 0.01
    DEFAULT_N_MEMBERS = 10
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories."""
        directories = [
            cls.DATA_PATH,
            cls.MODEL_PATH,
            cls.LOG_PATH,
            cls.PREDICTION_PATH,
            os.path.join(cls.LOG_PATH, 'reports')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("All directories created successfully.")
    
    @classmethod
    def get_default_input_info(cls, dataset: str, run: int = 1) -> dict:
        """Get default configuration for a dataset."""
        return {
            'dataset': dataset,
            'n_member': cls.DEFAULT_N_MEMBERS,
            'transformer_name': cls.DEFAULT_MODEL,
            'split': 'dev',
            'eval_metric': 'f1_micro',
            'alpha': 1,
            'beta': 1,
            'gamma': 1,
            'mu': 1,
            'run': run,
            'random_shuffle': 1
        }