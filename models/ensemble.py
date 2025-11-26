"""
Ensemble model management.
"""

import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import Config
from data_manager import DataManager
from models.trainer import ModelTrainer


class EnsembleManager:
    """Manages ensemble model training and predictions."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.model_trainer = ModelTrainer(config)
    
    def train_ensemble_members(self, input_info: Dict[str, Any]) -> None:
        """
        Train all ensemble members.
        
        Args:
            input_info: Configuration dictionary
        """
        dataset = input_info['dataset']
        print(f"Starting ensemble training for {dataset}")
        
        # Load development and test data
        X_dev, y_hard_dev, y_soft_dev = self.data_manager.get_data(
            input_info['dataset'], 
            input_info['split']
        )
        X_test, y_hard_test, y_soft_test = self.data_manager.get_data(
            input_info['dataset'], 
            'test'
        )
        
        # Train each ensemble member
        for i in range(input_info['n_member']):
            print(f"Training ensemble member: run_{input_info['run']}, "
                  f"num_{i} of {input_info['n_member']}")
            
            # Get training data for this member
            X_train, y_train = self.data_manager.select_candidates(
                input_info['dataset'], 
                'train'
            )
            
            # Train the member
            self.model_trainer.train_single_model(
                X_train, y_train,
                X_dev, y_hard_dev, y_soft_dev,
                X_test, y_hard_test, y_soft_test,
                input_info,
                model_name=input_info['transformer_name'],
                member_num=str(i)
            )
        
        print(f"Completed ensemble training for {dataset}")
    
    def _tokenize_texts(self, texts: List[str], model_name: str) -> Dict[str, torch.Tensor]:
        """Tokenize texts using the specified model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH,
            return_tensors="pt"
        )
    
    def _get_single_model_predictions(self, 
                                    X: List[str], 
                                    model_name: str, 
                                    model_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from a single model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            local_files_only=True
        )
        
        test_encodings = self._tokenize_texts(X, model_name)
        
        with torch.no_grad():
            test_outputs = model(**test_encodings)
        
        probs = torch.nn.functional.softmax(test_outputs.logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=1)
        
        return preds, probs