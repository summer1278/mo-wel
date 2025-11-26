"""
Model training functionality.
"""

import os
import time
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)

from config import Config
from utils.data_utils import CustomDataset
from utils.loss_calculator import LossCalculator


class ModelTrainer:
    """Handles training and evaluation of transformer models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_metrics_function(self, y_soft_label: List[Tuple[float, float]], 
                               input_info: Dict[str, Any]):
        """Create metrics computation function for Trainer."""
        def compute_metrics(eval_pred):
            probs, y_hard_label = eval_pred.predictions, eval_pred.label_ids
            loss_calc = LossCalculator()
            return loss_calc.compute_loss(probs, y_hard_label, y_soft_label, input_info)
        return compute_metrics
    
    def train_single_model(self, 
                         X_train: List[str], 
                         y_train: List[int],
                         X_dev: List[str], 
                         y_hard_dev: List[int], 
                         y_soft_dev: List[Tuple[float, float]],
                         X_test: List[str], 
                         y_hard_test: List[int], 
                         y_soft_test: List[Tuple[float, float]],
                         input_info: Dict[str, Any],
                         model_name: str = 'google-bert/bert-base-multilingual-uncased',
                         member_num: int = 0) -> Dict[str, float]:
        """
        Train a single transformer model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_dev: Development texts
            y_hard_dev: Development hard labels
            y_soft_dev: Development soft labels
            X_test: Test texts
            y_hard_test: Test hard labels
            y_soft_test: Test soft labels
            input_info: Configuration dictionary
            model_name: Transformer model name
            member_num: Ensemble member number
            
        Returns:
            Training metrics
        """
        dataset = input_info['dataset']
        run = input_info['run']
        
        # Dataset-specific preprocessing
        if dataset == 'ConvAbuse':
            y_train = [1 if y < 0 else 0 for y in y_train]
        
        # Initialize tokenizer and model
        print(f"Initializing model {member_num} with {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(set(y_hard_dev))
        )
        
        # Tokenize data
        train_encodings = tokenizer(
            X_train, 
            truncation=True, 
            padding=True, 
            max_length=self.config.MAX_LENGTH, 
            return_tensors="pt"
        )
        dev_encodings = tokenizer(
            X_dev, 
            truncation=True, 
            padding=True, 
            max_length=self.config.MAX_LENGTH, 
            return_tensors="pt"
        )
        
        # Create datasets
        train_dataset = CustomDataset(train_encodings, y_train)
        dev_dataset = CustomDataset(dev_encodings, y_hard_dev)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(
                self.config.LOG_PATH, 
                f"{dataset}/checkpoints_{member_num}"
            ),
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=20,
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01 * random.uniform(0.5, 1.5),
            logging_dir=os.path.join(
                self.config.LOG_PATH, 
                f"{dataset}/logs_{member_num}"
            ),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=input_info['eval_metric'],
            greater_is_better=True,
            save_total_limit=1,
            report_to=[],
        )
        
        # Early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=self._create_metrics_function(y_soft_dev, input_info),
            callbacks=[early_stopping]
        )
        
        # Train model
        print(f"Starting training for model {member_num}")
        start_time = time.time()
        trainer.train()
        train_duration = time.time() - start_time
        
        # Save model
        model_name_clean = model_name.split('/')[1] if '/' in model_name else model_name
        model_path = os.path.join(
            self.config.MODEL_PATH,
            f"{dataset}/{run}_{dataset}_{model_name_clean}_{member_num}"
        )
        Path(model_path).mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_path)
        
        # Evaluate model
        test_encodings = tokenizer(
            X_test,
            truncation=True, 
            padding=True, 
            max_length=self.config.MAX_LENGTH, 
            return_tensors="pt"
        )
        metrics = self.evaluate_model(
            model_path, test_encodings, y_hard_test, y_soft_test, input_info
        )
        metrics['train_dur'] = train_duration
        
        # Save training reports
        self._save_training_reports(metrics, trainer, run, dataset, model_name_clean, member_num)
        
        print(f"Finished training model {member_num} of {input_info['n_member']}, "
              f"evaluated based on {input_info['eval_metric']}")
        
        return metrics
    
    def evaluate_model(self, 
                      model_path: str, 
                      test_encodings: Dict[str, torch.Tensor],
                      y_hard_test: List[int], 
                      y_soft_test: List[Tuple[float, float]], 
                      input_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to saved model
            test_encodings: Tokenized test data
            y_hard_test: Test hard labels
            y_soft_test: Test soft labels
            input_info: Configuration dictionary
            
        Returns:
            Evaluation metrics
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            local_files_only=True
        )
        
        with torch.no_grad():
            test_outputs = model(**test_encodings)
        
        probs = torch.nn.functional.softmax(test_outputs.logits, dim=-1).numpy()
        loss_calc = LossCalculator()
        return loss_calc.compute_loss(probs, y_hard_test, y_soft_test, input_info)
    
    def _save_training_reports(self, 
                             metrics: Dict[str, float], 
                             trainer: Trainer,
                             run: int, 
                             dataset: str, 
                             model_name: str, 
                             member_num: int):
        """Save training metrics and logs."""
        # Save metrics
        df_metrics = pd.DataFrame([metrics])
        metrics_path = os.path.join(
            self.config.LOG_PATH, 
            'reports', 
            f'{run}_{dataset}_{model_name}_{member_num}.csv'
        )
        df_metrics.to_csv(metrics_path, index=False)
        
        # Save training history
        df_history = pd.DataFrame(trainer.state.log_history)
        history_path = os.path.join(
            self.config.LOG_PATH, 
            'reports', 
            f'log_{run}_{dataset}_{model_name}_{member_num}.csv'
        )
        df_history.to_csv(history_path, index=False)
        
        print(f"Training reports saved for {dataset} model {member_num}")