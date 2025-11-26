#!/usr/bin/env python3
"""
Evaluation script for ensemble transformer models.
"""

import argparse
import os
import numpy as np
from typing import List, Tuple

from config import Config
from data_manager import DataManager
from models.ensemble import EnsembleManager


class ResultsManager:
    """Handles saving and loading of prediction results."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_results(self, 
                    hard_results: List[np.ndarray], 
                    soft_results: List[np.ndarray],
                    dataset_name: str, 
                    split: str, 
                    output_dir: str = 'mBERT/') -> None:
        """
        Save ensemble results.
        
        Args:
            hard_results: Hard predictions
            soft_results: Soft predictions
            dataset_name: Name of dataset
            split: Data split
            output_dir: Output directory
        """
        output_path = os.path.join(self.config.PREDICTION_PATH, output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        hard_file = os.path.join(output_path, f'{dataset_name}_{split}_hard_results')
        soft_file = os.path.join(output_path, f'{dataset_name}_{split}_soft_results')
        
        np.save(hard_file + '.npy', hard_results)
        np.save(soft_file + '.npy', soft_results)
        
        print(f"Results saved to {hard_file} and {soft_file}")
    
    def load_results(self, 
                    dataset_name: str, 
                    split: str, 
                    input_dir: str = 'mBERT/') -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load saved ensemble results.
        
        Args:
            dataset_name: Name of dataset
            split: Data split
            input_dir: Input directory
            
        Returns:
            Tuple of (hard_results, soft_results)
        """
        input_path = os.path.join(self.config.PREDICTION_PATH, input_dir)
        
        hard_file = os.path.join(input_path, f'{dataset_name}_{split}_hard_results.npy')
        soft_file = os.path.join(input_path, f'{dataset_name}_{split}_soft_results.npy')
        
        hard_results = np.load(hard_file, allow_pickle=True).tolist()
        soft_results = np.load(soft_file, allow_pickle=True).tolist()
        
        print(f"Loaded results for {dataset_name} {split} split")
        return hard_results, soft_results


class EnsembleEvaluator:
    """Handles ensemble model evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.ensemble_manager = EnsembleManager(config)
        self.data_manager = DataManager(config)
        self.results_manager = ResultsManager(config)
    
    def get_ensemble_predictions(self, 
                               input_info: Dict[str, Any],
                               split: str = 'test') -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get predictions from all ensemble members.
        
        Args:
            input_info: Configuration dictionary
            split: Data split to evaluate on
            
        Returns:
            Tuple of (hard_predictions, soft_predictions)
        """
        dataset = input_info['dataset']
        run = input_info['run']
        n_member = input_info['n_member']
        transformer_name = input_info['transformer_name']
        
        model_name_clean = (
            transformer_name.split('/')[1] 
            if '/' in transformer_name 
            else transformer_name
        )
        
        # Load data
        X, _, _ = self.data_manager.get_data(dataset, split)
        
        member_hard_results = []
        member_soft_results = []
        
        for i in range(n_member):
            model_path = os.path.join(
                self.config.MODEL_PATH,
                f"{dataset}/{run}_{dataset}_{model_name_clean}_{i}"
            )
            print(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            
            # Get predictions
            hard_pred, soft_pred = self.ensemble_manager._get_single_model_predictions(
                X, transformer_name, model_path
            )
            member_hard_results.append(hard_pred)
            member_soft_results.append(soft_pred)
        
        return member_hard_results, member_soft_results
    
    def evaluate_ensemble(self, input_info: Dict[str, Any], split: str = 'test') -> None:
        """
        Evaluate ensemble and save predictions.
        
        Args:
            input_info: Configuration dictionary
            split: Data split to evaluate on
        """
        print(f"Evaluating ensemble on {input_info['dataset']} {split} split")
        
        # Get predictions
        hard_results, soft_results = self.get_ensemble_predictions(input_info, split)
        
        # Save results
        self.results_manager.save_results(
            hard_results, 
            soft_results, 
            input_info['dataset'], 
            split
        )
        
        print(f"Evaluation completed for {input_info['dataset']}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ensemble transformer models')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=Config.DATASETS,
                       help='Dataset to evaluate on')
    parser.add_argument('--run', type=int, default=1,
                       help='Run identifier')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'dev', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--n_members', type=int, default=Config.DEFAULT_N_MEMBERS,
                       help='Number of ensemble members')
    parser.add_argument('--model', type=str, default=Config.DEFAULT_MODEL,
                       help='Transformer model used')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    config.setup_directories()
    
    # Create input configuration
    input_info = {
        'dataset': args.dataset,
        'n_member': args.n_members,
        'transformer_name': args.model,
        'split': args.split,
        'eval_metric': 'f1_micro',
        'alpha': 1,
        'beta': 1,
        'gamma': 1,
        'mu': 1,
        'run': args.run,
        'random_shuffle': 1
    }
    
    # Evaluate ensemble
    evaluator = EnsembleEvaluator(config)
    evaluator.evaluate_ensemble(input_info, args.split)
    
    print(f"Evaluation completed for {args.dataset} run {args.run}")


if __name__ == '__main__':
    main()