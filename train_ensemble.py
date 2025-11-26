#!/usr/bin/env python3
"""
Training script for ensemble transformer models.
"""

import argparse
from config import Config
from models.ensemble import EnsembleManager


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ensemble transformer models')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=Config.DATASETS,
                       help='Dataset to train on')
    parser.add_argument('--run', type=int, default=1,
                       help='Run identifier')
    parser.add_argument('--n_members', type=int, default=Config.DEFAULT_N_MEMBERS,
                       help='Number of ensemble members')
    parser.add_argument('--model', type=str, default=Config.DEFAULT_MODEL,
                       help='Transformer model to use')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    config.setup_directories()
    
    # Create input configuration
    input_info = {
        'dataset': args.dataset,
        'n_member': args.n_members,
        'transformer_name': args.model,
        'split': 'dev',
        'eval_metric': 'f1_micro',
        'alpha': 1,
        'beta': 1,
        'gamma': 1,
        'mu': 1,
        'run': args.run,
        'random_shuffle': 1
    }
    
    # Train ensemble
    ensemble_manager = EnsembleManager(config)
    ensemble_manager.train_ensemble_members(input_info)
    
    print(f"Training completed for {args.dataset} run {args.run}")


if __name__ == '__main__':
    main()