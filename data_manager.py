"""
Data loading and processing utilities.
"""

import os
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path

from config import Config


class DataManager:
    """Handles data loading and processing."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def split_data_by_dataset(self, split: str = 'train') -> None:
        """
        Split annotated data by dataset.
        
        Args:
            split: Data split ('train', 'dev', 'test')
        """
        input_path = os.path.join(self.config.DATA_PATH, 'annotators.csv')
        output_path = os.path.join(self.config.DATA_PATH, 'annotators')
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Annotations file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        for dataset in self.config.DATASETS:
            dataset_df = df.loc[(df['dataset'] == dataset) & (df['split'] == split)]
            output_file = os.path.join(output_path, f'{dataset}_{split}_candidates.csv')
            dataset_df.to_csv(output_file, index=False)
        
        print(f"Successfully split data for {split} split")
    
    def get_data(self, dataset: str, split: str) -> Tuple[List[str], List[int], List[Tuple[float, float]]]:
        """
        Load dataset.
        
        Args:
            dataset: Dataset name
            split: Data split
            
        Returns:
            Tuple of (texts, hard_labels, soft_labels)
        """
        file_path = os.path.join(self.config.DATA_PATH, f'{dataset}_{split}.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        X = df.text.tolist()
        y_hard = df.hard_label.tolist()
        y_soft = list(zip(df.soft_label_0.tolist(), df.soft_label_1.tolist()))
        
        print(f"Loaded {len(X)} samples from {dataset} {split} split")
        return X, y_hard, y_soft
    
    def select_candidates(self, dataset: str, split: str) -> Tuple[List[str], List[int]]:
        """
        Randomly select candidate annotations.
        
        Args:
            dataset: Dataset name
            split: Data split
            
        Returns:
            Tuple of (texts, labels)
        """
        input_path = os.path.join(self.config.DATA_PATH, 'annotators')
        file_path = os.path.join(input_path, f'{dataset}_{split}_candidates.csv')
        
        if not os.path.exists(file_path):
            self.split_data_by_dataset(split)
        
        df = pd.read_csv(file_path)
        sampled_df = df.groupby('id').sample(n=1)
        
        print(f"Selected {len(sampled_df)} candidates for {dataset} {split} split")
        return sampled_df.text.to_list(), sampled_df.annotator_label.to_list()