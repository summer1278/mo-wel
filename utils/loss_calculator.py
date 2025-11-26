"""
Loss calculation utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import f1_score

from utils import sigmoid, cross_entropy, average_MD


class LossCalculator:
    """Calculator for various loss functions."""
    
    @staticmethod
    def compute_loss(predictions: np.ndarray, 
                    y_hard_target: List[int], 
                    y_soft_target: List[Tuple[float, float]], 
                    input_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute combined loss from multiple metrics.
        
        Args:
            predictions: Model predictions
            y_hard_target: Hard labels
            y_soft_target: Soft labels
            input_info: Configuration dictionary
            
        Returns:
            Dictionary containing individual and combined loss values
        """
        # F1 micro score
        f1_micro = f1_score(
            y_true=y_hard_target, 
            y_pred=np.argmax(predictions, axis=1), 
            average='micro'
        )
        negative_f1_micro = -f1_micro
        
        # Cross entropy score
        ce_score = sigmoid(cross_entropy(
            targets=y_soft_target, 
            predictions=predictions
        ))
        
        # Average MD score
        md_score = average_MD(
            targets=y_soft_target, 
            predictions=predictions
        )
        
        # Combined weighted loss
        weighted_loss = (
            input_info["alpha"] * negative_f1_micro + 
            input_info["beta"] * ce_score + 
            input_info["gamma"] * md_score
        )
        
        return {
            "negative_f1_micro": negative_f1_micro,
            "f1_micro": f1_micro,
            "cross_entropy": ce_score,
            "average_MD": md_score,
            "weighted_loss": weighted_loss
        }