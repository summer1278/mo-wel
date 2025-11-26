"""
Ensemble utility functions for MO-WEL optimization.
"""

import os
import numpy as np
from typing import Any, Dict, Tuple, List, Optional, Union
from sklearn.metrics import f1_score

from utils.loss_calculator import LossCalculator


def validate_input_info(input_info: Dict[str, Any]) -> None:
    """Validate input configuration dictionary."""
    required_keys = {"alpha", "beta", "gamma", "l2_reg", "dataset", "method"}
    missing = required_keys - input_info.keys()
    if missing:
        raise ValueError(f"Missing required keys in input_info: {missing}")

    for key in ["alpha", "beta", "gamma", "l2_reg"]:
        if input_info[key] < 0:
            raise ValueError(f"{key} must be non-negative, got {input_info[key]}")

    if not isinstance(input_info["dataset"], str) or not input_info["dataset"]:
        raise ValueError("Dataset must be a non-empty string.")

    if not isinstance(input_info["method"], str) or not input_info["method"]:
        raise ValueError("Method must be a non-empty string.")


def ensemble_predictions(
    member_results: np.ndarray,
    weights: Union[np.ndarray, List[float]],
    eval_type: str = "soft",
) -> np.ndarray:
    """
    Compute weighted ensemble predictions.

    Parameters
    ----------
    member_results :
        - 'soft': shape (n_models, n_samples, 2) — class probabilities
        - 'hard': shape (n_models, n_samples) — {0, 1} labels
    weights : array-like of shape (n_models,)
        Model weights (non-negative, will be normalized).
    eval_type : {'soft', 'hard'}
        Prediction type.

    Returns
    -------
    predictions : ndarray
        - 'soft': (n_samples, 2)
        - 'hard': (n_samples,)
    """
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    if weights.sum() == 0:
        weights = np.full_like(weights, 1.0 / len(weights))
    else:
        weights /= weights.sum()

    member_results = np.asarray(member_results)

    if eval_type == "soft":
        if member_results.ndim != 3 or member_results.shape[2] != 2:
            raise ValueError(
                "For soft predictions, member_results must be (n_models, n_samples, 2)."
            )
        return np.einsum("m,mnk->nk", weights, member_results)  # (n_samples, 2)

    elif eval_type == "hard":
        if member_results.ndim != 2:
            raise ValueError(
                "For hard predictions, member_results must be (n_models, n_samples)."
            )
        weighted_votes = np.einsum("m,mn->n", weights, member_results)
        return (weighted_votes >= 0.5).astype(int)

    else:
        raise ValueError("eval_type must be 'soft' or 'hard'")


def evaluate_ensemble(
    hard_preds: np.ndarray,
    soft_preds: np.ndarray,
    y_true_hard: np.ndarray,
    y_true_soft: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate ensemble using multiple objectives.

    Parameters
    ----------
    hard_preds : (n_samples,)
    soft_preds : (n_samples, 2)
    y_true_hard : (n_samples,) — ground-truth {0,1}
    y_true_soft : (n_samples, 2) — ground-truth probabilities

    Returns
    -------
    metrics : dict with keys: 'f1_micro', 'cross_entropy', 'average_MD'
    """
    f1_micro = f1_score(y_true=y_true_hard, y_pred=hard_preds, average="micro")
    
    loss_calc = LossCalculator()
    metrics = loss_calc.compute_loss(soft_preds, y_true_hard, y_true_soft, {
        "alpha": 1, "beta": 1, "gamma": 1
    })
    
    return {
        "f1_micro": float(f1_micro),
        "cross_entropy": metrics["cross_entropy"],
        "average_MD": metrics["average_MD"],
    }


def mo_wel_loss(
    params: np.ndarray,
    member_hard: np.ndarray,
    member_soft: np.ndarray,
    y_hard: np.ndarray,
    y_soft: np.ndarray,
    input_info: Dict[str, Any],
    shuffle_models: bool = False,
    return_details: bool = False,
    random_state: Optional[int] = None,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Loss function for MO-WEL optimization.

    Parameters
    ----------
    params : (n_models + 1,) — [w_1, ..., w_M, k_float]
    shuffle_models : if True, randomly select k models (for robustness)
    return_details : if True, return indices, k, rng_state, etc.
    random_state : seed for reproducibility

    Returns
    -------
    loss : scalar (or tuple with extra info if return_details=True)
    """
    n_models = len(member_soft)
    weights_unnorm = params[:-1]
    k_est = params[-1]

    k = int(np.clip(round(k_est), 1, n_models))
    weights = np.maximum(weights_unnorm[:k], 1e-12)
    weights /= weights.sum()

    rng = np.random.RandomState(random_state or 0)

    if shuffle_models:
        indices = rng.permutation(n_models)[:k]
    else:
        indices = np.arange(k)

    selected_hard = np.array(member_hard)[indices]
    selected_soft = np.array(member_soft)[indices]

    soft_ens = ensemble_predictions(selected_soft, weights, "soft")
    hard_ens = ensemble_predictions(selected_hard, weights, "hard")

    metrics = evaluate_ensemble(hard_ens, soft_ens, y_hard, y_soft)

    from utils import l2_norm  # Import your existing l2_norm function
    
    loss = (
        input_info["alpha"] * (-metrics["f1_micro"])
        + input_info["beta"] * metrics["cross_entropy"]
        + input_info["gamma"] * metrics["average_MD"]
        + input_info["l2_reg"] * l2_norm(weights)
    )

    if return_details:
        return loss, {
            "indices": indices,
            "k": k,
            "weights": weights,
            "metrics": metrics,
            "random_state_used": random_state,
        }

    return loss


def load_member_results(
    dataset_name: str,
    split: str,
    input_path: str = 'predictions/BERT/',
    base_path: str = ''
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load saved member results.
    
    This function replaces the original loadList functionality.
    """
    import numpy as np
    
    full_input_path = os.path.join(base_path, input_path)
    hard_file = os.path.join(full_input_path, f'{dataset_name}_{split}_hard_results.npy')
    soft_file = os.path.join(full_input_path, f'{dataset_name}_{split}_soft_results.npy')
    
    hard_results = np.load(hard_file, allow_pickle=True).tolist()
    soft_results = np.load(soft_file, allow_pickle=True).tolist()
    
    return hard_results, soft_results


def predict_with_mo_wel(
    mo_wel_solution: Dict[str, Any],
    member_hard: Union[List, np.ndarray],
    member_soft: Union[List, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using a fitted MO-WEL solution.

    Parameters
    ----------
    mo_wel_solution : dict from `optimize_mo_wel`
    member_hard, member_soft : predictions from all base models

    Returns
    -------
    hard_pred, soft_pred : ensemble predictions
    """
    weights = np.asarray(mo_wel_solution["weights"])
    indices = np.asarray(mo_wel_solution["indices"])

    selected_hard = np.array(member_hard)[indices]
    selected_soft = np.array(member_soft)[indices]

    hard_pred = ensemble_predictions(selected_hard, weights, "hard")
    soft_pred = ensemble_predictions(selected_soft, weights, "soft")

    return hard_pred, soft_pred