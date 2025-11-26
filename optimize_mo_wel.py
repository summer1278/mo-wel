#!/usr/bin/env python3
"""
Multi-Objective Weighted Ensemble Learning (MO-WEL)

A robust, optimization-based ensemble method that jointly optimizes:
- F1-micro (classification accuracy)
- Cross-entropy (calibration)
- Average Mahanttan Distance (bias/variance trade-off)
- L2 regularization (weight sparsity)

Optimized via differential evolution (scipy) or Bayesian search (Optuna).
"""

import os
import logging
import argparse
from typing import Any, Dict, Tuple, List, Optional, Union
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from config import Config
from data_manager import DataManager
from utils.ensemble_utils import (
    ensemble_predictions,
    evaluate_ensemble,
    mo_wel_loss,
    validate_input_info
)


class MOWELOptimizer:
    """Multi-Objective Weighted Ensemble Learning optimizer."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
    
    def optimize_mo_wel(
        self,
        input_info: Dict[str, Any],
        maxiter: int = 100,
        seed: int = 42,
        callback_interval: int = 10,
    ) -> Dict[str, Any]:
        """Fit MO-WEL on dev set and evaluate on test set."""
        validate_input_info(input_info)

        # Load member results
        from utils.ensemble_utils import load_member_results
        
        hard_dev, soft_dev = load_member_results(
            input_info["dataset"],
            "dev",
            input_path=f'predictions/{input_info["method"]}/',
            base_path=self.config.BASE_PATH
        )
        _, y_hard_dev, y_soft_dev = self.data_manager.get_data(input_info["dataset"], "dev")

        n_models = len(soft_dev)
        bounds = [(0, 1)] * n_models + [(1, n_models)]

        best_params = None
        best_loss = np.inf
        best_random_state = None

        def callback_fn(xk: np.ndarray, convergence: float) -> bool:
            nonlocal best_params, best_loss, best_random_state
            # Evaluate deterministically at current params (no shuffle in callback)
            loss = mo_wel_loss(
                xk, hard_dev, soft_dev, y_hard_dev, y_soft_dev, input_info,
                shuffle_models=False,
            )
            if loss < best_loss:
                best_loss = loss
                best_params = xk.copy()
                best_random_state = int(abs(hash(tuple(xk))) % (2**32 - 1))
            return False  # don't stop early

        # Optimization objective (with shuffling for robustness)
        def objective(x):
            return mo_wel_loss(
                x, hard_dev, soft_dev, y_hard_dev, y_soft_dev, input_info,
                shuffle_models=True,
                random_state=int(abs(hash(tuple(x))) % (2**32 - 1)),
            )

        logging.info(f"Starting MO-WEL optimization for {input_info['dataset']} (method={input_info['method']})...")

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=maxiter,
            tol=1e-6,
            seed=seed,
            strategy="best1bin",
            init="random",
            updating="deferred",
            workers=-1,
            callback=callback_fn,
        )

        # Reconstruct best solution deterministically
        final_loss, details = mo_wel_loss(
            best_params,
            hard_dev, soft_dev,
            y_hard_dev, y_soft_dev,
            input_info,
            shuffle_models=True,
            return_details=True,
            random_state=best_random_state,
        )

        # Load test set
        hard_test, soft_test = load_member_results(
            input_info["dataset"],
            "test",
            input_path=f'predictions/{input_info["method"]}/',
            base_path=self.config.BASE_PATH
        )
        _, y_hard_test, y_soft_test = self.data_manager.get_data(input_info["dataset"], "test")

        # Apply same model selection to test set
        selected_hard = np.array(hard_test)[details["indices"]]
        selected_soft = np.array(soft_test)[details["indices"]]

        test_soft_pred = ensemble_predictions(selected_soft, details["weights"], "soft")
        test_hard_pred = ensemble_predictions(selected_hard, details["weights"], "hard")

        test_metrics = evaluate_ensemble(
            test_hard_pred, test_soft_pred, y_hard_test, y_soft_test
        )

        logging.info(
            f"Optimization complete. Best loss: {result.fun:.4f}, "
            f"F1 (test): {test_metrics['f1_micro']:.4f}"
        )

        return {
            "weights": details["weights"],
            "indices": details["indices"],
            "n_ensembles": details["k"],
            "loss": result.fun,
            "dev_metrics": details["metrics"],
            "test_metrics": test_metrics,
            "optimization_result": result,
            "config": input_info,
        }

    def grid_search_mo_wel(
        self,
        dataset: str = "ArMIS",
        method: str = "BERT",
        hyperparam_values: Optional[List[float]] = None,
        output_dir: str = "reports/mo_wel_grid/",
        maxiter: int = 50,
    ) -> pd.DataFrame:
        """Grid search over α, β, γ, λ (l2_reg)."""
        if hyperparam_values is None:
            hyperparam_values = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0]

        output_path = os.path.join(self.config.BASE_PATH, output_dir)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{dataset}_{method}_grid_results.csv")

        results = []
        total = len(hyperparam_values) ** 4

        logging.info(f"Starting grid search ({total} combinations) for {dataset}/{method}...")

        for i, (alpha, beta, gamma, l2_reg) in enumerate(
            product(hyperparam_values, repeat=4), 1
        ):
            config = {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "l2_reg": l2_reg,
                "dataset": dataset,
                "method": method,
            }

            try:
                res = self.optimize_mo_wel(config, maxiter=maxiter)
                row = {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "l2_reg": l2_reg,
                    "n_ensembles": res["n_ensembles"],
                    "loss": res["loss"],
                    **{f"dev_{k}": v for k, v in res["dev_metrics"].items()},
                    **{f"test_{k}": v for k, v in res["test_metrics"].items()},
                    "weights": np.array2string(res["weights"], precision=4, separator=", "),
                    "indices": np.array2string(res["indices"], separator=", "),
                }
                results.append(row)

                if i % 10 == 0 or i == total:
                    df = pd.DataFrame(results)
                    df.to_csv(output_file, index=False)
                    logging.info(f"[{i}/{total}] Saved {len(results)} results to {output_file}")

            except Exception as e:
                logging.warning(f"Failed at combo {i} ({config}): {e}")
                continue

        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logging.info(f"Grid search complete. Results saved to {output_file}")
        return df

    def optuna_search_mo_wel(
        self,
        dataset: str = "ArMIS",
        method: str = "BERT",
        n_trials: int = 50,
        output_dir: str = "reports/mo_wel_optuna/",
        maxiter: int = 50,
    ) -> pd.DataFrame:
        """Bayesian hyperparameter optimization using Optuna."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for this function. Install with: pip install optuna")

        output_path = os.path.join(self.config.BASE_PATH, output_dir)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{dataset}_{method}_optuna_results.csv")

        def objective(trial: optuna.Trial) -> float:
            config = {
                "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
                "beta": trial.suggest_float("beta", 1e-6, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-6, 1.0, log=True),
                "l2_reg": trial.suggest_float("l2_reg", 1e-6, 1.0, log=True),
                "dataset": dataset,
                "method": method,
            }

            try:
                res = self.optimize_mo_wel(config, maxiter=maxiter)
                # Attach metrics to trial
                for k, v in res["test_metrics"].items():
                    trial.set_user_attr(f"test_{k}", v)
                trial.set_user_attr("n_ensembles", res["n_ensembles"])
                trial.set_user_attr("weights", res["weights"].tolist())
                trial.set_user_attr("indices", res["indices"].tolist())
                return res["loss"]
            except Exception as e:
                trial.set_user_attr("error", str(e))
                return float("inf")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        # Compile results
        records = []
        for trial in study.trials:
            if trial.value == float("inf"):
                continue
            record = {
                "loss": trial.value,
                **trial.params,
                **{k: v for k, v in trial.user_attrs.items() if k != "error"},
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        logging.info(
            f"Optuna search complete ({len(df)} successful trials). "
            f"Best loss: {study.best_value:.4f}, saved to {output_file}"
        )
        return df


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main function for MO-WEL optimization."""
    parser = argparse.ArgumentParser(description='MO-WEL Ensemble Optimization')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=Config.DATASETS,
                       help='Dataset to optimize on')
    parser.add_argument('--method', type=str, required=True,
                       choices=['BERT', 'RF'],  # Add other methods as needed
                       help='Base model method')
    parser.add_argument('--search_type', type=str, default='grid',
                       choices=['grid', 'optuna', 'single'],
                       help='Type of hyperparameter search')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for Optuna search')
    parser.add_argument('--maxiter', type=int, default=50,
                       help='Maximum iterations for optimization')
    
    args = parser.parse_args()
    
    # Setup configuration and logging
    config = Config()
    config.setup_directories()
    setup_logging()
    
    # Create optimizer
    optimizer = MOWELOptimizer(config)
    
    if args.search_type == 'grid':
        logging.info(f"Starting grid search for {args.dataset} with {args.method}")
        optimizer.grid_search_mo_wel(
            dataset=args.dataset,
            method=args.method,
            maxiter=args.maxiter
        )
    
    elif args.search_type == 'optuna':
        logging.info(f"Starting Optuna search for {args.dataset} with {args.method}")
        optimizer.optuna_search_mo_wel(
            dataset=args.dataset,
            method=args.method,
            n_trials=args.n_trials,
            maxiter=args.maxiter
        )
    
    elif args.search_type == 'single':
        logging.info(f"Running single MO-WEL optimization for {args.dataset} with {args.method}")
        input_info = {
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
            "l2_reg": 0.01,
            "dataset": args.dataset,
            "method": args.method,
        }
        result = optimizer.optimize_mo_wel(input_info, maxiter=args.maxiter)
        logging.info(f"Single optimization completed. Test F1: {result['test_metrics']['f1_micro']:.4f}")
    
    logging.info("MO-WEL optimization completed successfully!")


if __name__ == "__main__":
    main()