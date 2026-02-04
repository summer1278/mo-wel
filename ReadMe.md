# MO-WEL (Multi-Objective Weighted Ensemble Learning)

MO-WEL trains and evaluates robust ensemble models using multi-objective optimization to combine accuracy, calibration, and regularization for improved predictions in subjective text classification task with multiple annotators. The current implementation uses transformer as backbone, please note it is model-agnostic. 

Key features:
- Train multiple transformer ensemble members with [`ModelTrainer`](models/trainer.py).
- Save/load predictions and evaluate ensemble using [`ResultsManager`](evaluate_ensemble.py) and [`EnsembleEvaluator`](evaluate_ensemble.py).
- Optimize ensemble weights using multi-objective optimization (MO-WEL) via [`MOWELOptimizer`](optimize_mo_wel.py).
- Utilities for dataset handling, custom datasets, metrics and loss calculation.

Project layout
- Root scripts:
  - [train_ensemble.py](train_ensemble.py) — CLI to train ensemble members
  - [evaluate_ensemble.py](evaluate_ensemble.py) — CLI to evaluate an ensemble and save predictions
  - [optimize_mo_wel.py](optimize_mo_wel.py) — CLI to run MO-WEL optimization
- Core configuration & data:
  - [config.py](config.py) — project config and directory setup via [`Config`](config.py)
  - [data_manager.py](data_manager.py) — data loading/splitting: [`DataManager`](data_manager.py)
  - [data/](data/) — dataset CSVs and annotators (e.g., `ArMIS_train.csv`, `annotators.csv`, and `annotators/`)
- Models:
  - [models/ensemble.py](models/ensemble.py) — ensemble logic: [`EnsembleManager`](models/ensemble.py)
  - [models/trainer.py](models/trainer.py) — transformer training: [`ModelTrainer`](models/trainer.py)
  - [models/init.py](models/init.py)
- Utilities:
  - [utils/data_utils.py](utils/data_utils.py) — [`CustomDataset`](utils/data_utils.py)
  - [utils/ensemble_utils.py](utils/ensemble_utils.py) — `ensemble_predictions`, `evaluate_ensemble`, `mo_wel_loss`, `load_member_results`, `predict_with_mo_wel`
  - [utils/loss_calculator.py](utils/loss_calculator.py) — [`LossCalculator`](utils/loss_calculator.py)
  - [utils/init.py](utils/init.py)
- Requirements and deps:
  - [requirements.txt](requirements.txt)

Quick Start

1) Install dependencies:
```sh
pip install -r requirements.txt
```

2) Prepare data:
- The data folder expects files like `data/<DATASET>_train.csv`, `data/<DATASET>_dev.csv`, `data/<DATASET>_test.csv` for annotated ground truth.
- The experiment data used: LeWiDi2023 [https://le-wi-di.github.io/LeWiDi2023/]
- Annotator candidate splitting uses `data/annotators.csv` to produce `data/annotators/<DATASET>_<split>_candidates.csv` via [`DataManager.split_data_by_dataset`](data_manager.py).

3) Setup config and directories (handled by scripts but can be used programmatically):
```py
from config import Config
cfg = Config()
cfg.setup_directories()
```

Usage examples

- Train an ensemble:
```sh
python train_ensemble.py --dataset ArMIS --run 1 --n_members 5 --model google-bert/bert-base-multilingual-uncased
```
This runs [`EnsembleManager.train_ensemble_members`](models/ensemble.py) which uses [`ModelTrainer.train_single_model`](models/trainer.py) and saves each model to `Config.MODEL_PATH`.

- Evaluate an ensemble:
```sh
python evaluate_ensemble.py --dataset ArMIS --run 1 --split test --n_members 5 --model google-bert/bert-base-multilingual-uncased
```
Uses [`EnsembleEvaluator.evaluate_ensemble`](evaluate_ensemble.py) → gets predictions by calling `EnsembleManager._get_single_model_predictions` and writes results with [`ResultsManager.save_results`](evaluate_ensemble.py).

- Optimize ensemble weights with MO-WEL:
```sh
python optimize_mo_wel.py --dataset ArMIS --method BERT --maxiter 100
```
This runs [`MOWELOptimizer.optimize_mo_wel`](optimize_mo_wel.py), which loads member predictions via `utils/ensemble_utils.load_member_results`, optimizes weights using `scipy.optimize.differential_evolution`, and reports metrics and saved outputs.

Key classes & functions (quick reference)
- [`Config`](config.py) — configuration and filesystem paths
- [`DataManager`](data_manager.py) — get_data, split_data_by_dataset, select_candidates
- [`EnsembleManager`](models/ensemble.py) — train_ensemble_members, _get_single_model_predictions, _tokenize_texts
- [`ModelTrainer`](models/trainer.py) — train_single_model, evaluate_model, metrics integration using [`LossCalculator`](utils/loss_calculator.py)
- [`MOWELOptimizer`](optimize_mo_wel.py) — optimize_mo_wel, grid_search_mo_wel, optuna_search_mo_wel
- [`EnsembleEvaluator`](evaluate_ensemble.py) and [`ResultsManager`](evaluate_ensemble.py) — evaluation and saving/loading of ensemble predictions
- [`utils/ensemble_utils.py`](utils/ensemble_utils.py) — ensemble predictions, evaluation helpers, and MO-WEL loss

Output locations
- Models saved under `Config.MODEL_PATH` (see `config.py`)
- Predictions saved under `Config.PREDICTION_PATH` (via [`ResultsManager.save_results`](evaluate_ensemble.py))

Notes & tips
- The training script samples annotator candidates with [`DataManager.select_candidates`](data_manager.py). Ensure `data/annotators.csv` exists if running candidate selection or call `DataManager.split_data_by_dataset`.
- The trainer uses a metrics callback integrated into Hugging Face `Trainer` via [`ModelTrainer._create_metrics_function`](models/trainer.py) using `LossCalculator`.
- MO-WEL supports different optimization approaches; see [`optimize_mo_wel.py`](optimize_mo_wel.py) for `differential_evolution` and grid/Optuna search helpers.

Development
- Code is organized into modular packages: models and utils. Use imports from `models.init` and `utils.init` for convenience (`ModelTrainer`, `EnsembleManager`, `LossCalculator`, utilities).

Further reading / Next steps
- Inspect the ensemble trainer and tokenizer usage in [`models/ensemble.py`](models/ensemble.py) and [`models/trainer.py`](models/trainer.py).
- Inspect MO-WEL loss calculation and evaluation functions in [`utils/ensemble_utils.py`](utils/ensemble_utils.py).

Files mentioned in this README
- [config.py](config.py)
- [data_manager.py](data_manager.py)
- [train_ensemble.py](train_ensemble.py)
- [evaluate_ensemble.py](evaluate_ensemble.py)
- [optimize_mo_wel.py](optimize_mo_wel.py)
- [models/ensemble.py](models/ensemble.py)
- [models/trainer.py](models/trainer.py)
- [models/init.py](models/init.py)
- [utils/data_utils.py](utils/data_utils.py)
- [utils/ensemble_utils.py](utils/ensemble_utils.py)
- [utils/loss_calculator.py](utils/loss_calculator.py)
- [utils/init.py](utils/init.py)
- [requirements.txt](requirements.txt)
- data/ (dataset files and `annotators/` subfolder)

### Cite the work

Cui, X., Huang, Z. and Abeynayake, N.R. (2026) ‘Learning from annotator disagreement via weighted ensemble optimisation for subjective text classification’, Datenbank-Spektrum, (In Press).

Huang, Z., Abeynayake, N.R. and Cui, X., 2025, November. Weak ensemble learning from multiple annotators for subjective text classification. In Proceedings of the The 4th Workshop on Perspectivist Approaches to NLP (pp. 87-99).


```
@inproceedings{huang-etal-2025-weak,
    title = "Weak Ensemble Learning from Multiple Annotators for Subjective Text Classification",
    author = "Huang, Ziyi  and
      Abeynayake, N. R.  and
      Cui, Xia",
    booktitle = "Proceedings of the The 4th Workshop on Perspectivist Approaches to NLP",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.nlperspectives-1.8/",
    doi = "10.18653/v1/2025.nlperspectives-1.8",
    pages = "87--99",
    ISBN = "979-8-89176-350-0"
}
```
```
@article{cui2026learning,
  title={Learning from annotator disagreement via weighted ensemble optimisation for subjective text classification},
  author={Cui, Xia and Huang, Ziyi and Abeynayake, Nishanthi Rupika},
  journal={Datenbank-Spektrum},
  year={2026},
  note={In Press}
}
```
