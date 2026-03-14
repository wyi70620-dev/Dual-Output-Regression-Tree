# Dual-Output Regression Tree for Insurance Claims

This repository contains my research implementation of dual-output regression trees for insurance frequency-severity modeling.

## Project Overview

The project models claim **frequency (N)** and **severity (S)** jointly, then evaluates aggregate loss prediction (`N * S`) under imbalanced insurance data settings.

Implemented models include:
- Baseline CART-style regression tree (`SSETree`)
- CHDL tree (`CSSEHuberDualLossTree`)
- Poisson-Gamma negative log-likelihood tree (`PGNLLTree`)
- Clayton copula-based tree (`ClaytonCopulaTree`)

## Repository Structure

- `Data Set/`
  - Research datasets used for experiments (`n=100`, `n=500`)
- `Model/`
  - Saved best-tree JSON artifacts for each dataset/model setup
- `Reference Code/`
  - Core model implementations, training pipeline, and evaluation scripts
- `Backup_Code/`
  - Archived/experimental scripts and tuning notes
- `Research_and_Notes/`
  - Research references and methodology notes

## Typical Workflow

1. Prepare data from `Data Set/`
2. Train/tune models using scripts in `Reference Code/`
3. Export best trees as JSON
4. Evaluate prediction quality with statistical metrics (RMSE, MAE, R2, Gini, Tweedie deviance, etc.)

## Tech Stack

- Python
- NumPy / Pandas / SciPy
- scikit-learn
- Optuna

## Notes

- This repository reflects an active research codebase; some scripts are prototype-oriented.
- File names are preserved as in the original research process to keep traceability.

