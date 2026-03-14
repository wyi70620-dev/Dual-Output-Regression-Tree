# === * Library Imports ===
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# === 1. Data Loading and preprocessing ===
train_df = pd.read_csv("~/Desktop/Training_data.csv")
x_cols = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10"]
y_cols = ["N","S"]
X_train = train_df[x_cols].to_numpy(dtype=float)
Y_train = train_df[y_cols].to_numpy(dtype=float)


# === 2. Hyperparameter Tuning ===
def get_min_xerror(model, X, Y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    errors = []
    for train_idx, val_idx in kf.split(X):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        model.fit(X_train, Y_train)
        preds = model.predict(X_val)
        N_true, S_true = Y_val[:, 0], Y_val[:, 1]
        N_pred, S_pred = preds[:, 0], preds[:, 1]        
        Y_true = N_true * S_true
        Y_pred = N_pred * S_pred
        loss = mean_squared_error(Y_true, Y_pred)
        errors.append(loss)
    errors = [e for e in errors if e > 0]
    return np.mean(errors) if errors else np.inf

def run_optuna_tpe_search(X, Y, param_ranges, n_jobs=-1, model_class=None):
    if model_class is None:
        raise ValueError("Please enter model_class")

    total_combos = 1
    for k in param_ranges:
        total_combos *= max(1, len(param_ranges[k]))
    n_trials = int(min(total_combos, 250))

    sampler = TPESampler(seed=42, multivariate=False, n_startup_trials=10)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=None)

    def objective(trial):
        params = {}
        for key, values in param_ranges.items():
            params[key] = trial.suggest_categorical(key, list(values))
        if 'sigma1' in params and 'sigma2' not in params:
            params['sigma2'] = 1 - float(params['sigma1'])
        model = model_class(**params, n_jobs=n_jobs)
        return get_min_xerror(model, X, Y, n_splits=10)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)

    best_params = study.best_trial.params.copy()
    if 'sigma1' in best_params and 'sigma2' not in best_params:
        best_params['sigma2'] = 1 - float(best_params['sigma1'])

    return best_params


# === 3. Model Training ===
# SSETree
sse_param_range = {
    "max_depth": [int(x) for x in range(1, 17)],
    "min_samples_split": [int(x) for x in range(2, 101)],
}

best_sse_params = run_optuna_tpe_search(X_train, Y_train, sse_param_range, n_jobs=-1, model_class=ParallelRegressionTree)
best_tree = ParallelRegressionTree(**best_sse_params)
best_tree.fit(X_train, Y_train)
desktop_path = Path.home() / "Desktop" / "sse_best_tree.json"
best_tree.save_json(desktop_path, feature_names=x_cols, ndigits=6, indent=2)

# CHDLTree
chdl_param_range = {
    "max_depth": [int(x) for x in range(1, 17)],
    "min_samples_split": [int(x) for x in range(2, 101)],
    "sigma1": [round(0.01 * i, 2) for i in range(1, 101)],
    "delta": [int(x) for x in range(50, 5001, 25)],
    "eps": [10**(-i) for i in range(6, 11)]
}

best_chdl_params = run_optuna_tpe_search(X_train, Y_train, chdl_param_range, n_jobs=-1, model_class=CSSEHuberDualLossTree)
best_tree = CSSEHuberDualLossTree(**best_chdl_params)
best_tree.fit(X_train, Y_train)
desktop_path = Path.home() / "Desktop" / "chdl_best_tree.json"
best_tree.save_json(desktop_path, feature_names=x_cols, ndigits=6, indent=2)

# PGNLLTree
pg_param_range = {
    "max_depth": [int(x) for x in range(1, 17)],
    "min_samples_split": [int(x) for x in range(2, 101)]
}

best_pg_params = run_optuna_tpe_search(X_train, Y_train, pg_param_range, n_jobs=-1, model_class=PoissonGammaNLLTree)
best_pg_tree = PoissonGammaNLLTree(**best_pg_params, n_jobs=-1)
best_pg_tree.fit(X_train, Y_train)
desktop_path = Path.home() / "Desktop" / "pg_best_tree.json"
best_pg_tree.save_json(desktop_path, feature_names=x_cols, ndigits=6, indent=2)

# CopulaTree
copula_param_range = {
    "max_depth": [int(x) for x in range(1, 17)],
    "min_samples_split": [int(x) for x in range(2, 101)],
}

best_cop_params = run_optuna_tpe_search(X_train, Y_train, copula_param_range, n_jobs=-1, model_class=ClaytonCopulaTree)
best_tree = ClaytonCopulaTree(**best_cop_params, n_jobs=-1)
best_tree.fit(X_train, Y_train)
desktop_json = Path.home() / "Desktop" / "copula_best_tree.json"
best_tree.save_json(desktop_json, feature_names=x_cols, ndigits=6, indent=2)