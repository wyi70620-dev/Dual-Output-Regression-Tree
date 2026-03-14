# === * Library Imports ===
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# === 1. Hyperparameter Tuning ===
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
        loss_N = mean_squared_error(N_true, N_pred)
        loss_S = mean_squared_error(S_true, S_pred)
        loss = loss_N + loss_S
        errors.append(loss)
    errors = [e for e in errors if e > 0]
    return np.mean(errors) if errors else np.inf

def run_grid_search(X, Y, param_ranges, n_jobs=-1, model_class=None):
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

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

    best_params = study.best_trial.params.copy()
    if 'sigma1' in best_params and 'sigma2' not in best_params:
        best_params['sigma2'] = 1 - float(best_params['sigma1'])

    return best_params


# === 2. Define auxiliary functions ===
def split_data(X, Y, test_size=0.3, random_state=42, shuffle=True):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    return X_train, X_test, Y_train, Y_test

def train_and_predict(model_class, param_dict, X_train, Y_train, X_test):
    model = model_class(**param_dict)
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    N_pred = preds[:, 0]
    S_pred = preds[:, 1]
    Y_pred = N_pred * S_pred
    return N_pred, S_pred, Y_pred


# === 3. Data Loading and preprocessing ===
df = pd.read_csv("~/Desktop/Simulated_Frequency_and_Severity_Data.csv")
X = df[[f'x{i}' for i in range(1, 11)]].values
Y = df[['N', 'S']].values

X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.3, random_state=42, shuffle=True)
N_test = Y_test[:, 0]
S_test = Y_test[:, 1]
Y_test = N_test * S_test


# === 4. Model Training ===
# SSETree
sse_param_range = {
    "max_depth": list(np.arange(1, 16, 1)),
    "min_samples_split": list(np.arange(1, 100, 1)),
}

best_sse_params = run_grid_search(X_train, Y_train, sse_param_range, n_jobs=-1, model_class=TraditionalDecisionTree)
N_pred_sse, S_pred_sse, Y_pred_sse = train_and_predict(TraditionalDecisionTree, best_sse_params, X_train, Y_train, X_test)

# CHDLTree
chdl_param_range = {
    "max_depth": list(np.arange(1, 16, 1)),
    "min_samples_split": list(np.arange(1, 100, 1)),
    'sigma1': list(np.arange(0.01, 1, 0.01)),
    'delta': list(np.arange(50, 1000, 50))
}

best_chdl_params = run_grid_search(X_train, Y_train, chdl_param_range, n_jobs=-1, model_class=CSSEHuberDualLossTree)
N_pred_chdl, S_pred_chdl, Y_pred_chdl = train_and_predict(CSSEHuberDualLossTree, best_chdl_params, X_train, Y_train, X_test)

# PGNLLTree
pgnll_param_range = {
    "max_depth": list(np.arange(1, 16, 1)),
    "min_samples_split": list(np.arange(1, 100, 1)),
}

best_pgnll_params = run_grid_search(X_train, Y_train, pgnll_param_range, n_jobs=-1, model_class=PoissonGammaNLLTree)
N_pred_pgnll, S_pred_pgnll, Y_pred_pgnll = train_and_predict(PoissonGammaNLLTree, best_pgnll_params, X_train, Y_train, X_test)

# CopulaTree
copula_param_range = {
    "max_depth": list(np.arange(1, 16, 1)),
    "min_samples_split": list(np.arange(1, 100, 1)),
}

best_copula_params = run_grid_search(X_train, Y_train, copula_param_range, n_jobs=-1, model_class=ClaytonCopulaTree)
N_pred_copula, S_pred_copula, Y_pred_copula = train_and_predict(ClaytonCopulaTree, best_copula_params, X_train, Y_train, X_test)
