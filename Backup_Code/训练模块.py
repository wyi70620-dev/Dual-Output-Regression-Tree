# === * Library Imports ===
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from itertools import product
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
    return np.min(errors) if errors else np.inf

def run_grid_search(X, Y, param_ranges, n_jobs=-1, model_class=None):
    if model_class is None:
        raise ValueError("Please enter model_class")
        
    keys = list(param_ranges.keys())
    combinations = list(product(*[param_ranges[k] for k in keys]))
    param_grid = pd.DataFrame(combinations, columns=keys)

    def evaluate(row):
        params = row.to_dict()
        if 'sigma1' in params and 'sigma2' not in params:
            params['sigma2'] = 1 - params['sigma1']
        model = model_class(**params, n_jobs=n_jobs)
        min_xerror = get_min_xerror(model, X, Y, n_splits=10)
        return min_xerror

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(row) for _, row in param_grid.iterrows()
    )

    param_grid['MinXerror'] = results
    best_row = param_grid.loc[param_grid['MinXerror'].idxmin()]
    return best_row.to_dict()


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
    "max_depth": list(np.arange(1, 10, 1)),
    "min_samples_split": list(np.arange(1, 10, 1)),
}

best_sse_params = run_grid_search(X_train, Y_train, sse_param_range, n_jobs=-1, model_class=TraditionalDecisionTree)
N_pred_sse, S_pred_sse, Y_pred_sse = train_and_predict(TraditionalDecisionTree, best_sse_params, X_train, Y_train, X_test)

# CHDLTree
chdl_param_range = {
    "max_depth": list(np.arange(1, 10, 1)),
    "min_samples_split": list(np.arange(1, 10, 1)),
    'sigma1': list(np.arange(0.1, 1, 0.1)),
    'delta': list(np.arange(50, 100, 10))
}

best_chdl_params = run_grid_search(X_train, Y_train, chdl_param_range, n_jobs=-1, model_class=CSSEHuberDualLossTree)
N_pred_chdl, S_pred_chdl, Y_pred_chdl = train_and_predict(CSSEHuberDualLossTree, best_chdl_params, X_train, Y_train, X_test)

# PGNLLTree
pgnll_param_range = {
    "max_depth": list(np.arange(1, 10, 1)),
    "min_samples_split": list(np.arange(1, 10, 1)),
}

best_pgnll_params = run_grid_search(X_train, Y_train, pgnll_param_range, n_jobs=-1, model_class=PoissonGammaNLLTree)
N_pred_pgnll, S_pred_pgnll, Y_pred_pgnll = train_and_predict(PoissonGammaNLLTree, best_pgnll_params, X_train, Y_train, X_test)

# CopulaTree
copula_param_range = {
    "max_depth": list(np.arange(1, 10, 1)),
    "min_samples_split": list(np.arange(1, 10, 1)),
}

best_copula_params = run_grid_search(X_train, Y_train, copula_param_range, n_jobs=-1, model_class=ClaytonCopulaTree)
N_pred_copula, S_pred_copula, Y_pred_copula = train_and_predict(ClaytonCopulaTree, best_copula_params, X_train, Y_train, X_test)
        