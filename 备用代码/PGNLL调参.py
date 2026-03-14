# === * Library Imports ===
import numpy as np
import pandas as pd
import json
from pathlib import Path
import optuna
from scipy.optimize import minimize_scalar
import scipy.special as sp
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


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
        Y_true = N_true * S_true
        Y_pred = N_pred * S_pred
        loss = mean_squared_error(Y_true, Y_pred)
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

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)

    best_params = study.best_trial.params.copy()
    if 'sigma1' in best_params and 'sigma2' not in best_params:
        best_params['sigma2'] = 1 - float(best_params['sigma1'])

    return best_params


# === 2. Data Loading and preprocessing ===
train_df = pd.read_csv("~/Desktop/Training_data.csv")
x_cols = ["Cat1","Cat2","Cat3","Cat4","Con1","Con2","Con3","Con4"]
y_cols = ["N","S"]
X_all = train_df[x_cols].to_numpy(dtype=float)
Y_all = train_df[y_cols].to_numpy(dtype=float)
split_idx = int(0.8 * len(train_df))
X_train, Y_train = X_all[:split_idx], Y_all[:split_idx]
X_test,  Y_test  = X_all[split_idx:], Y_all[split_idx:]


# === 3. Define Poisson-Gamma NLL Loss Function ===
def PGNLL_loss(N_true, lambda_pred, S_true, alpha, gamma, epsilon=1e-8):
    N_true, lambda_pred  = np.asarray(N_true), np.asarray(lambda_pred)
    S_true, alpha, gamma = np.asarray(S_true), np.asarray(alpha), np.asarray(gamma)

    # Poisson part
    lambda_pred = np.clip(lambda_pred, epsilon, None)
    lambda_pred = np.broadcast_to(lambda_pred, N_true.shape)
    poisson_nll = lambda_pred - N_true * np.log(lambda_pred) + sp.gammaln(N_true + 1)
    total_poisson_loss = np.sum(poisson_nll)

    # Gamma part
    gamma  = np.clip(gamma, epsilon, None)
    S_true = np.clip(S_true, epsilon, None)
    alpha  = np.broadcast_to(alpha, S_true.shape)
    gamma  = np.broadcast_to(gamma, S_true.shape)
    gamma_nll = sp.gammaln(alpha) + S_true / gamma + alpha * np.log(gamma) - (alpha - 1) * np.log(S_true)
    total_gamma_loss = np.sum(gamma_nll)

    total_loss = total_poisson_loss + total_gamma_loss
    return total_loss


# === 4. Parameters Optimization Function ===
def optimize_parameter(Y, epsilon=1e-8):
    N_true = Y[:, 0]
    S_true = Y[:, 1]

    # Poisson Part
    lambda_opt = np.mean(N_true)

    # Gamma Part
    S_true = np.clip(S_true, epsilon, None)
    mean_S = np.mean(S_true)
    
    def gamma_nll_alpha(alpha):
        if alpha <= 0:
            return np.inf
        gamma = mean_S / alpha
        nll = sp.gammaln(alpha) + S_true / gamma + alpha * np.log(gamma) - (alpha - 1) * np.log(S_true)
        return np.sum(nll)

    result = minimize_scalar(gamma_nll_alpha, bounds=(1e-3, 10000), method='bounded')
    if not result.success:
        raise RuntimeError("Gamma alpha optimization failed")
    alpha_opt = result.x
    gamma_opt = mean_S / alpha_opt

    return np.array([lambda_opt, alpha_opt, gamma_opt])


# === 5. Define Custom Decision Tree ===
class PoissonGammaNLLTreeNode:
    def __init__(self,
                 feature_index=None, threshold=None,
                 left=None, right=None,
                 value=None,
                 n_samples=None, loss=None, depth=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.loss = loss
        self.depth = depth

class PoissonGammaNLLTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_jobs=-1, epsilon=1e-8):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.epsilon = epsilon
        self.root = None

    def fit(self, X, Y):
        self.root = self._build_tree(X, Y, depth=0)

    def _leaf_node(self, Y, depth):
        lam, alpha, gamma = optimize_parameter(Y, epsilon=self.epsilon)
        loss = PGNLL_loss(Y[:,0], lam, Y[:,1], alpha, gamma, epsilon=self.epsilon)
        return PoissonGammaNLLTreeNode(
            value=np.array([lam, alpha, gamma], dtype=float),
            n_samples=int(Y.shape[0]),
            loss=float(loss),
            depth=int(depth)
        )

    def _build_tree(self, X, Y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            return self._leaf_node(Y, depth)

        def evaluate_split(feature_index):
            sorted_idx = np.argsort(X[:, feature_index])
            X_sorted = X[sorted_idx, feature_index]
            thresholds = (X_sorted[:-1] + X_sorted[1:]) / 2

            best_loss = float('inf')
            best_split = None

            for threshold in thresholds:
                left_data = X[:, feature_index] <= threshold
                right_data = ~left_data
                if left_data.sum() == 0 or right_data.sum() == 0:
                    continue

                Y_left, Y_right = Y[left_data], Y[right_data]
                lamL, aL, gL = optimize_parameter(Y_left, epsilon=self.epsilon)
                lamR, aR, gR = optimize_parameter(Y_right, epsilon=self.epsilon)

                loss_left = PGNLL_loss(Y_left[:,0], lamL, Y_left[:,1], aL, gL, epsilon=self.epsilon)
                loss_right = PGNLL_loss(Y_right[:,0], lamR, Y_right[:,1], aR, gR, epsilon=self.epsilon)
                total_loss = loss_left + loss_right

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_data': left_data,
                        'right_data': right_data,
                        'loss': float(total_loss)
                    }
            return best_split

        splits = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_split)(i) for i in range(n_features)
        )
        splits = [s for s in splits if s is not None]
        if not splits:
            return self._leaf_node(Y, depth)

        best = min(splits, key=lambda s: s['loss'])
        left = self._build_tree(X[best['left_data']], Y[best['left_data']], depth+1)
        right = self._build_tree(X[best['right_data']], Y[best['right_data']], depth+1)

        return PoissonGammaNLLTreeNode(
            feature_index=int(best['feature_index']),
            threshold=float(best['threshold']),
            left=left, right=right,
            n_samples=int(n_samples),
            loss=float(best['loss']),
            depth=int(depth)
        )

    def predict(self, X):
        out = []
        for x in X:
            lam, alpha, gamma = self._predict_params(x, self.root)
            S_mean = alpha * gamma
            out.append([lam, S_mean])
        return np.asarray(out, dtype=float)

    def _predict_params(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_params(x, node.left)
        else:
            return self._predict_params(x, node.right)

    def _node_to_dict(self, node, feature_names=None, ndigits=6):
        if node is None:
            return None

        def _r(x):
            if x is None:
                return None
            try:
                return round(float(x), ndigits)
            except Exception:
                return float(x)

        is_leaf = node.value is not None
        base = {
            "type": "leaf" if is_leaf else "split",
            "depth": node.depth,
            "n_samples": node.n_samples,
            "loss": _r(node.loss)
        }

        if is_leaf:
            lam, a, g = node.value
            S_mean = a * g
            return {
                **base,
                "params": {"lambda": _r(lam), "alpha": _r(a), "gamma": _r(g)},
                "pred":   {"N_pred": _r(lam), "S_pred": _r(S_mean), "Y_pred": _r(lam * S_mean)}
            }
        else:
            idx = node.feature_index
            name = None
            if feature_names is not None and 0 <= idx < len(feature_names):
                name = str(feature_names[idx])
            return {
                **base,
                "feature_index": int(idx),
                "feature_name": name,
                "threshold": _r(node.threshold),
                "left":  self._node_to_dict(node.left,  feature_names, ndigits),
                "right": self._node_to_dict(node.right, feature_names, ndigits)
            }

    def export_json(self, feature_names=None, ndigits=6, indent=2, ensure_ascii=False):
        tree_dict = self._node_to_dict(self.root, feature_names=feature_names, ndigits=ndigits)
        return json.dumps(tree_dict, indent=indent, ensure_ascii=ensure_ascii)

    def save_json(self, filepath, feature_names=None, ndigits=6, indent=2, ensure_ascii=False):
        j = self.export_json(feature_names=feature_names, ndigits=ndigits, indent=indent, ensure_ascii=ensure_ascii)
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(j, encoding="utf-8")
        return str(p)


# 1) 定义搜索空间
pg_param_range = {
    "max_depth": [int(x) for x in range(1, 8)],
    "min_samples_split": [int(x) for x in range(2, 51)]
}

# 2) 搜索最优参
best_pg_params = run_grid_search(X_train, Y_train, pg_param_range, n_jobs=-1, model_class=PoissonGammaNLLTree)
print("Best PG-Tree Params:", best_pg_params)

# 3) 训练最优模型
best_pg_tree = PoissonGammaNLLTree(**best_pg_params, n_jobs=-1)
best_pg_tree.fit(X_train, Y_train)
desktop_path = Path.home() / "Desktop" / "pg_best_tree.json"
best_pg_tree.save_json(desktop_path, feature_names=x_cols, ndigits=6, indent=2)
print(f"PG Tree JSON saved to: {desktop_path}")
