# === * Library Imports ===
import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed
import json
from pathlib import Path
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# === 1. Traditional Decision Tree ===
class ParallelRegressionTreeNode:
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

class ParallelRegressionTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_jobs=-1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
                
    def _leaf_node(self, y, depth):
        n = y.shape[0]
        loss = float(np.var(y[:, 0]) + np.var(y[:, 1]))
        return ParallelRegressionTreeNode(
            value=np.array(np.mean(y, axis=0), dtype=float),
            n_samples=int(n),
            loss=loss,
            depth=int(depth)
            )

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            return self._leaf_node(y, depth)

        def evaluate_split(feature_index):
            sorted_idx = np.argsort(X[:, feature_index])
            X_sorted = X[sorted_idx, feature_index]
            thresholds = (X_sorted[:-1] + X_sorted[1:]) / 2

            best_mse = float('inf')
            best_split = None

            for threshold in thresholds:
                left_data = X[:, feature_index] <= threshold
                right_data = ~left_data
                if np.sum(left_data) == 0 or np.sum(right_data) == 0:
                    continue

                y_left, y_right = y[left_data], y[right_data]
                mse = (
                    len(y_left) * (np.var(y_left[:, 0]) + np.var(y_left[:, 1])) +
                    len(y_right) * (np.var(y_right[:, 0]) + np.var(y_right[:, 1]))
                ) / (2 * n_samples)

                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_data': left_data,
                        'right_data': right_data,
                        'loss': mse
                    }
            return best_split

        splits = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_split)(feature_index) for feature_index in range(n_features)
        )
        splits = [s for s in splits if s is not None]
        if not splits:
            return self._leaf_node(y, depth)

        best_split = min(splits, key=lambda x: x['loss'])

        left_subtree = self._build_tree(X[best_split['left_data']], y[best_split['left_data']], depth + 1)
        right_subtree = self._build_tree(X[best_split['right_data']], y[best_split['right_data']], depth + 1)

        return ParallelRegressionTreeNode(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            left=left_subtree,
            right=right_subtree,
            n_samples=int(n_samples),
            loss=float(best_split['loss']),
            depth=int(depth)
        )

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
        
    def predict_named(self, X):
        preds = self.predict(X)
        return [{'N': row[0], 'S': row[1]} for row in preds]

    def _node_to_dict(self, node, feature_names=None, ndigits=6, depth=0):
        if node is None:
            return None

        def _round(x):
            if x is None:
                return None
            try:
                return round(float(x), ndigits) if ndigits is not None else float(x)
            except Exception:
                return x

        is_leaf = (node.value is not None)
        base = {
            "type": "leaf" if is_leaf else "split",
            "depth": node.depth,
            "n_samples": node.n_samples,
            "loss": _round(node.loss)
            }

        if is_leaf:
            n_pred = _round(node.value[0]) if node.value is not None else None
            s_pred = _round(node.value[1]) if node.value is not None else None

            return {
                **base,
                "value": {
                    "N_pred": n_pred,
                    "S_pred": s_pred,
                    "Y_pred": _round((node.value[0] * node.value[1]) if node.value is not None else None)
                    }
                }
        
        feat_idx = int(node.feature_index)
        feat_name = None
        if feature_names is not None and 0 <= feat_idx < len(feature_names):
            feat_name = str(feature_names[feat_idx])

        split_info = {
            "feature_index": feat_idx,
            "feature_name": feat_name,
            "threshold": _round(node.threshold)
            }
        
        return {
            **base,
            **split_info,
            "left": self._node_to_dict(node.left, feature_names=feature_names, ndigits=ndigits, depth=depth + 1),
            "right": self._node_to_dict(node.right, feature_names=feature_names, ndigits=ndigits, depth=depth + 1)
            }

    def export_json(self, feature_names=None, ndigits=6, indent=2, ensure_ascii=False):
        tree_dict = self._node_to_dict(self.root, feature_names=feature_names, ndigits=ndigits, depth=0)
        return json.dumps(tree_dict, indent=indent, ensure_ascii=ensure_ascii)

    def save_json(self, filepath, feature_names=None, ndigits=6, indent=2, ensure_ascii=False):
        j = self.export_json(
            feature_names=feature_names,
            ndigits=ndigits,
            indent=indent,
            ensure_ascii=ensure_ascii
            )
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(j, encoding="utf-8")
        return str(p)


# === 2. Hyperparameter Tuning ===
def get_min_xerror(model, X, Y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    errors = []
    for train_idx, val_idx in kf.split(X):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        model.fit(X_train, Y_train)
        preds = model.predict(X_val)
        loss = mean_squared_error(Y_val, preds)
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
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=None)

    def objective(trial):
        params = {}
        for key, values in param_ranges.items():
            params[key] = trial.suggest_categorical(key, list(values))
        model = model_class(**params, n_jobs=n_jobs)
        return get_min_xerror(model, X, Y, n_splits=10)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)

    best_params = study.best_trial.params.copy()
    return best_params


# === 3. Data Loading and preprocessing ===
train_df = pd.read_csv("~/Desktop/Training_data.csv")
x_cols = ["Cat1","Cat2","Cat3","Cat4","Con1","Con2","Con3","Con4"]
y_cols = ["N","S"]
X_all = train_df[x_cols].to_numpy(dtype=float)
Y_all = train_df[y_cols].to_numpy(dtype=float)
split_idx = int(0.8 * len(train_df))
X_train, Y_train = X_all[:split_idx], Y_all[:split_idx]


# === 4. Training ===
param_range = {
    "max_depth": [int(x) for x in range(1, 8)],
    "min_samples_split": [int(x) for x in range(2, 51)],
}

best_params = run_grid_search(X_train, Y_train, param_range, n_jobs=-1, model_class=ParallelRegressionTree)
print("Best Parameters:",best_params)
best_tree = ParallelRegressionTree(**best_params)
best_tree.fit(X_train, Y_train)
desktop_path = Path.home() / "Desktop" / "seconedbest_tree.json"
best_tree.save_json(desktop_path, feature_names=x_cols, ndigits=6, indent=2)
print(f"Tree JSON saved to: {desktop_path}")