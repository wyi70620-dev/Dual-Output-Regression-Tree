# === * Library Imports ===
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_tweedie_deviance


# === 1. Data Loading and preprocessing ===
train_df = pd.read_csv("~/Desktop/Testing_data.csv")
x_cols = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10"]
y_cols = ["N","S"]
X_test = train_df[x_cols].to_numpy(dtype=float)
Y_test = train_df[y_cols].to_numpy(dtype=float)
N_test = Y_test[:, 0]
S_test = Y_test[:, 1]
y_test = N_test * S_test


# === 2. Loading json file ===
class FrozenCHDLTree:
    def __init__(self, tree_json_dict, feature_names):
        self.root = tree_json_dict
        self.feature_names = list(feature_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.feature_names)}

    @classmethod
    def from_file(cls, json_path, feature_names):
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return cls(data, feature_names)

    def _predict_one(self, x_row, node):
        if node["type"] == "leaf":
            v = node.get("value", node.get("pred"))
            return np.array([float(v["N_pred"]), float(v["S_pred"])], dtype=float)

        feat_idx = node.get("feature_index", None)
        feat_name = node.get("feature_name", None)

        if feat_idx is None and feat_name is None:
            raise ValueError("Invalid node: no feature_index/feature_name")

        if feat_name is not None:
            i = self.name_to_idx[feat_name]
        else:
            i = int(feat_idx)

        thr = float(node["threshold"])
        if float(x_row[i]) <= thr:
            return self._predict_one(x_row, node["left"])
        else:
            return self._predict_one(x_row, node["right"])

    def predict_NS(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros((X.shape[0], 2), dtype=float)
        for k in range(X.shape[0]):
            out[k] = self._predict_one(X[k], self.root)
        return out

    def predict(self, X):
        ns = self.predict_NS(X)
        return ns[:, 0] * ns[:, 1]


# === 3. Json prediction ===
frozen = FrozenCHDLTree.from_file(
    Path.home() / "Desktop" / "chdl_best_tree.json",
    feature_names=x_cols,
)

Y_pred = frozen.predict(X_test)


# === 4. Statistical metrics ===
eps = 1e-8
power = 1.7

def mpe(y, py):
    y = np.array(y, dtype=float)
    py = np.array(py, dtype=float)
    y[y == 0] = np.nan
    mpe_value = np.nanmean((y - py) / y) * 100
    return mpe_value


def mape(y, py):
    y = np.array(y, dtype=float)
    py = np.array(py, dtype=float)
    y[y == 0] = np.nan
    mape_value = np.nanmean(np.abs((y - py) / y)) * 100
    return mape_value

RMSE = np.sqrt(mean_squared_error(y_test, Y_pred))
MAE = mean_absolute_error(y_test, Y_pred)
R2 = r2_score(y_test, Y_pred)
MPE = mpe(y_test, Y_pred)
MAPE = mape(y_test, Y_pred)

def gini_test(y, py):
    y = np.asarray(y)
    py = np.asarray(py)
    n = len(y)

    np.random.seed(1)
    rand_unif = np.random.rand(n)

    data = pd.DataFrame({'y': y, 'py': py, 'rand_unif': rand_unif})

    data_sorted = data.sort_values(by=['py', 'rand_unif'], ascending=[True, True])
    sorted_y = data_sorted['y'].values

    i = np.arange(1, n + 1)
    gini_index = 1 - 2 / (n - 1) * (n - np.sum(sorted_y * i) / np.sum(sorted_y))

    return gini_index

Gini = gini_test(y_test, Y_pred)
rho = np.corrcoef(Y_pred, y_test)[0, 1] if (np.std(Y_pred) > 0 and np.std(y_test) > 0) else np.nan
CCC = 2 * rho * np.std(Y_pred) * np.std(y_test) / (
    np.var(Y_pred) + np.var(y_test) + (np.mean(Y_pred) - np.mean(y_test))**2 + eps
)

PearsonR = pearsonr(y_test, Y_pred)[0] if np.var(y_test) > 0 and np.var(Y_pred) > 0 else np.nan
SpearmanR = spearmanr(y_test, Y_pred)[0] if np.var(y_test) > 0 and np.var(Y_pred) > 0 else np.nan

y_pred_safe = np.maximum(Y_pred, eps)
TweedieDev = mean_tweedie_deviance(y_test, y_pred_safe, power=1.7)

# result
print(f"RMSE: {RMSE:.6f}")
print(f"MAE: {MAE:.6f}")
print(f"R2: {R2:.6f}")
print(f"MPE: {MPE:.6f}")
print(f"MAPE: {MAPE:.6f}")
print(f"CCC: {CCC:.6f}")
print(f"Gini: {Gini:.6f}")
print(f"PearsonR: {PearsonR:.6f}")
print(f"SpearmanR: {SpearmanR:.6f}")
print(f"Tweedie Deviance: {TweedieDev:.6f}")