# === * Library Imports ===
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from contextlib import contextmanager
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from itertools import product


# === Global Timer Tool ===
TIMINGS = defaultdict(list)

def log_time(label, duration):
    TIMINGS[label].append(float(duration))

def merge_timings(local_dict):
    for k, v in local_dict.items():
        TIMINGS[k].extend([float(x) for x in v])

@contextmanager
def time_block(label: str):
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        log_time(label, t1 - t0)

def report_timings():
    rows = []
    for label, vals in TIMINGS.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        rows.append({
            "Stage": label,
            "Total(s)": round(arr.sum(), 6),
            "Count": int(len(arr)),
            "Mean(s)": round(arr.mean(), 6),
            "Median(s)": round(np.median(arr), 6),
            "Max(s)": round(arr.max(), 6),
            "Min(s)": round(arr.min(), 6),
        })
    if rows:
        df = pd.DataFrame(rows).sort_values(by="Total(s)", ascending=False, ignore_index=True)
        print(df.to_string(index=False))


# === 1. Define CSSE-Huber Loss Function ===
def CHDL_loss(N_true, N_pred, S_true, S_pred, sigma1=1.0, sigma2=1.0, epsilon=1e-8, delta=1.0):
    N_true, N_pred = np.asarray(N_true), np.asarray(N_pred)
    S_true, S_pred = np.asarray(S_true), np.asarray(S_pred)

    # N: CSSE
    numerator = np.abs(N_pred - N_true)
    denominator = np.abs(N_pred) + np.abs(N_true) + epsilon
    csse_loss = np.sum((numerator / denominator) ** 2)

    # S: Huber
    diff = S_true - S_pred
    abs_diff = np.abs(diff)
    huber_loss = np.where(
        abs_diff <= delta,
        0.5 * (diff ** 2),
        delta * (abs_diff - 0.5 * delta)
    )
    huber_loss = np.sum(huber_loss)

    return sigma1 * csse_loss + sigma2 * huber_loss


# === 2. Numerical Optimization Function ===
def optimize_leaf(Y, sigma1=1.0, sigma2=1.0, delta=1.0):
    N_true = Y[:, 0]
    S_true = Y[:, 1]

    def objective(pred):
        N_pred, S_pred = pred[0], pred[1]
        return CHDL_loss(N_true, N_pred, S_true, S_pred,
                         sigma1=sigma1, sigma2=sigma2, delta=delta)

    initial_guess = np.mean(Y, axis=0)
    res = minimize(objective, initial_guess, method='Nelder-Mead',
                   options={'xatol': 1e-8, 'fatol': 1e-8})
    return res.x


# === 3. Define Custom Decision Tree ===
class CSSEHuberDualLossTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CSSEHuberDualLossTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_jobs=-1,
                 sigma1=1.0, sigma2=1.0, delta=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.delta = delta
        self.root = None

    def fit(self, X, Y):
        with time_block("[Tree.fit] total"):
            self.root = self._build_tree(X, Y, depth=0)

    def _build_tree(self, X, Y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            with time_block("[Leaf] optimize_leaf"):
                best_pred = optimize_leaf(Y, self.sigma1, self.sigma2, self.delta)
            return CSSEHuberDualLossTreeNode(value=best_pred)

        def evaluate_split(feature_index):
            local_timings = defaultdict(list)
            t0 = time.time()

            sorted_idx = np.argsort(X[:, feature_index])
            X_sorted = X[sorted_idx, feature_index]
            thresholds = (X_sorted[:-1] + X_sorted[1:]) / 2

            best_loss = float('inf')
            best_split = None

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                Y_left, Y_right = Y[left_mask], Y[right_mask]

                t_l0 = time.time()
                pred_left = optimize_leaf(Y_left, self.sigma1, self.sigma2, self.delta)
                t_l1 = time.time()
                local_timings["optimize_leaf"].append(t_l1 - t_l0)

                t_r0 = time.time()
                pred_right = optimize_leaf(Y_right, self.sigma1, self.sigma2, self.delta)
                t_r1 = time.time()
                local_timings["optimize_leaf"].append(t_r1 - t_r0)

                t_loss0 = time.time()
                loss_left = CHDL_loss(Y_left[:, 0], pred_left[0], Y_left[:, 1], pred_left[1],
                                      sigma1=self.sigma1, sigma2=self.sigma2, delta=self.delta)
                loss_right = CHDL_loss(Y_right[:, 0], pred_right[0], Y_right[:, 1], pred_right[1],
                                       sigma1=self.sigma1, sigma2=self.sigma2, delta=self.delta)
                total_loss = loss_left + loss_right
                t_loss1 = time.time()
                local_timings["CHDL_loss"].append(t_loss1 - t_loss0)

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask,
                        'loss': total_loss
                    }

            local_timings[f"evaluate_split(feature={feature_index})"].append(time.time() - t0)
            return best_split, local_timings

        splits_info = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_split)(i) for i in range(n_features)
        )

        splits = []
        for best_split, local_dict in splits_info:
            if best_split is not None:
                splits.append(best_split)
            merge_timings(local_dict)

        if not splits:
            with time_block("[Leaf] optimize_leaf"):
                best_pred = optimize_leaf(Y, self.sigma1, self.sigma2, self.delta)
            return CSSEHuberDualLossTreeNode(value=best_pred)

        best = min(splits, key=lambda s: s['loss'])

        with time_block("[Node] build_left"):
            left = self._build_tree(X[best['left_mask']], Y[best['left_mask']], depth + 1)
        with time_block("[Node] build_right"):
            right = self._build_tree(X[best['right_mask']], Y[best['right_mask']], depth + 1)

        log_time("[Node] finalize", 0.0)
        return CSSEHuberDualLossTreeNode(best['feature_index'], best['threshold'], left, right)

    def predict(self, X):
        with time_block("[Predict] total"):
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


# === 4. Hyperparameter Tuning ===
def get_min_xerror(model, X, Y, n_splits=10):
    with time_block("[CV] get_min_xerror total"):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        errors = []
        for train_idx, val_idx in kf.split(X):
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]

            t_fit0 = time.time()
            model.fit(X_train, Y_train)
            log_time("[CV] fit", time.time() - t_fit0)

            t_pred0 = time.time()
            preds = model.predict(X_val)
            log_time("[CV] predict", time.time() - t_pred0)

            N_true, S_true = Y_val[:, 0], Y_val[:, 1]
            N_pred, S_pred = preds[:, 0], preds[:, 1]
            loss_N = mean_squared_error(N_true, N_pred)
            loss_S = mean_squared_error(S_true, S_pred)
            errors.append(loss_N + loss_S)

        errors = [e for e in errors if e > 0]
        return np.min(errors) if errors else np.inf

def run_grid_search(X, Y, param_ranges, n_jobs=-1, model_class=None):
    with time_block("[Grid] total"):
        if model_class is None:
            raise ValueError("Please enter model_class")

        keys = list(param_ranges.keys())
        combinations = list(product(*[param_ranges[k] for k in keys]))
        param_grid = pd.DataFrame(combinations, columns=keys)

        def evaluate(row):
            local_timings = defaultdict(list)
            params = row.to_dict()
            if 'sigma1' in params and 'sigma2' not in params:
                params['sigma2'] = 1 - params['sigma1']

            t_init0 = time.time()
            model = model_class(**params, n_jobs=n_jobs)
            local_timings["[Grid] model_init"].append(time.time() - t_init0)

            t_cv0 = time.time()
            min_xerror = get_min_xerror(model, X, Y, n_splits=10)
            local_timings["[Grid] CV(get_min_xerror)"].append(time.time() - t_cv0)

            return min_xerror, local_timings

        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate)(row) for _, row in param_grid.iterrows()
        )

        min_xerrors = []
        for mx, local in results:
            min_xerrors.append(mx)
            merge_timings(local)

        param_grid['MinXerror'] = min_xerrors
        best_row = param_grid.loc[param_grid['MinXerror'].idxmin()]
        return best_row.to_dict()


# === 5. example test ===
# generate data
np.random.seed(42)
X = np.random.normal(0, 1, (100, 5))
N = np.random.poisson(lam=10, size=100)
S = np.random.gamma(shape=2.0, scale=50.0, size=100)
Y = np.column_stack((N, S))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# train
tree = CSSEHuberDualLossTree(max_depth=3, min_samples_split=20, n_jobs=-1)
tree.fit(X_train, Y_train)

# Grid Search
'''
param_ranges = {
    "max_depth": [2, 3],
    "min_samples_split": [10, 20],
    "sigma1": [0.3, 0.7],
    "delta": [10, 50]
}
best param = run_grid_search(X_train, Y_train, param_ranges, n_jobs=-1, model_class=CSSEHuberDualLossTree)
'''
# predict
pred = tree.predict(X_test[:10])

# result
report_timings()
