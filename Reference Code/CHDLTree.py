# === * Library Imports ===
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from joblib import Parallel, delayed


# === 1. Define CSSE-Huber Loss Function ===
def CHDL_loss(N_true, N_pred, S_true, S_pred, sigma1=1.0, sigma2=1.0, epsilon=1e-8, delta=1.0):
    N_true, N_pred = np.asarray(N_true), np.asarray(N_pred)
    S_true, S_pred = np.asarray(S_true), np.asarray(S_pred)

    # N part: CSSE
    numerator = np.abs(N_pred - N_true)
    denominator = np.abs(N_pred) + np.abs(N_true) + epsilon
    csse_loss = np.sum((numerator / denominator) ** 2)

    # S part: Huber
    diff = S_true - S_pred
    abs_diff = np.abs(diff)
    huber_loss = np.where(
        abs_diff <= delta,
        0.5 * (diff ** 2),
        delta * (abs_diff - 0.5 * delta)
    )
    huber_loss = np.sum(huber_loss)

    total_loss = sigma1 * csse_loss + sigma2 * huber_loss
    return total_loss


# === 2. Numerical Optimization Function ===
def optimize_leaf(Y, sigma1=1.0, sigma2=1.0, delta=1.0):
    N_true = Y[:, 0]
    S_true = Y[:, 1]

    def objective(pred):
        N_pred, S_pred = pred[0], pred[1]
        return CHDL_loss(N_true, N_pred, S_true, S_pred,
                           sigma1=sigma1, sigma2=sigma2, delta=delta)

    initial_guess = np.mean(Y, axis=0)
    res = minimize(objective, initial_guess, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8})
    return res.x


# === 3. Define Custom Decision Tree ===
class CSSEHuberDualLossTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, n_samples=None, loss=None, depth=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  
        self.n_samples = n_samples
        self.loss = loss
        self.depth = depth

class CSSEHuberDualLossTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_jobs=-1,
                 sigma1=1.0, sigma2=1.0, delta=1.0, eps=1e-8):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.delta = delta
        self.epsilon = eps
        self.root = None

    def fit(self, X, Y):
        self.root = self._build_tree(X, Y, depth=0)

    def _build_tree(self, X, Y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            best_pred = optimize_leaf(Y, self.sigma1, self.sigma2, self.delta, self.epsilon)
            node = CSSEHuberDualLossTreeNode(
                value=best_pred,
                n_samples=int(n_samples),
                loss=float(CHDL_loss(Y[:,0], best_pred[0], Y[:,1], best_pred[1],
                                     sigma1=self.sigma1, sigma2=self.sigma2,
                                     epsilon=self.epsilon, delta=self.delta)),
                depth=int(depth)
            )
            return node

        def evaluate_split(feature_index):
            sorted_idx = np.argsort(X[:, feature_index])
            X_sorted = X[sorted_idx, feature_index]
            thresholds = (X_sorted[:-1] + X_sorted[1:]) / 2

            best_loss = float('inf')
            best_split = None

            for threshold in thresholds:
                left_data = X[:, feature_index] <= threshold
                right_data = ~left_data
                if np.sum(left_data) == 0 or np.sum(right_data) == 0:
                    continue

                Y_left, Y_right = Y[left_data], Y[right_data]
                pred_left = optimize_leaf(Y_left, self.sigma1, self.sigma2, self.delta, self.epsilon)
                pred_right = optimize_leaf(Y_right, self.sigma1, self.sigma2, self.delta, self.epsilon)

                loss_left = CHDL_loss(Y_left[:, 0], pred_left[0], Y_left[:, 1], pred_left[1],
                                        sigma1=self.sigma1, sigma2=self.sigma2, epsilon=self.epsilon, delta=self.delta)
                loss_right = CHDL_loss(Y_right[:, 0], pred_right[0], Y_right[:, 1], pred_right[1],
                                        sigma1=self.sigma1, sigma2=self.sigma2, epsilon=self.epsilon, delta=self.delta)

                total_loss = loss_left + loss_right

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_data': left_data,
                        'right_data': right_data,
                        'loss': total_loss
                    }
            return best_split

        splits = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_split)(i) for i in range(n_features)
        )
        splits = [s for s in splits if s is not None]
        if not splits:
            best_pred = optimize_leaf(Y, self.sigma1, self.sigma2, self.delta, self.epsilon)
            return CSSEHuberDualLossTreeNode(
                value=best_pred,
                n_samples=int(n_samples),
                loss=float(CHDL_loss(Y[:,0], best_pred[0], Y[:,1], best_pred[1],
                                     sigma1=self.sigma1, sigma2=self.sigma2,
                                     epsilon=self.epsilon, delta=self.delta)),
                depth=int(depth)
            )

        best = min(splits, key=lambda s: s['loss'])

        left = self._build_tree(X[best['left_data']], Y[best['left_data']], depth + 1)
        right = self._build_tree(X[best['right_data']], Y[best['right_data']], depth + 1)

        return CSSEHuberDualLossTreeNode(
            feature_index=best['feature_index'],
            threshold=best['threshold'],
            left=left,
            right=right,
            n_samples=int(n_samples),
            loss=float(best['loss']),
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
    
    def _node_to_dict(self, node, feature_names=None, ndigits=6):
        if node is None:
            return None

        def _round(x):
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
            "loss": _round(node.loss)
            }

        if is_leaf:
            return {
                **base,
                "value": {
                    "N_pred": _round(node.value[0]),
                    "S_pred": _round(node.value[1]),
                    "Y_pred": _round(node.value[0] * node.value[1])
                    }
                }
        else:
            feat_idx = node.feature_index
            feat_name = None
            if feature_names is not None:
                if 0 <= feat_idx < len(feature_names):
                    feat_name = str(feature_names[feat_idx])

            split_info = {
                "feature_index": int(feat_idx),
                "feature_name": feat_name,
                "threshold": _round(node.threshold)
                }
            return {
                **base,
                **split_info,
                "left": self._node_to_dict(node.left, feature_names, ndigits),
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