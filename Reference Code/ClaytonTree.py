# === * Library Imports ===
import json
import numpy as np
from pathlib import Path
import scipy.special as sp
from joblib import Parallel, delayed
from scipy.stats import poisson, gamma as gamma_dist
from scipy.optimize import minimize_scalar, root_scalar


# === 1. Define Clayton Loss Function ===
# PGNLL Loss
def L_PGNLL(N, S, lambda_, alpha, gamma, eps=1e-8):
    N, lambda_      = np.asarray(N, dtype=int), np.asarray(lambda_, dtype=float)
    S, alpha, gamma = np.asarray(S, dtype=float), np.asarray(alpha, dtype=float), np.asarray(gamma, dtype=float)
    
    I_plus = (N >= 1) & (S > 0.0)
    N_plus = N[I_plus]
    S_plus = S[I_plus]

    if N_plus.size == 0:
        return 0.0

    lambda_ = np.broadcast_to(lambda_, N_plus.shape)
    alpha   = np.broadcast_to(alpha, S_plus.shape)
    gamma   = np.broadcast_to(gamma, S_plus.shape)
    
    lambda_ = np.clip(lambda_, eps, None)
    alpha   = np.clip(alpha, eps, None)
    gamma   = np.clip(gamma, eps, None)
    S_plus = np.clip(S_plus, eps, None)

    # Poisson part
    poisson_nll = lambda_ - N_plus * np.log(lambda_) + sp.gammaln(N_plus + 1.0)
    L_poisson = np.sum(poisson_nll)

    # Gamma part
    gamma_nll = sp.gammaln(alpha) + alpha * np.log(gamma) - (alpha - 1.0) * np.log(S_plus) + S_plus / gamma
    L_gamma = np.sum(gamma_nll)

    return float(L_poisson + L_gamma)

# Joint Loss
def L_joint(N, S, lambda_, alpha, gamma, theta, eps=1e-8):
    N = np.asarray(N, dtype=int)
    S = np.asarray(S, dtype=float)

    I_plus = (N >= 1) & (S > 0.0)
    N_plus = N[I_plus]
    S_plus = S[I_plus]

    if N_plus.size == 0:
        return 0.0

    # u_i
    u_i = gamma_dist.cdf(S_plus, a=alpha, scale=gamma)
    u_i = np.clip(u_i, eps, 1.0 - eps)

    # v_{n_i}, v_{n_i-1}
    v_n   = (poisson.cdf(N_plus, mu=lambda_) - np.exp(-lambda_)) / (1.0 - np.exp(-lambda_))
    v_nm1 = (poisson.cdf(N_plus - 1, mu=lambda_) - np.exp(-lambda_)) / (1.0 - np.exp(-lambda_))
    v_n = np.clip(v_n, eps, 1.0 - eps)
    v_nm1 = np.clip(v_nm1, eps, 1.0 - eps)
    
    # Utility function
    def C(u, v, theta):
        return (u**(-theta) + v**(-theta) - 1.0) ** (-1.0 / theta)

    def D_1(u, v, theta):
        return u**(-(theta + 1.0)) * (u**(-theta) + v**(-theta) - 1.0) ** (-(1.0/theta + 1.0))
    
    # Δ_i
    Delta_i = D_1(u_i, v_n, theta) - D_1(u_i, v_nm1, theta)
    Delta_i = np.maximum(Delta_i, eps)
    
    # f_N
    fN = poisson.pmf(N_plus, mu=lambda_)
    fN = np.maximum(fN, eps)
    return float(-np.sum(np.log(Delta_i / fN)))

# Zero–positive indicator term
def L_indicator(N, S, lambda_, eps=1e-8):
    N = np.asarray(N, dtype=int)
    S = np.asarray(S, dtype=float)

    I_0 = (N == 0) & (S == 0.0)
    I_plus = (N >= 1) & (S > 0.0)

    P_N_0 = poisson.pmf(0, mu=lambda_)
    P_N_pos = 1.0 - P_N_0

    L_0 = -np.sum(I_0) * np.log(np.maximum(P_N_0, eps))
    L_p = -np.sum(I_plus) * np.log(np.maximum(P_N_pos, eps))
    return float(L_0 + L_p)

# Clayton Loss (total Loss)
def L_clayton(N, S, lambda_, alpha, gamma, theta):
    return (
        L_PGNLL(N, S, lambda_, alpha, gamma)
        + L_joint(N, S, lambda_, alpha, gamma, theta)
        + L_indicator(N, S, lambda_)
    )


# === 2. Parameters Optimization Function ===
def optimize_parameter(Y, eps=1e-8, lambda_bounds=(1e-6, 100), alpha_bounds=(1e-3, 1e4), theta_bounds=(1e-3, 50.0)):
    Y = np.asarray(Y)
    N = np.asarray(Y[:, 0], dtype=int)
    S = np.asarray(Y[:, 1], dtype=float)

    I_plus = (N >= 1) & (S > 0.0)
    I_0    = (N == 0) & (S == 0.0)

    # Pure zero node
    if np.sum(I_plus) == 0:
        lambda_hat = 0
        alpha_hat  = 1.0
        gamma_hat  = 0.0
        theta_hat  = float(theta_bounds[0])
        return np.array([lambda_hat, alpha_hat, gamma_hat, theta_hat], dtype=float)
    
    # ---------- Step 1: Marginal optimization ----------
    # Poisson Part
    m = int(np.sum(I_plus))
    k = int(np.sum(I_0))
    T = float(np.sum(N[I_plus]))

    def poisson_nll_lambda(lambda_):
        return (m + k) - T / lambda_ - m / np.expm1(lambda_)

    result = root_scalar(poisson_nll_lambda, bracket=lambda_bounds, method="brentq")
    lambda_hat = float(result.root)

    # Gamma Part
    S_plus = np.clip(S[I_plus], eps, None)
    mean_S = float(np.mean(S_plus))

    def gamma_nll_alpha(alpha):
        if alpha <= 0:
            return np.inf
        gamma = mean_S / alpha
        nll = sp.gammaln(alpha) + alpha * np.log(gamma) - (alpha - 1.0) * np.log(S_plus) + (S_plus / gamma)
        return float(np.sum(nll))

    res_alpha = minimize_scalar(gamma_nll_alpha, bounds=alpha_bounds, method="bounded")
    if not res_alpha.success or (not np.isfinite(res_alpha.x)):
        raise RuntimeError("Gamma alpha optimization failed in optimize_parameter().")

    alpha_hat = float(res_alpha.x)
    gamma_hat = float(np.clip(mean_S / alpha_hat, eps, None))

    # ---------- Step 2: Copula (theta) optimization ----------
    def copula_nll_theta(theta):
        if theta <= 0:
            return np.inf
        return L_clayton(N, S, lambda_hat, alpha_hat, gamma_hat, theta, eps=eps)

    res_theta = minimize_scalar(copula_nll_theta, bounds=theta_bounds, method="bounded")
    if not res_theta.success or (not np.isfinite(res_theta.x)):
        raise RuntimeError("Copula theta optimization failed in optimize_parameter().")

    theta_hat = float(res_theta.x)
    return np.array([lambda_hat, alpha_hat, gamma_hat, theta_hat], dtype=float)


# === 3. Define Custom Decision Tree ===
class ClaytonCopulaTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None,
                 value=None, n_samples=None, depth=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.depth = depth

class ClaytonCopulaTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_jobs=-1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.root = None

    def fit(self, X, Y):
        self.root = self._build_tree(X, Y, depth=0)

    def _build_tree(self, X, Y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            leaf_value = np.mean(Y, axis=0)
            return ClaytonCopulaTreeNode(value=leaf_value, n_samples=int(n_samples), depth=int(depth))

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
                param_left = optimize_parameter(Y_left)
                param_right = optimize_parameter(Y_right)

                loss_left = L_joint(Y_left[:, 0], Y_left[:, 1],
                                       param_left[0], param_left[1], param_left[2], param_left[3])
                loss_right = L_joint(Y_right[:, 0], Y_right[:, 1],
                                        param_right[0], param_right[1], param_right[2], param_right[3])
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
            leaf_value = np.mean(Y, axis=0)
            return ClaytonCopulaTreeNode(value=leaf_value, n_samples=int(n_samples), depth=int(depth))

        best = min(splits, key=lambda s: s['loss'])
        left = self._build_tree(X[best['left_data']], Y[best['left_data']], depth + 1)
        right = self._build_tree(X[best['right_data']], Y[best['right_data']], depth + 1)

        return ClaytonCopulaTreeNode(best['feature_index'], best['threshold'], left, right,
                                     n_samples=int(n_samples), depth=int(depth))

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
        def rd(v):
            try: return round(float(v), ndigits)
            except: return v
        if node is None:
            return None

        base = {
            "depth": node.depth,
            "n_samples": node.n_samples,
        }

        if node.value is not None:
            n_pred = float(node.value[0])
            s_pred = float(node.value[1])
            return {
                **base,
                "type": "leaf",
                "pred": {
                    "N_pred": rd(n_pred),
                    "S_pred": rd(s_pred),
                    "Y_pred": rd(n_pred * s_pred)
                }
            }
        else:
            feat_idx = int(node.feature_index)
            feat_name = None
            if feature_names is not None and 0 <= feat_idx < len(feature_names):
                feat_name = str(feature_names[feat_idx])
            return {
                **base,
                "type": "split",
                "feature_index": feat_idx,
                "feature_name": feat_name,
                "threshold": rd(node.threshold),
                "left": self._node_to_dict(node.left, feature_names, ndigits),
                "right": self._node_to_dict(node.right, feature_names, ndigits),
            }

    def export_json(self, feature_names=None, ndigits=6, indent=2, ensure_ascii=False):
        d = self._node_to_dict(self.root, feature_names=feature_names, ndigits=ndigits)
        return json.dumps(d, indent=indent, ensure_ascii=ensure_ascii)

    def save_json(self, filepath, feature_names=None, ndigits=6, indent=2, ensure_ascii=False):
        j = self.export_json(feature_names=feature_names, ndigits=ndigits, indent=indent, ensure_ascii=ensure_ascii)
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(j, encoding="utf-8")
        return str(p)