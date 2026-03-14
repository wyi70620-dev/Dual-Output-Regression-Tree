import numpy as np
import pandas as pd
import json
from pathlib import Path
import optuna
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar, root_scalar
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import poisson, gamma as gamma_dist
import scipy.special as sp


# =============================================================================
# 1) Hyperparameter Tuning
# =============================================================================
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

    # 注意：你原代码 pruner=None，这里保持一致；如果你想启用剪枝改成 pruner=pruner
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=None)

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


# =============================================================================
# 2) Data Loading and preprocessing
# =============================================================================
train_df = pd.read_csv("~/Desktop/Training_data.csv")
x_cols = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10"]
y_cols = ["N","S"]
X_train = train_df[x_cols].to_numpy(dtype=float)
Y_train = train_df[y_cols].to_numpy(dtype=float)


# =============================================================================
# 3) Copula: Joint Loss & Parameter Optimization (Stable log-domain Clayton)
# =============================================================================

# ---- numeric constants ----
LOG_MIN = -745.0  # ~ log(np.finfo(float).tiny)
LOG_MAX =  709.0  # ~ log(np.finfo(float).max)

def _log_diff_exp(a, b, eps=1e-12):
    """
    Stable log(exp(a) - exp(b)) elementwise.
    If exp(a) <= exp(b) numerically or difference ~ 0, returns log(eps).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a2 = np.maximum(a, b)
    b2 = np.minimum(a, b)

    d = b2 - a2  # <= 0
    near = (np.abs(d) < 1e-14)

    x = np.exp(np.clip(d, LOG_MIN, 0.0))  # in (0,1]
    tiny = np.finfo(float).eps
    x = np.clip(x, 0.0, 1.0 - tiny)       # avoid x==1 -> log1p(-1)

    out = a2 + np.log1p(-x)
    out = np.where(near, np.log(eps), out)
    out = np.where(np.isfinite(out), out, np.log(eps))
    out = np.maximum(out, np.log(eps))
    return out


def _log_a_clayton(u, v, theta, eps_uv=1e-8):
    """
    log( u^(-theta) + v^(-theta) - 1 ) computed stably.
    u,v in (0,1). For Clayton, the inside is >= 1 (positive).
    """
    u = np.clip(u, eps_uv, 1.0 - eps_uv)
    v = np.clip(v, eps_uv, 1.0 - eps_uv)

    logu = np.log(u)
    logv = np.log(v)

    t1 = -theta * logu  # log(u^-theta)
    t2 = -theta * logv  # log(v^-theta)

    m = np.maximum(np.maximum(t1, t2), 0.0)

    e1 = np.exp(np.clip(t1 - m, LOG_MIN, 0.0))
    e2 = np.exp(np.clip(t2 - m, LOG_MIN, 0.0))
    e3 = np.exp(np.clip(-m,  LOG_MIN, 0.0))  # exp(0 - m)

    inside = e1 + e2 - e3
    inside = np.maximum(inside, eps_uv)

    return m + np.log(inside)


def _log_D1_clayton(u, v, theta, eps_uv=1e-8):
    """
    log D1(u,v;theta) where
    D1 = u^(-(theta+1)) * (u^(-theta) + v^(-theta) - 1)^(-(1/theta + 1))
    """
    u = np.clip(u, eps_uv, 1.0 - eps_uv)
    logu = np.log(u)
    loga = _log_a_clayton(u, v, theta, eps_uv=eps_uv)

    logD1 = -(theta + 1.0) * logu - (1.0/theta + 1.0) * loga
    return np.clip(logD1, LOG_MIN, LOG_MAX)


# === 1. Define Joint Loss Function ===
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
    S_plus  = np.clip(S_plus, eps, None)

    poisson_nll = lambda_ - N_plus * np.log(lambda_) + sp.gammaln(N_plus + 1.0)
    L_poisson = np.sum(poisson_nll)

    gamma_nll = sp.gammaln(alpha) + alpha * np.log(gamma) - (alpha - 1.0) * np.log(S_plus) + S_plus / gamma
    L_gamma = np.sum(gamma_nll)

    return float(L_poisson + L_gamma)


def L_copula(N, S, lambda_, alpha, gamma, theta, eps_uv=1e-8, eps_delta=1e-12):
    """
    Stable Clayton copula loss using log-domain.
    Returns: -sum log(Delta_i / fN)
    """
    N = np.asarray(N, dtype=int)
    S = np.asarray(S, dtype=float)

    I_plus = (N >= 1) & (S > 0.0)
    N_plus = N[I_plus]
    S_plus = S[I_plus]

    if N_plus.size == 0:
        return 0.0

    # u_i
    u_i = gamma_dist.cdf(S_plus, a=alpha, scale=gamma)
    u_i = np.clip(u_i, eps_uv, 1.0 - eps_uv)

    # truncated poisson mapping
    e0 = np.exp(-lambda_)
    denom = np.maximum(1.0 - e0, eps_uv)

    v_n   = (poisson.cdf(N_plus,     mu=lambda_) - e0) / denom
    v_nm1 = (poisson.cdf(N_plus - 1, mu=lambda_) - e0) / denom

    v_n   = np.clip(v_n,   eps_uv, 1.0 - eps_uv)
    v_nm1 = np.clip(v_nm1, eps_uv, 1.0 - eps_uv)

    # log D1
    logD1_n   = _log_D1_clayton(u_i, v_n,   theta, eps_uv=eps_uv)
    logD1_nm1 = _log_D1_clayton(u_i, v_nm1, theta, eps_uv=eps_uv)

    # log Delta = log( exp(logD1_n) - exp(logD1_nm1) )
    logDelta = _log_diff_exp(logD1_n, logD1_nm1, eps=eps_delta)

    # log fN
    logfN = poisson.logpmf(N_plus, mu=lambda_)
    logfN = np.clip(logfN, LOG_MIN, LOG_MAX)

    nll = -np.sum(logDelta - logfN)
    if not np.isfinite(nll):
        return float(np.inf)
    return float(nll)


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


def L_joint(N, S, lambda_, alpha, gamma, theta):
    return (
        L_PGNLL(N, S, lambda_, alpha, gamma)
        + L_copula(N, S, lambda_, alpha, gamma, theta)
        + L_indicator(N, S, lambda_)
    )


# === 2. Parameters Optimization Function ===
def optimize_parameter(
    Y,
    eps=1e-8,
    lambda_bounds=(1e-6, 100),
    alpha_bounds=(1e-3, 1e4),
    theta_bounds=(1e-3, 50.0),
    eps_uv=1e-8,
    eps_delta=1e-12,
):
    Y = np.asarray(Y)
    N = np.asarray(Y[:, 0], dtype=int)
    S = np.asarray(Y[:, 1], dtype=float)

    I_plus = (N >= 1) & (S > 0.0)
    I_0    = (N == 0) & (S == 0.0)

    # Pure zero node
    if np.sum(I_plus) == 0:
        lambda_hat = 0.0
        alpha_hat  = 1.0
        gamma_hat  = 0.0
        theta_hat  = float(theta_bounds[0])
        return np.array([lambda_hat, alpha_hat, gamma_hat, theta_hat], dtype=float)

    # ---------- Step 1: Marginal optimization ----------
    m = int(np.sum(I_plus))
    k = int(np.sum(I_0))
    T = float(np.sum(N[I_plus]))

    def poisson_score_lambda(lam):
        # same as your original function (kept)
        return (m + k) - T / lam - m / np.expm1(lam)

    result = root_scalar(poisson_score_lambda, bracket=lambda_bounds, method="brentq")
    lambda_hat = float(result.root)

    S_plus = np.clip(S[I_plus], eps, None)
    mean_S = float(np.mean(S_plus))

    def gamma_nll_alpha(alpha):
        if alpha <= 0:
            return np.inf
        scale = mean_S / alpha
        nll = sp.gammaln(alpha) + alpha * np.log(scale) - (alpha - 1.0) * np.log(S_plus) + (S_plus / scale)
        val = float(np.sum(nll))
        return val if np.isfinite(val) else np.inf

    res_alpha = minimize_scalar(gamma_nll_alpha, bounds=alpha_bounds, method="bounded")
    if (not res_alpha.success) or (not np.isfinite(res_alpha.x)):
        raise RuntimeError("Gamma alpha optimization failed in optimize_parameter().")

    alpha_hat = float(res_alpha.x)
    gamma_hat = float(np.clip(mean_S / alpha_hat, eps, None))

    # ---------- Step 2: Copula (theta) optimization ----------
    def copula_nll_theta(theta):
        if theta <= 0:
            return np.inf
        val = L_copula(N, S, lambda_hat, alpha_hat, gamma_hat, theta, eps_uv=eps_uv, eps_delta=eps_delta)
        return val if np.isfinite(val) else np.inf

    res_theta = minimize_scalar(copula_nll_theta, bounds=theta_bounds, method="bounded")
    if (not res_theta.success) or (not np.isfinite(res_theta.x)):
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

                # optimize params for each child
                param_left = optimize_parameter(Y_left)
                param_right = optimize_parameter(Y_right)

                loss_left = L_joint(
                    Y_left[:, 0], Y_left[:, 1],
                    param_left[0], param_left[1], param_left[2], param_left[3]
                )
                loss_right = L_joint(
                    Y_right[:, 0], Y_right[:, 1],
                    param_right[0], param_right[1], param_right[2], param_right[3]
                )
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

        return ClaytonCopulaTreeNode(
            best['feature_index'], best['threshold'], left, right,
            n_samples=int(n_samples), depth=int(depth)
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
        def rd(v):
            try:
                return round(float(v), ndigits)
            except Exception:
                return v

        if node is None:
            return None

        base = {"depth": node.depth, "n_samples": node.n_samples}

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


# =============================================================================
# 5) Model Training
# =============================================================================
copula_param_range = {
    "max_depth": [int(x) for x in range(1, 17)],
    "min_samples_split": [int(x) for x in range(2, 101)],
}

best_cop_params = run_optuna_tpe_search(
    X_train, Y_train,
    copula_param_range,
    n_jobs=1,
    model_class=ClaytonCopulaTree
)
print("Best Parameters (Copula Tree):", best_cop_params)

best_tree = ClaytonCopulaTree(**best_cop_params, n_jobs=-1)
best_tree.fit(X_train, Y_train)

desktop_json = Path.home() / "Desktop" / "copula_best_tree.json"
best_tree.save_json(desktop_json, feature_names=x_cols, ndigits=6, indent=2)

print(f"Tree JSON saved to: {desktop_json}")
