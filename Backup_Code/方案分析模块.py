# === * Library Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr


# === 1. Result: Statistical indicators ===
def prediction_metrics(y_true, y_pred, power=1.7, metrics_to_include=None):
    eps = 1e-8
    results = {}

    if metrics_to_include is None:
        metrics_to_include = ['RMSE', 'MAE', 'R2', 'MPE', 'MAPE', 'CCC', 'Gini', 'TweedieDev']

    if 'RMSE' in metrics_to_include:
        results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    if 'MAE' in metrics_to_include:
        results['MAE'] = mean_absolute_error(y_true, y_pred)
    if 'R2' in metrics_to_include:
        results['R2'] = r2_score(y_true, y_pred)
    if 'MPE' in metrics_to_include:
        results['MPE'] = np.mean((y_pred - y_true) / (y_true + eps))
    if 'MAPE' in metrics_to_include:
        results['MAPE'] = np.mean(np.abs((y_pred - y_true) / (y_true + eps)))
    if 'CCC' in metrics_to_include:
        results['CCC'] = 2 * np.corrcoef(y_pred, y_true)[0, 1] * np.std(y_pred) * np.std(y_true) / (np.var(y_pred) + np.var(y_true) + (np.mean(y_pred) - np.mean(y_true))**2 + eps)
    if 'Gini' in metrics_to_include:
        results['Gini'] = 2 * spearmanr(y_pred, y_true)[0] - 1
    if 'PoissonDeviance' in metrics_to_include:
        y_true_safe = np.maximum(y_true, eps)
        y_pred_safe = np.maximum(y_pred, eps)
        results['PoissonDeviance'] = 2 * np.sum(y_true * np.log(y_true_safe / y_pred_safe) - y_true + y_pred)
    if 'GammaDeviance' in metrics_to_include:
        y_true_safe = np.maximum(y_true, eps)
        y_pred_safe = np.maximum(y_pred, eps)
        results['GammaDeviance'] = 2 * np.sum(-np.log(y_true_safe / y_pred_safe) + (y_true - y_pred) / y_pred_safe)
    if 'TweedieDev' in metrics_to_include:
        y_true_safe = np.maximum(y_true, eps)
        y_pred_safe = np.maximum(y_pred, eps)
        tweedie_dev = 2 * (np.maximum(y_true, 0)**(2 - power) / ((1 - power)*(2 - power)) - y_true_safe * y_pred_safe**(1 - power) / (1 - power) + y_pred_safe**(2 - power) / (2 - power))
        results['TweedieDev'] = np.sum(tweedie_dev)
    return results

metrics_dict = {
    'N': ['MAE', 'Gini', 'PoissonDeviance'],
    'S': ['RMSE', 'MAE', 'MPE', 'MAPE', 'CCC', 'Gini', 'GammaDeviance'],
    'Y': ['RMSE', 'MAE', 'R2', 'MPE', 'MAPE', 'CCC', 'Gini', 'TweedieDev']
}

results = {
    'SSETree': {
        'N': prediction_metrics(N_test, N_pred_sse, metrics_to_include=metrics_dict['N']),
        'S': prediction_metrics(S_test, S_pred_sse, metrics_to_include=metrics_dict['S']),
        'Y': prediction_metrics(Y_test, Y_pred_sse, metrics_to_include=metrics_dict['Y'])
    },
    'CHDLTree': {
        'N': prediction_metrics(N_test, N_pred_chdl, metrics_to_include=metrics_dict['N']),
        'S': prediction_metrics(S_test, S_pred_chdl, metrics_to_include=metrics_dict['S']),
        'Y': prediction_metrics(Y_test, Y_pred_chdl, metrics_to_include=metrics_dict['Y'])
    },
    'PGNLLTree': {
        'N': prediction_metrics(N_test, N_pred_pgnll, metrics_to_include=metrics_dict['N']),
        'S': prediction_metrics(S_test, S_pred_pgnll, metrics_to_include=metrics_dict['S']),
        'Y': prediction_metrics(Y_test, Y_pred_pgnll, metrics_to_include=metrics_dict['Y'])
    },
    'CopulaTree': {
        'N': prediction_metrics(N_test, N_pred_copula, metrics_to_include=metrics_dict['N']),
        'S': prediction_metrics(S_test, S_pred_copula, metrics_to_include=metrics_dict['S']),
        'Y': prediction_metrics(Y_test, Y_pred_copula, metrics_to_include=metrics_dict['Y'])
    }
}

results_df = pd.DataFrame({
    (model, target): results[model][target]
    for model in results
    for target in results[model]
}).T

results_df.index.names = ['Model', 'Target']
results_df = results_df.reset_index()
print(results_df)


# === 2. Result: Visualization ===
# heat map
all_metrics = ['RMSE', 'MAE', 'R2', 'MPE', 'MAPE', 'CCC', 'Gini', 'PoissonDeviance', 'GammaDeviance', 'TweedieDev']
high_metrics = ['R2', 'CCC', 'Gini']
low_metrics = ['RMSE', 'MAE', 'MAPE', 'MPE', 'PoissonDeviance', 'GammaDeviance', 'TweedieDev']

def normalize_column(col, inverse=False):
    arr = col.copy()
    if inverse:
        arr = 1 / (np.abs(arr) + 1e-8)
    scaled = (arr - arr.min()) / (arr.max() - arr.min()) * 100
    return scaled

for metric in all_metrics:
    if metric not in results_df.columns:
        continue
    heat_df = results_df.copy()
    heat_df[metric] = normalize_column(heat_df[metric], inverse=(metric in low_metrics))
    pivot_data = heat_df.pivot(index='Model', columns='Target', values=metric)

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        pivot_data,
        annot=results_df.pivot(index='Model', columns='Target', values=metric).round(3),
        cmap="RdBu", fmt=".2f", linewidths=.5,
        cbar_kws={"label": "Scaled Score"}
    )
    plt.title(f"Tree Performance Heatmap: {metric}")
    plt.tight_layout()
    plt.show()

# density map
def build_density_df(y_true, pred_dict, log_transform=True):
    df = pd.DataFrame({'Actual': y_true})
    for name, pred in pred_dict.items():
        df[name] = pred
    if log_transform:
        df = np.log1p(df)
    df_melted = df.melt(var_name='Model', value_name='Log(Y+1)')
    return df_melted

target_list = ['N', 'S', 'Y']
true_dict = {'N': N_test, 'S': S_test, 'Y': Y_test}
pred_dicts = {
    'N': {
        'SSETree': N_pred_sse,
        'CHDLTree': N_pred_chdl,
        'PGNLLTree': N_pred_pgnll,
        'CopulaTree': N_pred_copula
    },
    'S': {
        'SSETree': S_pred_sse,
        'CHDLTree': S_pred_chdl,
        'PGNLLTree': S_pred_pgnll,
        'CopulaTree': S_pred_copula
    },
    'Y': {
        'SSETree': Y_pred_sse,
        'CHDLTree': Y_pred_chdl,
        'PGNLLTree': Y_pred_pgnll,
        'CopulaTree': Y_pred_copula
    }
}

for target in target_list:
    density_df = build_density_df(true_dict[target], pred_dicts[target], log_transform=True)
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=density_df, x="Log(Y+1)", hue="Model", fill=True, common_norm=False, alpha=0.4)
    plt.title(f"{target}: Log(Predicted + 1) vs Actual Density")
    plt.tight_layout()
    plt.show()
