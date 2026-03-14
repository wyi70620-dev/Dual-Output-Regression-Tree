import numpy as np
from scipy.optimize import minimize


# === function ===
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


# === experiment1 ===
Y = np.array([[5,4.8],[5,5.1],[5,5.0],[5,5.2],[5,4.9]], dtype=float)

print("Nelder–Mead result:", optimize_leaf(Y, sigma1=1.0, sigma2=1.0, delta=1.0))
print("Expected result:", np.array([5.0, 5.0]))


# === experiment2 ===
Y = np.array([[5.1,4.8],[5.2,5.1],[4.8,5.0],[4.9,5.2],[5,4.9]], dtype=float)

print("Nelder–Mead result:", optimize_leaf(Y, sigma1=1.0, sigma2=1.0, delta=1.0))
print("Expected result:", np.array([4.9980, 5.0]))


# === experiment3 ===
Y = np.array([[5,4.5],[5,5.1],[5,5.0],[5,4.2],[5,6.9]], dtype=float)

print("Nelder–Mead result:", optimize_leaf(Y, sigma1=1.0, sigma2=1.0, delta=0.5))
print("Expected result:", np.array([5.0, 4.8666]))


# === experiment4 ===
Y = np.array([[5.1,4.5],[5.2,5.1],[4.8,5.0],[4.9,4.2],[5,6.9]], dtype=float)

print("Nelder–Mead result:", optimize_leaf(Y, sigma1=0.25, sigma2=0.75, delta=0.5))
print("Expected result:", np.array([4.9980, 4.8667]))


# === experiment5 ===
Y = np.array([[9.8,100],[10.1,102],[10.0,98],[9.6,150],[15.0,95],[7.0,400]], dtype=float)

print("Nelder–Mead result:", optimize_leaf(Y, sigma1=5, sigma2=0.75, delta=5.0))
print("Expected result:", np.array([9.9843, 101.6667]))
