import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class ParallelRegressionTreeNode2D:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  

class ParallelRegressionTree2D:
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
            return ParallelRegressionTreeNode2D(value=np.mean(Y, axis=0))

        def evaluate_split(feature_index):
            sorted_idx = np.argsort(X[:, feature_index])
            X_sorted = X[sorted_idx, feature_index]
            thresholds = (X_sorted[:-1] + X_sorted[1:]) / 2

            best_mse = float('inf')
            best_split = None

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                Y_left, Y_right = Y[left_mask], Y[right_mask]
                mse = (
                    len(Y_left) * (np.var(Y_left[:, 0]) + np.var(Y_left[:, 1])) +
                    len(Y_right) * (np.var(Y_right[:, 0]) + np.var(Y_right[:, 1]))
                ) / (2 * n_samples)

                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
            return best_split

        splits = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_split)(i) for i in range(n_features)
        )
        splits = [s for s in splits if s is not None]
        if not splits:
            return ParallelRegressionTreeNode2D(value=np.mean(Y, axis=0))

        best = min(splits, key=lambda s: (
            len(Y[s['left_mask']]) * (np.var(Y[s['left_mask']][:, 0]) + np.var(Y[s['left_mask']][:, 1])) +
            len(Y[s['right_mask']]) * (np.var(Y[s['right_mask']][:, 0]) + np.var(Y[s['right_mask']][:, 1]))
        ))

        left = self._build_tree(X[best['left_mask']], Y[best['left_mask']], depth + 1)
        right = self._build_tree(X[best['right_mask']], Y[best['right_mask']], depth + 1)

        return ParallelRegressionTreeNode2D(best['feature_index'], best['threshold'], left, right)

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

# --- 打印树结构 ---
def print_tree_2d_named(node, depth=0):
    indent = "  " * depth
    if node.value is not None:
        print(f"{indent}Leaf: Predict N = {node.value[0]:.2f}, S = {node.value[1]:.2f}")
    else:
        print(f"{indent}Node: X[{node.feature_index}] <= {node.threshold:.2f}")
        print_tree_2d_named(node.left, depth + 1)
        print_tree_2d_named(node.right, depth + 1)


# === 1. 数据生成 ===
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += 0.5 - rng.rand(20, 2)

# === 2. 模型拟合 ===
sk_tree = DecisionTreeRegressor(max_depth=8)
sk_tree.fit(X, y)

my_tree = ParallelRegressionTree2D(max_depth=8, min_samples_split=2, n_jobs=-1)
my_tree.fit(X, y)

# === 3. 预测 ===
X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
y_sk = sk_tree.predict(X_test)
y_my = my_tree.predict(X_test)

# === 4. 可视化对比 ===
plt.figure()
s = 25
plt.scatter(y[:, 0], y[:, 1], c="yellow", s=s, edgecolor="black", label="True Data")
plt.scatter(y_sk[:, 0], y_sk[:, 1], c="cornflowerblue", s=s, edgecolor="black", label="Sklearn Tree")
plt.scatter(y_my[:, 0], y_my[:, 1], c="red", s=s, edgecolor="black", label="My Parallel Tree")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1 (N)")
plt.ylabel("target 2 (S)")
plt.title("Multi-output Decision Tree Regression")
plt.legend(loc="best")
plt.show()

# === 7. 预测值比较 ===
print("\n==== 预测值差异 ====")
for i in range(0, len(X_test), 1000):  # 可以每隔一些点打印，避免输出过长
    print(f"X = {X_test[i][0]:.2f} | "
          f"Sklearn = ({y_sk[i][0]:.2f}, {y_sk[i][1]:.2f}) | "
          f"MyTree = ({y_my[i][0]:.2f}, {y_my[i][1]:.2f}) | "
          f"Diff = ({abs(y_sk[i][0] - y_my[i][0]):.4f}, {abs(y_sk[i][1] - y_my[i][1]):.4f})")

# === 8. 误差分析（整体）===
diff = np.abs(y_sk - y_my)
print("\n平均绝对差值（MAE）: {:.4f}".format(np.mean(diff)))
print("最大绝对差值: {:.4f}".format(np.max(diff)))
