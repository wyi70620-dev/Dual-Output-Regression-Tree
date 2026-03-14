from sklearn.tree import export_text
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# 我自己的树
class ParallelRegressionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class ParallelRegressionTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_jobs=-1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            return ParallelRegressionTreeNode(value=np.mean(y))

        def evaluate_split(feature_index):
            # Step 1: 排序该特征值
            sorted_idx = np.argsort(X[:, feature_index])
            X_sorted = X[sorted_idx, feature_index]

            # Step 2: 计算相邻中点作为候选阈值
            thresholds = (X_sorted[:-1] + X_sorted[1:]) / 2

            best_mse = float('inf')
            best_split = None

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                mse = (len(y_left) * np.var(y_left) + len(y_right) * np.var(y_right)) / n_samples

                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask,
                        'mse': mse
                    }
            return best_split

        splits = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_split)(feature_index) for feature_index in range(n_features)
        )
        splits = [s for s in splits if s is not None]
        if not splits:
            return ParallelRegressionTreeNode(value=np.mean(y))

        best_split = min(splits, key=lambda x: x['mse'])

        left_subtree = self._build_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_subtree = self._build_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return ParallelRegressionTreeNode(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            left=left_subtree,
            right=right_subtree
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

def print_tree(node, depth=0):
    indent = "  " * depth
    if node.value is not None:
        print(f"{indent}Leaf: Predict value = {node.value:.2f}")
    else:
        print(f"{indent}Node: X[{node.feature_index}] <= {node.threshold}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)


# === 1. 数据生成 ===
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# === 2. 模型拟合 ===
sk_tree = DecisionTreeRegressor(max_depth=5)
sk_tree.fit(X, y)

my_tree = ParallelRegressionTree(max_depth=5, min_samples_split=2, n_jobs=-1)
my_tree.fit(X, y)

# === 3. 预测 ===
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_sk = sk_tree.predict(X_test)
y_my = my_tree.predict(X_test)

# === 4. 可视化对比 ===
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_sk, color="cornflowerblue", label="sklearn_tree", linewidth=2)
plt.plot(X_test, y_my, color="yellowgreen", label="my_tree", linestyle="--", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Comparison: sklearn vs. My Regression Tree")
plt.legend()
plt.show()

# === 5. 打印你自己的树结构 ===
print("\n==== My Tree Structure ====")
print_tree(my_tree.root)

# === 6. 打印 sklearn 树结构（使用 export_text） ===
print("\n==== Sklearn Tree Structure ====")
print(export_text(sk_tree, feature_names=["X[0]"]))

# === 7. 预测值比较 ===
print("\n==== 预测值差异 ====")
for i in range(0, len(X_test), 50):  # 每隔 50 个点展示一次
    print(f"X = {X_test[i][0]:.2f} | Sklearn = {y_sk[i]:.2f} | MyTree = {y_my[i]:.2f} | Diff = {abs(y_sk[i] - y_my[i]):.4f}")

# === 8. 误差分析（整体）===
diff = np.abs(y_sk - y_my)
print("\n平均绝对差值（MAE）: {:.4f}".format(np.mean(diff)))
print("最大绝对差值: {:.4f}".format(np.max(diff)))
