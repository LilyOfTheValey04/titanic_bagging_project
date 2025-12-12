import numpy as np
from src.decision_tree import DecisionTree

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=10, max_depth=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Bootstrap sample
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]
            tree = self.base_estimator(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
