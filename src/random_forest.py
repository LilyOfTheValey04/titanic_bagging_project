import numpy as np
from src.decision_tree import DecisionTree

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _select_features(self, n_features):
        if self.max_features == 'sqrt':
            k = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            k = int(np.log2(n_features))
        else:
            k = n_features
        return np.random.choice(n_features, k, replace=False)

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]
            # Feature subset
            feature_idxs = self._select_features(n_features)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample[:, feature_idxs], y_sample)
            self.trees.append((tree, feature_idxs))

    def predict(self, X):
        predictions = []
        for tree, feature_idxs in self.trees:
            pred = tree.predict(X[:, feature_idxs])
            predictions.append(pred)
        predictions = np.array(predictions)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
