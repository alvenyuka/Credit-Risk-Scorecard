"""Logistic regression implemented from scratch: sigmoid, cross-entropy loss, batch gradient descent, L2."""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow in exp() for very negative/positive z.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, l2: float) -> float:
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    ce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    reg = (l2 / (2 * len(y_true))) * np.sum(weights[1:] ** 2)  # exclude bias from regularization
    return ce + reg


class FromScratchLogisticRegression:
    """
    Batch gradient descent logistic regression with L2 regularization.

    Matches sklearn.linear_model.LogisticRegression's convention: features should
    be standardized by the caller (this class does not scale internally).
    """

    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 2000,
                 l2: float = 1.0, tol: float = 1e-7, verbose: bool = False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.weights_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FromScratchLogisticRegression":
        n_samples, n_features = X.shape
        X_b = np.hstack([np.ones((n_samples, 1)), X])  # bias column
        w = np.zeros(n_features + 1)

        prev_loss = np.inf
        for i in range(self.n_iterations):
            z = X_b @ w
            y_pred = sigmoid(z)

            grad = (X_b.T @ (y_pred - y)) / n_samples
            grad[1:] += (self.l2 / n_samples) * w[1:]  # L2 term, bias excluded

            w -= self.learning_rate * grad

            loss = cross_entropy_loss(y, y_pred, w, self.l2)
            self.loss_history_.append(loss)

            if self.verbose and i % 200 == 0:
                print(f"  iter {i:5d}  loss={loss:.6f}")

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        self.weights_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_b = np.hstack([np.ones((n_samples, 1)), X])
        p1 = sigmoid(X_b @ self.weights_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def coef_(self) -> np.ndarray:
        return self.weights_[1:]

    @property
    def intercept_(self) -> float:
        return self.weights_[0]
