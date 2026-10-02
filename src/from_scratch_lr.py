"""
Logistic regression by batch gradient descent, implemented from first principles.

L2-regularised cross-entropy with the intercept excluded from the penalty
(standard practice: only the feature weights are shrunk, not the baseline
log-odds). Features are expected to be standardised by the caller; WoE values
are already on a comparable scale, but gradient descent converges far more
reliably on zero-mean, unit-variance inputs.

After fit, n_iter_ holds the number of iterations run, converged_ whether the
change in cost fell below tol before n_iter ran out, and final_cost_change_ the
last change in cost. A fit that stops on the iteration limit issues a
RuntimeWarning instead of passing silently.
"""
import warnings

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class FromScratchLogisticRegression:
    def __init__(self, lr: float = 0.5, n_iter: int = 3000, l2: float = 1e-4,
                 tol: float = 1e-10, verbose: bool = False):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.coef_ = None       # shape (n_features,)
        self.intercept_ = None  # scalar
        self.cost_history_ = []
        self.n_iter_ = None
        self.converged_ = None
        self.final_cost_change_ = None

    def _cost(self, X, y, w, b, sw):
        z = X @ w + b
        p = _sigmoid(z)
        eps = 1e-12
        ce = -np.sum(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) / np.sum(sw)
        reg = (self.l2 / (2 * len(y))) * np.sum(w ** 2)
        return ce + reg

    def fit(self, X: np.ndarray, y: np.ndarray, class_weight: str | None = None
            ) -> "FromScratchLogisticRegression":
        """class_weight="balanced" matches sklearn's convention exactly:
        weight_c = n_samples / (n_classes * count_c) -- otherwise a model
        trained on an 8%/92% imbalanced target just learns to lean toward
        the majority class, and any comparison against a sklearn fit that
        *does* use class_weight="balanced" is comparing two different
        optimization objectives, not two solvers of the same problem.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        if class_weight == "balanced":
            n_pos = y.sum()
            n_neg = n - n_pos
            w_pos = n / (2 * n_pos)
            w_neg = n / (2 * n_neg)
            sample_weight = np.where(y == 1, w_pos, w_neg)
        else:
            sample_weight = np.ones(n)

        w = np.zeros(d)
        b = 0.0
        prev_cost = np.inf
        sw_sum = sample_weight.sum()
        self.cost_history_ = []  # a refit starts a fresh history
        converged = False
        i = -1

        for i in range(self.n_iter):
            z = X @ w + b
            p = _sigmoid(z)
            error = sample_weight * (p - y)

            grad_w = (X.T @ error) / sw_sum + (self.l2 / n) * w
            grad_b = error.sum() / sw_sum

            w -= self.lr * grad_w
            b -= self.lr * grad_b

            cost = self._cost(X, y, w, b, sample_weight)
            self.cost_history_.append(cost)

            if abs(prev_cost - cost) < self.tol:
                converged = True
                if self.verbose:
                    print(f"converged at iter {i}, cost={cost:.6f}")
                break
            prev_cost = cost

        self.coef_ = w
        self.intercept_ = b
        self.n_iter_ = i + 1
        self.converged_ = converged
        hist = self.cost_history_
        self.final_cost_change_ = float(abs(hist[-2] - hist[-1])) if len(hist) > 1 else None
        if not converged:
            warnings.warn(
                f"gradient descent stopped at the iteration limit ({self.n_iter}) before the "
                f"change in cost fell below tol={self.tol:g}; last change {self.final_cost_change_:.3g}",
                RuntimeWarning, stacklevel=2,
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _sigmoid(X @ self.coef_ + self.intercept_)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    X, y = make_classification(
        n_samples=8000, n_features=8, n_informative=6, n_redundant=0,
        class_sep=1.2, random_state=0,
    )
    X = StandardScaler().fit_transform(X)

    # sklearn's C is inverse regularization strength on the *sum* loss by
    # default (not mean) -- match conventions by using an unregularized
    # sklearn fit against a near-zero l2 here, since the point is to check
    # the optimization is correct, not to match a specific penalty strength.
    sk = LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)
    scratch = FromScratchLogisticRegression(lr=0.5, n_iter=5000, l2=1e-6).fit(X, y)

    sk_pred = sk.predict_proba(X)[:, 1]
    scratch_pred = scratch.predict_proba(X)

    sk_auc = roc_auc_score(y, sk_pred)
    scratch_auc = roc_auc_score(y, scratch_pred)
    coef_diff = np.max(np.abs(sk.coef_.ravel() - scratch.coef_))
    pred_corr = np.corrcoef(sk_pred, scratch_pred)[0, 1]

    print(f"sklearn AUC : {sk_auc:.6f}")
    print(f"scratch AUC : {scratch_auc:.6f}")
    print(f"max |coef diff|: {coef_diff:.4f}")
    print(f"prediction correlation: {pred_corr:.6f}")

    assert abs(sk_auc - scratch_auc) < 1e-3, "AUC should match sklearn closely on a clean, low-collinearity set"
    assert coef_diff < 0.05, "coefficients should be tight on a small, low-collinearity feature set"
    print("validation passed: from-scratch LR matches sklearn on a clean feature set")
