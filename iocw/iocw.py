from copy import deepcopy
from typing import Optional
import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class IOCW(BaseEstimator, RegressorMixin):
    """Iterative Outlier Censoring Wrapper"""

    def __init__(
        self,
        estimator: BaseEstimator,
        threshold: Optional[float],
        tol: float = 1e-4,
        n_iters: int = 100,
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.tol = tol
        self.n_iters = n_iters

        self.y_out = None

    def _soft_threshold(self, values):
        if self.threshold is None:
            abs_res = np.abs(values)
            mad = np.median(abs_res)
            sigma = mad / 0.6745
            lam = sigma * np.sqrt(2 * np.log(values.size))
        else:
            lam = self.threshold

        E = np.sign(values) * np.maximum(np.abs(values) - lam, 0)
        return E

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.fitted_estimator_ = clone(self.estimator)

        # Enable warm start if the underlying model supports it for speed
        if hasattr(self.fitted_estimator_, "warm_start"):
            self.fitted_estimator_.warm_start = True

        y_clean = np.copy(y)
        old_norm = 0.0

        for j in range(self.n_iters):
            self.fitted_estimator_.fit(X, y_clean)
            y_hat = self.fitted_estimator_.predict(X)

            residuals = y - y_hat
            self.y_out_ = self._soft_threshold(residuals)
            y_clean = y - self.y_out_

            norm = np.linalg.norm(self.y_out_)
            # Avoid division by zero and handle first iteration
            denom = norm + 1e-9
            diff = np.abs(old_norm - norm) / denom

            if j > 0 and diff < self.tol:
                self.n_iter_converged_ = j
                break
            old_norm = norm
        else:
            self.n_iter_converged_ = self.n_iters

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.fitted_estimator_.predict(X)
