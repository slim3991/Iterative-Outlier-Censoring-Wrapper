import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import check_estimator
from iocw import IOCW


## 1. Basic Functionality & Outlier Resilience
def test_outlier_resilience():
    """Verify that RobustThing is less influenced by a huge outlier than LinearRegression."""
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.flatten() + 1 + np.random.normal(0, 0.1, 100)

    # Introduce a massive outlier
    y[50] += 1000

    lr = LinearRegression()
    rt = IOCW(LinearRegression(), threshold=1.0)

    lr.fit(X, y)
    rt.fit(X, y)

    # The slope should be close to 2.0.
    # Linear Regression will be pulled away; RobustThing should stay steady.
    lr_slope = lr.coef_[0]
    rt_slope = rt.fitted_estimator_.coef_[0]

    assert abs(rt_slope - 2.0) < abs(lr_slope - 2.0)
    assert np.abs(rt.y_out_[50]) > 900  # It should have identified the outlier


## 2. Automatic Thresholding
def test_automatic_threshold():
    """Verify the code doesn't crash when threshold is None."""
    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    rt = IOCW(LinearRegression(), threshold=None)
    rt.fit(X, y)

    assert hasattr(rt, "y_out_")
    assert rt.n_iter_converged_ > 0


## 3. Convergence Logic
def test_convergence_early_stop():
    """Check if it stops early on clean data."""
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X.flatten()  # Perfectly linear, no noise

    rt = IOCW(LinearRegression(), threshold=0.1, n_iters=100)
    rt.fit(X, y)

    # On perfectly clean data, it should converge very quickly
    assert rt.n_iter_converged_ < 10


## 4. Prediction Consistency
def test_predict_shape():
    """Ensure predict returns the correct dimensions."""
    X = np.random.rand(10, 1)
    y = np.random.rand(10)

    rt = IOCW(LinearRegression(), threshold=0.5)
    rt.fit(X, y)

    y_pred = rt.predict(X)
    assert y_pred.shape == (10,)
