import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
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


def test_pipeline_compatibility():
    """Ensure IOCW works inside a scikit-learn Pipeline."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("robust_reg", IOCW(LinearRegression(), threshold=1.0)),
        ]
    )

    X = np.random.rand(30, 3)
    y = np.random.rand(30)

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (30,)


@pytest.mark.parametrize(
    "estimator",
    [
        SVR(kernel="rbf", C=100),
        HistGradientBoostingRegressor(max_iter=20),
        KNeighborsRegressor(n_neighbors=3),
    ],
)
def test_complex_estimators(estimator):
    """Test IOCW with non-linear and ensemble models."""
    X = np.sort(np.random.rand(40, 1) * 10, axis=0)
    y = np.sin(X).flatten() + np.random.normal(0, 0.05, 40)

    # Spike a middle value
    y[20] += 10.0

    rt = IOCW(estimator, threshold=0.5)
    rt.fit(X, y)

    y_pred = rt.predict(X)
    # The prediction at the outlier index should still be relatively close
    # to the sine wave (approx 0.9 at X=5) rather than 10.9
    assert abs(y_pred[20] - np.sin(X[20])) < 5.0


@pytest.mark.parametrize("base_model", [Ridge(), DecisionTreeRegressor(max_depth=5)])
def test_different_base_estimators(base_model):
    """Verify IOCW works with various sklearn regressor types."""
    X = np.random.rand(30, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.05, 30)

    # Introduce one outlier
    y[0] += 50.0

    rt = IOCW(base_model, threshold=1.0)
    rt.fit(X, y)

    assert hasattr(rt, "fitted_estimator_")
    assert rt.predict(X).shape == (30,)


def test_heavy_noise_convergence():
    """Verify stability under high-variance/low-signal data."""
    X = np.random.rand(50, 1)
    y = np.random.normal(0, 100, 50)  # Pure noise

    rt = IOCW(LinearRegression(), threshold=1.0, n_iters=10)
    rt.fit(X, y)

    assert rt.n_iter_converged_ <= 10
    assert not np.isnan(rt.y_out_).any()


def test_invalid_input_shapes():
    """Verify it raises errors on mismatched X and y."""
    X = np.random.rand(10, 2)
    y = np.random.rand(5)  # Mismatched length

    rt = IOCW(LinearRegression(), threshold=0.1)
    with pytest.raises(ValueError):
        rt.fit(X, y)


def test_unfitted_predict():
    """Ensure predict raises NotFittedError if fit hasn't been called."""
    rt = IOCW(LinearRegression(), threshold=0.1)
    with pytest.raises(Exception):  # sklearn.exceptions.NotFittedError
        rt.predict(np.random.rand(5, 2))
