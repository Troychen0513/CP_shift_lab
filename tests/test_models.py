import numpy as np
import pytest

from src.models import PolyModel


def test_poly_model_predict_after_fit():
    x = np.array([-1.0, 0.0, 1.0])
    y = 2 * x + 1

    model = PolyModel(degree=1)
    model.fit(x, y)

    pred = model.predict(x)

    np.testing.assert_allclose(pred, y)


def test_poly_model_predict_before_fit_raises_error():
    model = PolyModel(degree=1)

    with pytest.raises(ValueError):
        model.predict(np.array([0.0, 1.0]))