import pytest

import numpy as np

from app.ml.predictor import predict

def test_predict_return():
    data = np.array([[3, "male", 1]])
    result = predict(data)
    
    assert isinstance(result, int)

