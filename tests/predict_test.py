import pytest

import numpy as np
import pandas as pd

from src.ml.scoring import predict

def test_predict_return():
    data = pd.DataFrame(
        {
            "Pclass": [3, 1, 3],
            "Sex": ["male", "female", "female"],
            "Fare": [7.2500, 71.2833, 7.9250]
        }
    )
 
    result = predict(data)

    assert isinstance(result, np.ndarray)
    assert len(result) != 0
    assert isinstance(result[0], np.int64)

