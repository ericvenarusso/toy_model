import joblib
import numpy as np

from app.core.config import MODEL_FILE_NAME


def _load_model():
    """
        Load the Sklearn Model.
    """
    return joblib.load(MODEL_FILE_NAME)

def predict(data: np.array) -> int:
    """"
        Score the data that was given with the Sklearn Model.
    """
    model = _load_model()
    prediction = int(model.predict(data)[0])
    return prediction
