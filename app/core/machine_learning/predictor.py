import joblib

from app.core.config import MODEL_FILE_NAME


def _load_model():
    return joblib.load(MODEL_FILE_NAME)

def predict(data):
    model = _load_model()
    prediction = int(model.predict(data)[0])
    return prediction
