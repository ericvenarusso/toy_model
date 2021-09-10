import numpy as np

from fastapi import FastAPI

from app.log import log_to_els
from app.ml.predictor import predict
from app.models import TitanicPassengers


app = FastAPI(
    title="Titanic API",
    version="1.0",
    description="Would you survive the sinking of the Titanic?"
)

@app.get("/healthcheck", tags=["health"])
async def healthcheck():
    """
        Checks the API Health.
    """
    return {"status": "alive"}

@app.post("/predict", tags=["model"])
def post_predict(input_data: TitanicPassengers) -> int:
    """
        Predict if you would survive the sinking of the Titanic.
    """
    data = np.array([[input_data.age, input_data.sex, input_data.pclass]]) 
    predict_result = predict(data)

    log_to_els(input_data, predict_result)

    return predict_result
