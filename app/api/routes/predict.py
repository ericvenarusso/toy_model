import numpy as np
import pandas as pd

from fastapi import APIRouter

from app.core.machine_learning.predictor import predict
from app.api.schemas.passengers import TitanicPassengers


router = APIRouter()

@router.post("/predict")
def post_predict(input_data: TitanicPassengers):
    data = np.array([[input_data.age, input_data.sex, input_data.pclass]]) 
    prediction = predict(data)
    return {'prediction': prediction}
