import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import APIRouter
from elasticsearch import Elasticsearch

from app.core.machine_learning.predictor import predict
from app.api.schemas.passengers import TitanicPassengers


router = APIRouter()

@router.post("/predict")
def post_predict(input_data: TitanicPassengers):
    data = np.array([[input_data.age, input_data.sex, input_data.pclass]]) 
    predict_result = predict(data)

    log_body = {
        'age': input_data.age,
        'sex': input_data.sex,
        'pclass': input_data.pclass,
        'timestamp': datetime.now()
    }
    es = Elasticsearch("http://elasticsearch:9200")
    es.index(f"titanic_api_post_predict_{str(datetime.now().date())}", body=log_body)

    return predict_result
