import numpy as np
from datetime import datetime

from fastapi import APIRouter
from elasticsearch import Elasticsearch

from app.core.machine_learning.predictor import predict
from app.api.schemas.passengers import TitanicPassengers


router = APIRouter()

def log_to_els(input_data: TitanicPassengers, predict_result: int) -> None:
    """
        Log inputs and prediction to ElasticSearch.
    """
    log_body = {
        'age': input_data.age,
        'sex': input_data.sex,
        'pclass': input_data.pclass,
        'predict': predict_result,
        'timestamp': datetime.now()
    }
    es = Elasticsearch(
        hosts=[{"host": "host.docker.internal", "port": 9200}],
        retry_on_timeout=True
    )
    es.index(f"titanic-api-post-predict-{str(datetime.now().date())}", body=log_body)

@router.post("/predict")
def post_predict(input_data: TitanicPassengers) -> int:
    """
        Predict if you would survive the sinking of the Titanic.
    """
    data = np.array([[input_data.age, input_data.sex, input_data.pclass]]) 
    predict_result = predict(data)

    log_to_els(input_data, predict_result)

    return predict_result
