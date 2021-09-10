from datetime import datetime
from elasticsearch import Elasticsearch

from app.models import TitanicPassengers


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