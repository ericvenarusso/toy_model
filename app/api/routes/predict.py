import pandas as pd

from fastapi import APIRouter

from app.core.machine_learning.predictor import predict
from app.core.schemas.passengers import TitanicPassengers


router = APIRouter()

@router.post("/predict")
def post_predict(data: TitanicPassengers):

    df = pd.DataFrame(
        [
            {
                "Pclass": data.pclass,
                "Sex": data.sex,
                "Fare": data.fare
            }
        ]
    )

    prediction = int(predict(df)[0])
    return {'prediction': prediction}
