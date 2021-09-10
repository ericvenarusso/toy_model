from pydantic import BaseModel


class TitanicPassengers(BaseModel):
    age: int
    sex: str
    pclass: int

    class Config:
        schema_extra = {
            "example": {
                "age": 5,
                "sex": "female",
                "pclass": 1
            }
        }