from pydantic import BaseModel


class TitanicPassengers(BaseModel):
    pclass: int
    sex: str
    fare: float