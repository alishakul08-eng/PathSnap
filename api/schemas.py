from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    trip_id: str

class ObservationPrediction(BaseModel):
    observation_id: int
    predicted_arc_id: int
    classifier_arc_id: int
    probability: float

class PredictResponse(BaseModel):
    trip_id: str
    n_observations: int
    predictions: List[ObservationPrediction]
    route_arc_ids: List[int]
