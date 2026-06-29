from pydantic import BaseModel
from datetime import datetime

class PredictionHistoryBase(BaseModel):
    image_name: str
    predicted_gender: str
    predicted_age_group: str
    confidence: float

class PredictionHistoryResponse(PredictionHistoryBase):
    id: int
    prediction_time: datetime

    class Config:
        from_attributes = True  # Pydantic v2 support for SQLAlchemy models
