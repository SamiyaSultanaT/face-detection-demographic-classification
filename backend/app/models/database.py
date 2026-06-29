from sqlalchemy import Column, Integer, String, Float, DateTime, func
from backend.app.core.database import Base

class DBResultHistory(Base):
    """
    SQLAlchemy database model for prediction history logs.
    """
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String(255), nullable=False)
    predicted_gender = Column(String(50), nullable=False)
    predicted_age_group = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    prediction_time = Column(DateTime, server_default=func.now(), index=True)
